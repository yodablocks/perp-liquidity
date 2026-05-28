"""
Base interface for perpetual DEX clients.

Every venue fetcher implements PerpDEXClient. The Coverage enum declares
which data dimensions each venue actually supports; methods that are not
implemented raise NotImplementedError with a documented reason rather than
returning fake data.

Design notes:
- All IO is async (httpx.AsyncClient). The CLI wraps with asyncio.run.
- Returned dataclasses are frozen and timestamped at fetch time, so downstream
  analyzers can reason about staleness.
- Fees live in config/fees.yaml, never hardcoded in fetchers.
- Liquidations have two modes: REST (historical snapshot) and WS_TAIL (live
  subscription for N seconds). The Coverage enum distinguishes them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar

import httpx


# ============================================================================
# Coverage declaration
# ============================================================================


class Coverage(Enum):
    """How a venue exposes a given data dimension.

    Used in PerpDEXClient.COVERAGE to declare upfront what works and why.
    Honesty over breadth: NOT_AVAILABLE is a valid, documented answer.
    """

    REST = "rest"  # Standard REST endpoint, returns a snapshot
    WS_TAIL = "ws_tail"  # Live websocket subscription, tail for N seconds
    NOT_AVAILABLE = "not_available"  # Venue does not expose this publicly
    NOT_IMPLEMENTED = "not_implemented"  # Could exist, not built yet


# ============================================================================
# Return types
# ============================================================================


@dataclass(frozen=True)
class OrderBookLevel:
    """One price level in an order book."""

    price: float
    qty: float  # Base asset quantity


@dataclass(frozen=True)
class OrderBook:
    """Snapshot of a venue's order book for one instrument.

    bids are sorted descending by price (best bid first).
    asks are sorted ascending by price (best ask first).
    Fetchers are responsible for enforcing this invariant.
    """

    venue: str
    token: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Venue-specific extras (e.g. Paradex RPI spread) live here. Analyzers
    # that don't recognize a key should ignore it, not crash.
    extras: dict = field(default_factory=dict)

    @property
    def mid_price(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2

    @property
    def spread_bps(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        best_bid = self.bids[0].price
        best_ask = self.asks[0].price
        return ((best_ask - best_bid) / best_bid) * 10_000


@dataclass(frozen=True)
class FundingRate:
    """Current funding rate snapshot.

    Funding mechanics differ wildly across venues. We normalize to two things:
    - rate_per_period: the raw rate charged per funding interval
    - apr_annualized: that rate expressed as APR for cross-venue comparison

    Annualization uses period_hours: APR = rate_per_period * (8760 / period_hours)
    """

    venue: str
    token: str
    rate_per_period: float  # e.g. 0.0001 = 1 bp per period
    period_hours: float  # 8.0 for Hyperliquid, 1.0 for Aster, etc.
    apr_annualized: float  # rate_per_period * (8760 / period_hours)
    next_funding_at: datetime | None = None
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass(frozen=True)
class OpenInterest:
    """Open interest snapshot, normalized to USD.

    Venues report OI in different units (contracts, base asset, USD). The
    fetcher normalizes to USD using the current mark price, so cross-venue
    comparison is meaningful.
    """

    venue: str
    token: str
    oi_base: float  # In base asset units (e.g. BTC)
    oi_usd: float  # oi_base * mark_price
    mark_price: float
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass(frozen=True)
class Liquidation:
    """A single liquidation event.

    side='long' means a long position was liquidated (a forced sell).
    side='short' means a short was liquidated (a forced buy).
    """

    venue: str
    token: str
    side: str  # 'long' or 'short'
    price: float
    qty_base: float
    qty_usd: float
    occurred_at: datetime


# ============================================================================
# Errors
# ============================================================================


class FetcherError(Exception):
    """Base class for fetcher errors. Catch this to handle any venue failure."""

    def __init__(self, venue: str, message: str):
        self.venue = venue
        self.message = message
        super().__init__(f"[{venue}] {message}")


class VenueUnavailable(FetcherError):
    """Venue returned an error response, timeout, or unparseable data."""


class TokenNotListed(FetcherError):
    """The token is not traded on this venue."""


# ============================================================================
# Abstract base client
# ============================================================================


class PerpDEXClient(ABC):
    """Abstract interface every venue fetcher must implement.

    Subclasses declare their coverage via the COVERAGE class attribute. Any
    method whose coverage is NOT_AVAILABLE or NOT_IMPLEMENTED should raise
    NotImplementedError when called, with a docstring explaining why.

    Example:
        class HyperliquidClient(PerpDEXClient):
            VENUE = "hyperliquid"
            COVERAGE = {
                "orderbook": Coverage.REST,
                "funding": Coverage.REST,
                "open_interest": Coverage.REST,
                "liquidations": Coverage.WS_TAIL,
            }

            async def get_orderbook(self, token): ...
    """

    # Subclasses must set these.
    VENUE: ClassVar[str]
    COVERAGE: ClassVar[dict[str, Coverage]]

    # Default HTTP timeout in seconds. Override per-venue if needed.
    TIMEOUT: ClassVar[float] = 10.0

    def __init__(self, client: httpx.AsyncClient | None = None):
        """Optionally accept a shared httpx client for connection pooling."""
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> PerpDEXClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.TIMEOUT)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def http(self) -> httpx.AsyncClient:
        """Internal accessor that raises a clear error if used outside `async with`."""
        if self._client is None:
            raise RuntimeError(
                f"{self.VENUE} client used outside `async with`. "
                f"Use: `async with {type(self).__name__}() as c: ...`"
            )
        return self._client

    # --------------------------------------------------------------------
    # Coverage helpers
    # --------------------------------------------------------------------

    @classmethod
    def supports(cls, dimension: str) -> bool:
        """Return True if this venue exposes the given dimension."""
        cov = cls.COVERAGE.get(dimension, Coverage.NOT_IMPLEMENTED)
        return cov in (Coverage.REST, Coverage.WS_TAIL)

    @classmethod
    def coverage_note(cls, dimension: str) -> str:
        """Human-readable note for this venue/dimension pair."""
        cov = cls.COVERAGE.get(dimension, Coverage.NOT_IMPLEMENTED)
        return f"{cls.VENUE}.{dimension}: {cov.value}"

    # --------------------------------------------------------------------
    # Data methods
    # --------------------------------------------------------------------

    @abstractmethod
    async def get_orderbook(self, token: str) -> OrderBook:
        """Fetch a snapshot of the order book for the given token.

        Raises:
            TokenNotListed: token does not trade on this venue
            VenueUnavailable: API error, timeout, or malformed response
        """
        ...

    @abstractmethod
    async def get_funding_rate(self, token: str) -> FundingRate:
        """Fetch the current funding rate and annualize it.

        Raises:
            NotImplementedError: if COVERAGE["funding"] is NOT_AVAILABLE
            TokenNotListed, VenueUnavailable: same as get_orderbook
        """
        ...

    @abstractmethod
    async def get_open_interest(self, token: str) -> OpenInterest:
        """Fetch open interest, normalized to USD via current mark price.

        Raises:
            NotImplementedError: if COVERAGE["open_interest"] is NOT_AVAILABLE
            TokenNotListed, VenueUnavailable: same as get_orderbook
        """
        ...

    @abstractmethod
    async def get_recent_liquidations(
        self, token: str, *, lookback_seconds: int = 60
    ) -> list[Liquidation]:
        """Fetch recent liquidation events.

        For venues with Coverage.REST liquidations, returns events from the
        last `lookback_seconds`. For Coverage.WS_TAIL, subscribes and tails
        the live stream for `lookback_seconds` then returns what was captured.

        Returns an empty list if no liquidations occurred in the window;
        raises only on actual failure.

        Raises:
            NotImplementedError: if COVERAGE["liquidations"] is NOT_AVAILABLE
            TokenNotListed, VenueUnavailable: same as get_orderbook
        """
        ...
