"""
ApeX Omni perpetual DEX client.

API reference:
- https://api-docs.omni.apex.exchange (note: geo-blocked in France; use Google DNS to access)
- SDK: https://github.com/ApeX-Protocol/apexpro-openapi
- Base: https://omni.apex.exchange

Endpoints used:
- GET /api/v3/depth?symbol=BTCUSDT        -> order book
- GET /api/v3/ticker?symbol=BTCUSDT       -> funding rate + OI + mark price

Venue notes:
- Symbol convention: {TOKEN}USDT for depth/ticker (e.g. BTCUSDT).
  history-funding uses {TOKEN}-USDT with hyphen, but we use ticker for current rate.
- Funding period is 1 hour (confirmed from fundingTime timestamp deltas: 3600s).
  This was unexpected -- Bybit-style APIs typically use 8h, but ApeX uses 1h.
- Depth response: "a" = asks, "b" = bids (both as [price, size] string arrays).
- Ticker response: data is a list; openInterest is in base asset (BTC);
  nextFundingTime is an ISO 8601 string (e.g. "2026-05-28T10:00:00Z").
- Liquidations are NOT_AVAILABLE. The only liquidation feed is the private
  account WS stream (ws_zk_accounts_v3), which requires API key authentication.
  No public liquidation endpoint or stream exists. Confirmed from SDK source 2026-05-28.
- API shapes confirmed from live probing 2026-05-28 (required /etc/hosts workaround
  for French ISP DNS block on omni.apex.exchange).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from .base import (
    Coverage,
    DNSOverrideTransport,
    FundingRate,
    Liquidation,
    OpenInterest,
    OrderBook,
    OrderBookLevel,
    PerpDEXClient,
    TokenNotListed,
    VenueUnavailable,
)


FUNDING_PERIOD_HOURS = 1.0
HOURS_PER_YEAR = 8760

API_BASE = "https://omni.apex.exchange"

# Ticker returns funding + OI + markPrice in one call. Cache briefly so
# get_funding_rate + get_open_interest back-to-back share one HTTP call.
TICKER_CACHE_TTL_SECONDS = 2.0


class ApeXClient(PerpDEXClient):
    VENUE = "apex"
    COVERAGE = {
        "orderbook": Coverage.REST,
        "funding": Coverage.REST,
        "open_interest": Coverage.REST,
        # Liquidations only available on private account WS (ws_zk_accounts_v3).
        # Requires API key auth. No public liquidation feed. Confirmed 2026-05-28.
        "liquidations": Coverage.NOT_AVAILABLE,
    }

    # omni.apex.exchange resolves to 127.0.0.1 on some ISPs (confirmed French
    # residential). Real IP confirmed via Google DNS 2026-05-28. DNSOverrideTransport
    # routes TCP to the real IP while preserving TLS SNI and Host header.
    DNS_OVERRIDES = {"omni.apex.exchange": "157.185.129.119"}

    def __init__(self, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        # ticker cache: symbol -> (fetched_at, data_dict)
        self._ticker_cache: dict[str, tuple[datetime, dict]] = {}
        self._ticker_lock = asyncio.Lock()

    async def __aenter__(self) -> "ApeXClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                transport=DNSOverrideTransport(self.DNS_OVERRIDES),
                timeout=self.TIMEOUT,
            )
        return self  # type: ignore[return-value]

    @staticmethod
    def _symbol(token: str) -> str:
        """'BTC' -> 'BTCUSDT'"""
        return f"{token.upper()}USDT"

    # ------------------------------------------------------------------
    # Shared ticker (funding + OI + markPrice in one call)
    # ------------------------------------------------------------------

    async def _get_ticker(self, token: str) -> dict:
        """Fetch /api/v3/ticker for token, with a short TTL cache."""
        symbol = self._symbol(token)

        async with self._ticker_lock:
            now = datetime.now(timezone.utc)
            cached = self._ticker_cache.get(symbol)
            if (
                cached is not None
                and (now - cached[0]).total_seconds() < TICKER_CACHE_TTL_SECONDS
            ):
                return cached[1]

            try:
                resp = await self.http.get(
                    f"{API_BASE}/api/v3/ticker",
                    params={"symbol": symbol},
                    timeout=self.TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 404):
                    raise TokenNotListed(self.VENUE, f"{token} not listed") from e
                raise VenueUnavailable(
                    self.VENUE, f"ticker HTTP error: {e}"
                ) from e
            except httpx.HTTPError as e:
                raise VenueUnavailable(
                    self.VENUE, f"ticker request failed: {e}"
                ) from e

            try:
                ticker = data["data"][0]
            except (KeyError, IndexError, TypeError) as e:
                raise VenueUnavailable(
                    self.VENUE, f"unexpected ticker shape: {e}"
                ) from e

            self._ticker_cache[symbol] = (now, ticker)
            return ticker

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    async def get_orderbook(self, token: str) -> OrderBook:
        symbol = self._symbol(token)

        try:
            resp = await self.http.get(
                f"{API_BASE}/api/v3/depth",
                params={"symbol": symbol},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 404):
                raise TokenNotListed(self.VENUE, f"{token} not listed") from e
            raise VenueUnavailable(self.VENUE, f"depth HTTP error: {e}") from e
        except httpx.HTTPError as e:
            raise VenueUnavailable(self.VENUE, f"depth request failed: {e}") from e

        try:
            book = data["data"]
            bids_raw = book.get("b") or []
            asks_raw = book.get("a") or []
        except (KeyError, TypeError) as e:
            raise VenueUnavailable(self.VENUE, f"unexpected depth shape: {e}") from e

        if not bids_raw or not asks_raw:
            raise TokenNotListed(self.VENUE, f"{token} returned empty orderbook")

        bids = [OrderBookLevel(price=float(lvl[0]), qty=float(lvl[1])) for lvl in bids_raw]
        asks = [OrderBookLevel(price=float(lvl[0]), qty=float(lvl[1])) for lvl in asks_raw]

        bids.sort(key=lambda b: b.price, reverse=True)
        asks.sort(key=lambda a: a.price)

        return OrderBook(venue=self.VENUE, token=token.upper(), bids=bids, asks=asks)

    # ------------------------------------------------------------------
    # Funding rate
    # ------------------------------------------------------------------

    async def get_funding_rate(self, token: str) -> FundingRate:
        ticker = await self._get_ticker(token)

        try:
            rate_per_period = float(ticker["fundingRate"])
            next_funding_str = ticker["nextFundingTime"]
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"missing or invalid funding field: {e}"
            ) from e

        apr = rate_per_period * (HOURS_PER_YEAR / FUNDING_PERIOD_HOURS)

        # nextFundingTime is an ISO 8601 string: "2026-05-28T10:00:00Z"
        try:
            next_funding_at = datetime.fromisoformat(
                next_funding_str.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError) as e:
            raise VenueUnavailable(
                self.VENUE, f"invalid nextFundingTime format: {e}"
            ) from e

        return FundingRate(
            venue=self.VENUE,
            token=token.upper(),
            rate_per_period=rate_per_period,
            period_hours=FUNDING_PERIOD_HOURS,
            apr_annualized=apr,
            next_funding_at=next_funding_at,
        )

    # ------------------------------------------------------------------
    # Open interest
    # ------------------------------------------------------------------

    async def get_open_interest(self, token: str) -> OpenInterest:
        ticker = await self._get_ticker(token)

        try:
            oi_base = float(ticker["openInterest"])
            mark_price = float(ticker["markPrice"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"missing or invalid OI/markPrice field: {e}"
            ) from e

        return OpenInterest(
            venue=self.VENUE,
            token=token.upper(),
            oi_base=oi_base,
            oi_usd=oi_base * mark_price,
            mark_price=mark_price,
        )

    # ------------------------------------------------------------------
    # Liquidations: NOT AVAILABLE
    # ------------------------------------------------------------------

    async def get_recent_liquidations(
        self, token: str, *, lookback_seconds: int = 60
    ) -> list[Liquidation]:
        """ApeX liquidation events are only available on the private account WS stream.

        The ws_zk_accounts_v3 channel requires API key authentication (key/secret/
        passphrase from registration). No public liquidation endpoint or stream exists.
        Confirmed from SDK source (websocket_api.py) on 2026-05-28.
        """
        raise NotImplementedError(
            "ApeX liquidations require private WS authentication. "
            "Coverage.NOT_AVAILABLE; see docstring for details."
        )
