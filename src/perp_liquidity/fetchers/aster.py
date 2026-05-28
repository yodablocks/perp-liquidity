"""
Aster perpetual DEX client.

API reference:
- https://fapi.asterdex.com  (Binance-style fapi)

Endpoints used:
- GET /fapi/v1/depth?symbol=BTCUSDT&limit=50      -> order book
- GET /fapi/v1/premiumIndex?symbol=BTCUSDT        -> funding rate + mark price
- GET /fapi/v1/openInterest?symbol=BTCUSDT        -> OI in base asset

Venue notes:
- Symbol convention: {TOKEN}USDT (e.g. BTCUSDT), no separator, no -PERP suffix.
- Funding period is 8 hours (confirmed from fundingTime timestamp deltas).
  The architecture note guessed 1h -- that was wrong.
- openInterest is in base asset (BTC). We fetch markPrice from premiumIndex
  to compute oi_usd.
- Liquidations are NOT available. The REST /fapi/v1/allForceOrders endpoint
  returns HTTP 400 "The endpoint has been out of maintenance". The WebSocket
  forceOrder stream accepts subscriptions but emits no events. Both confirmed
  by live probing on 2026-05-27.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from .base import (
    Coverage,
    FundingRate,
    Liquidation,
    OpenInterest,
    OrderBook,
    OrderBookLevel,
    PerpDEXClient,
    TokenNotListed,
    VenueUnavailable,
)


FUNDING_PERIOD_HOURS = 8.0
HOURS_PER_YEAR = 8760

API_BASE = "https://fapi.asterdex.com"

# premiumIndex returns both funding rate and mark price. Cache it briefly so
# get_funding_rate + get_open_interest back-to-back share one HTTP call.
PREMIUM_CACHE_TTL_SECONDS = 2.0


class AsterClient(PerpDEXClient):
    VENUE = "aster"
    COVERAGE = {
        "orderbook": Coverage.REST,
        "funding": Coverage.REST,
        "open_interest": Coverage.REST,
        # REST allForceOrders: HTTP 400 "out of maintenance" as of 2026-05-27.
        # WS forceOrder stream: accepts subscription, emits nothing.
        "liquidations": Coverage.NOT_AVAILABLE,
    }

    def __init__(self, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        self._premium_cache: dict[str, tuple[datetime, dict]] = {}
        self._premium_lock = asyncio.Lock()

    @staticmethod
    def _symbol(token: str) -> str:
        """'BTC' -> 'BTCUSDT'"""
        return f"{token.upper()}USDT"

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    async def get_orderbook(self, token: str) -> OrderBook:
        symbol = self._symbol(token)
        try:
            resp = await self.http.get(
                f"{API_BASE}/fapi/v1/depth",
                params={"symbol": symbol, "limit": 50},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise TokenNotListed(self.VENUE, f"{token} not listed") from e
            raise VenueUnavailable(self.VENUE, f"depth HTTP error: {e}") from e
        except httpx.HTTPError as e:
            raise VenueUnavailable(self.VENUE, f"depth request failed: {e}") from e

        bids_raw = data.get("bids") or []
        asks_raw = data.get("asks") or []
        if not bids_raw or not asks_raw:
            raise TokenNotListed(self.VENUE, f"{token} returned empty orderbook")

        bids = [OrderBookLevel(price=float(lvl[0]), qty=float(lvl[1])) for lvl in bids_raw]
        asks = [OrderBookLevel(price=float(lvl[0]), qty=float(lvl[1])) for lvl in asks_raw]

        bids.sort(key=lambda b: b.price, reverse=True)
        asks.sort(key=lambda a: a.price)

        return OrderBook(venue=self.VENUE, token=token.upper(), bids=bids, asks=asks)

    # ------------------------------------------------------------------
    # Shared premiumIndex (funding + mark price)
    # ------------------------------------------------------------------

    async def _get_premium_index(self, token: str) -> dict:
        """Fetch premiumIndex for token, with a short TTL cache."""
        symbol = self._symbol(token)

        async with self._premium_lock:
            now = datetime.now(timezone.utc)
            cached = self._premium_cache.get(symbol)
            if (
                cached is not None
                and (now - cached[0]).total_seconds() < PREMIUM_CACHE_TTL_SECONDS
            ):
                return cached[1]

            try:
                resp = await self.http.get(
                    f"{API_BASE}/fapi/v1/premiumIndex",
                    params={"symbol": symbol},
                    timeout=self.TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 404):
                    raise TokenNotListed(self.VENUE, f"{token} not listed") from e
                raise VenueUnavailable(
                    self.VENUE, f"premiumIndex HTTP error: {e}"
                ) from e
            except httpx.HTTPError as e:
                raise VenueUnavailable(
                    self.VENUE, f"premiumIndex request failed: {e}"
                ) from e

            self._premium_cache[symbol] = (now, data)
            return data

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------

    async def get_funding_rate(self, token: str) -> FundingRate:
        data = await self._get_premium_index(token)

        try:
            rate_per_period = float(data["lastFundingRate"])
            next_funding_ms = int(data["nextFundingTime"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"missing or invalid funding field: {e}"
            ) from e

        apr = rate_per_period * (HOURS_PER_YEAR / FUNDING_PERIOD_HOURS)
        next_funding_at = datetime.fromtimestamp(next_funding_ms / 1000, tz=timezone.utc)

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
        symbol = self._symbol(token)

        try:
            resp = await self.http.get(
                f"{API_BASE}/fapi/v1/openInterest",
                params={"symbol": symbol},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            oi_data = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 404):
                raise TokenNotListed(self.VENUE, f"{token} not listed") from e
            raise VenueUnavailable(self.VENUE, f"openInterest HTTP error: {e}") from e
        except httpx.HTTPError as e:
            raise VenueUnavailable(self.VENUE, f"openInterest request failed: {e}") from e

        try:
            oi_base = float(oi_data["openInterest"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(self.VENUE, f"invalid openInterest field: {e}") from e

        # Mark price comes from premiumIndex (shared cache).
        premium = await self._get_premium_index(token)
        try:
            mark_price = float(premium["markPrice"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(self.VENUE, f"invalid markPrice field: {e}") from e

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
        """Aster does not expose a public liquidation feed.

        - REST /fapi/v1/allForceOrders returns HTTP 400 "The endpoint has been
          out of maintenance" as of 2026-05-27.
        - The WebSocket forceOrder stream (wss://fstream.asterdex.com/ws) accepts
          subscriptions without error but emits no events.

        Both confirmed by live probing. This is a venue limitation, not an
        implementation gap.
        """
        raise NotImplementedError(
            "Aster does not expose public liquidations. "
            "Coverage.NOT_AVAILABLE; see docstring for details."
        )
