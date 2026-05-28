"""
GRVT perpetual DEX client.

API reference:
- Base: https://market-data.grvt.io/full/v1

Endpoints used:
- POST /book    {"instrument": "BTC_USDT_Perp", "depth": 10}  -> order book
- POST /ticker  {"instrument": "BTC_USDT_Perp"}               -> funding + OI + mark price

Venue notes:
- Instrument name: {TOKEN}_USDT_Perp (e.g. BTC_USDT_Perp).
- All endpoints are POST with JSON body (confirmed 2026-05-28; GET /ticker returns 405).
- Funding period is 8 hours (funding_interval_hours=8 from /instrument endpoint;
  confirmed by funding_time timestamp deltas: 1779955200000000000 - 1779926400000000000
  = 28800s = 8h).
- All timestamps are nanoseconds. Convert to seconds via / 1e9.
- Ticker returns funding_rate (per 8h period), next_funding_time (ns),
  open_interest (base asset, BTC), and mark_price in a single call. TTL-cached 2s.
- funding_rate=0.01 at probe time = 1% per 8h = 1095% APR. This is the cap
  (adjusted_funding_rate_cap=0.3 in instrument metadata for BTC, but 1.0 in practice).
  High rate reflects low liquidity / bootstrapping phase.
- Liquidations: NOT_AVAILABLE. Trade messages from POST /trade have no is_liquidation
  flag. No dedicated liquidation endpoint exists. Confirmed 2026-05-28.
- Live OI probe (2026-05-28): 2683 BTC ~ $197M. Smallest in the panel.
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

API_BASE = "https://market-data.grvt.io/full/v1"

# Ticker returns funding + OI + markPrice in one call. Cache briefly so
# get_funding_rate + get_open_interest back-to-back share one HTTP call.
TICKER_CACHE_TTL_SECONDS = 2.0


class GRVTClient(PerpDEXClient):
    VENUE = "grvt"
    COVERAGE = {
        "orderbook": Coverage.REST,
        "funding": Coverage.REST,
        "open_interest": Coverage.REST,
        # Trade messages have no is_liquidation flag; no public liq endpoint.
        # Confirmed by probing POST /trade on 2026-05-28.
        "liquidations": Coverage.NOT_AVAILABLE,
    }

    def __init__(self, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        # ticker cache: instrument -> (fetched_at, result_dict)
        self._ticker_cache: dict[str, tuple[datetime, dict]] = {}
        self._ticker_lock = asyncio.Lock()

    @staticmethod
    def _instrument(token: str) -> str:
        """'BTC' -> 'BTC_USDT_Perp'"""
        return f"{token.upper()}_USDT_Perp"

    # ------------------------------------------------------------------
    # Shared ticker (funding + OI + markPrice in one call)
    # ------------------------------------------------------------------

    async def _get_ticker(self, token: str) -> dict:
        """Fetch POST /ticker for token, with a short TTL cache."""
        instrument = self._instrument(token)

        async with self._ticker_lock:
            now = datetime.now(timezone.utc)
            cached = self._ticker_cache.get(instrument)
            if (
                cached is not None
                and (now - cached[0]).total_seconds() < TICKER_CACHE_TTL_SECONDS
            ):
                return cached[1]

            try:
                resp = await self.http.post(
                    f"{API_BASE}/ticker",
                    json={"instrument": instrument},
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
                result = data["result"]
            except (KeyError, TypeError) as e:
                raise VenueUnavailable(
                    self.VENUE, f"unexpected ticker shape: {e}"
                ) from e

            self._ticker_cache[instrument] = (now, result)
            return result

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    async def get_orderbook(self, token: str) -> OrderBook:
        instrument = self._instrument(token)

        try:
            resp = await self.http.post(
                f"{API_BASE}/book",
                json={"instrument": instrument, "depth": 10},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 404):
                raise TokenNotListed(self.VENUE, f"{token} not listed") from e
            raise VenueUnavailable(self.VENUE, f"book HTTP error: {e}") from e
        except httpx.HTTPError as e:
            raise VenueUnavailable(self.VENUE, f"book request failed: {e}") from e

        try:
            result = data["result"]
            bids_raw = result.get("bids") or []
            asks_raw = result.get("asks") or []
        except (KeyError, TypeError) as e:
            raise VenueUnavailable(self.VENUE, f"unexpected book shape: {e}") from e

        if not bids_raw or not asks_raw:
            raise TokenNotListed(self.VENUE, f"{token} returned empty orderbook")

        bids = [OrderBookLevel(price=float(lvl["price"]), qty=float(lvl["size"])) for lvl in bids_raw]
        asks = [OrderBookLevel(price=float(lvl["price"]), qty=float(lvl["size"])) for lvl in asks_raw]

        bids.sort(key=lambda b: b.price, reverse=True)
        asks.sort(key=lambda a: a.price)

        return OrderBook(venue=self.VENUE, token=token.upper(), bids=bids, asks=asks)

    # ------------------------------------------------------------------
    # Funding rate
    # ------------------------------------------------------------------

    async def get_funding_rate(self, token: str) -> FundingRate:
        ticker = await self._get_ticker(token)

        try:
            rate_per_period = float(ticker["funding_rate"])
            next_funding_ns = int(ticker["next_funding_time"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"missing or invalid funding field: {e}"
            ) from e

        apr = rate_per_period * (HOURS_PER_YEAR / FUNDING_PERIOD_HOURS)
        next_funding_at = datetime.fromtimestamp(next_funding_ns / 1e9, tz=timezone.utc)

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
            oi_base = float(ticker["open_interest"])
            mark_price = float(ticker["mark_price"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"missing or invalid OI/mark_price field: {e}"
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
        """GRVT does not expose a public liquidation feed.

        POST /trade returns recent trades but fields are:
        event_time, instrument, is_taker_buyer, size, price, mark_price,
        index_price, interest_rate, forward_price, trade_id, venue, is_rpi.
        No is_liquidation flag exists. No dedicated liquidation endpoint.
        Confirmed by live probing on 2026-05-28.
        """
        raise NotImplementedError(
            "GRVT does not expose public liquidations. "
            "Coverage.NOT_AVAILABLE; see docstring for details."
        )
