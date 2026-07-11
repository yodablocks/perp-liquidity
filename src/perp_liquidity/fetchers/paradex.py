"""
Paradex perpetual DEX client.

API reference:
- https://docs.paradex.trade/api/general-information
- https://docs.paradex.trade/risk/funding-mechanism

Endpoints used:
- GET /v1/orderbook/{market}/interactive?depth=N -> order book + RPI/API best prices
- GET /v1/markets/summary?market={market}        -> funding + OI + mark price

Venue notes:
- Funding period is 8 hours (Binance convention), accrued continuously.
- Open interest in markets/summary is denominated in USD on Paradex (not base asset
  like Hyperliquid). We detect this heuristically and normalize.
- Public liquidations are NOT exposed: the /v1/liquidations endpoint requires
  authentication and returns the *user's own* liquidation history. There is no
  public market-wide liquidation feed via REST or WebSocket. We declare this
  as Coverage.NOT_AVAILABLE rather than ship a fake feed.

Bonus signal preserved from v2:
- The interactive orderbook endpoint returns both the "API" best bid/ask
  (visible to API takers) and the "RPI" best bid/ask (visible only to retail
  interactive flow). The spread between them is a unique Paradex microstructure
  signal worth capturing in OrderBook.extras["rpi_data"].
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

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

API_BASE = "https://api.prod.paradex.trade"

# Same TTL pattern as Hyperliquid: funding + OI come from the same endpoint,
# so back-to-back calls should share one HTTP request.
SUMMARY_CACHE_TTL_SECONDS = 2.0

# Sanity threshold for detecting whether OI is USD-denominated or base-denominated.
# If the value-divided-by-mark-price produces a base quantity larger than this,
# we assume the original value was already in USD.
OI_BASE_SANITY_CEILING = 1_000_000.0  # 1M BTC of OI is impossible; flag as USD


class ParadexClient(PerpDEXClient):
    VENUE = "paradex"
    COVERAGE = {
        "orderbook": Coverage.REST,
        "funding": Coverage.REST,
        "open_interest": Coverage.REST,
        # Public per-market liquidation feed does not exist. The /v1/liquidations
        # REST endpoint requires JWT auth and only returns the calling user's
        # own liquidations.
        "liquidations": Coverage.NOT_AVAILABLE,
    }

    def __init__(self, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        self._summary_cache: dict[str, tuple[datetime, dict]] = {}
        self._summary_lock = asyncio.Lock()

    @staticmethod
    def _market_symbol(token: str) -> str:
        """Token like 'BTC' -> 'BTC-USD-PERP' (Paradex market convention)."""
        return f"{token.upper()}-USD-PERP"

    # ------------------------------------------------------------------
    # Order book (with RPI extras)
    # ------------------------------------------------------------------

    async def get_orderbook(self, token: str) -> OrderBook:
        market = self._market_symbol(token)

        try:
            resp = await self.http.get(
                f"{API_BASE}/v1/orderbook/{market}/interactive",
                params={"depth": 50},
                headers={"Accept": "application/json"},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise TokenNotListed(self.VENUE, f"{token} not listed") from e
            raise VenueUnavailable(self.VENUE, f"orderbook HTTP error: {e}") from e
        except httpx.HTTPError as e:
            raise VenueUnavailable(self.VENUE, f"orderbook request failed: {e}") from e

        if not data.get("bids") or not data.get("asks"):
            raise TokenNotListed(self.VENUE, f"{token} returned empty orderbook")

        bids = [
            OrderBookLevel(price=float(lvl[0]), qty=float(lvl[1]))
            for lvl in data["bids"]
        ]
        asks = [
            OrderBookLevel(price=float(lvl[0]), qty=float(lvl[1]))
            for lvl in data["asks"]
        ]

        bids.sort(key=lambda b: b.price, reverse=True)
        asks.sort(key=lambda a: a.price)

        # Capture the Paradex-specific RPI vs API spread divergence.
        # This is real microstructure signal — it tells you the gap between
        # what retail interactive flow sees and what API takers see.
        extras: dict = {}
        if data.get("best_bid_interactive") and data.get("best_ask_interactive"):
            try:
                rpi_bid = float(data["best_bid_interactive"][0])
                rpi_ask = float(data["best_ask_interactive"][0])
                rpi_spread_bps = ((rpi_ask - rpi_bid) / rpi_bid) * 10_000

                api_bid_raw = data.get("best_bid_api")
                api_ask_raw = data.get("best_ask_api")
                api_bid = float(api_bid_raw[0]) if api_bid_raw else None
                api_ask = float(api_ask_raw[0]) if api_ask_raw else None
                api_spread_bps = (
                    ((api_ask - api_bid) / api_bid) * 10_000
                    if api_bid and api_ask
                    else None
                )

                extras["rpi_data"] = {
                    "api_bid": api_bid,
                    "api_ask": api_ask,
                    "api_spread_bps": api_spread_bps,
                    "rpi_bid": rpi_bid,
                    "rpi_ask": rpi_ask,
                    "rpi_spread_bps": rpi_spread_bps,
                }
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                # RPI data is a bonus, not load-bearing. Don't fail the
                # whole fetch if the optional fields are malformed.
                pass

        return OrderBook(
            venue=self.VENUE,
            token=token.upper(),
            bids=bids,
            asks=asks,
            extras=extras,
        )

    # ------------------------------------------------------------------
    # Shared summary endpoint (funding + OI + mark price)
    # ------------------------------------------------------------------

    async def _get_market_summary(self, token: str) -> dict:
        """Return the markets/summary entry for the given token, with TTL cache."""
        market = self._market_symbol(token)

        async with self._summary_lock:
            now = datetime.now(timezone.utc)
            cached = self._summary_cache.get(market)
            if (
                cached is not None
                and (now - cached[0]).total_seconds() < SUMMARY_CACHE_TTL_SECONDS
            ):
                return cached[1]

            try:
                resp = await self.http.get(
                    f"{API_BASE}/v1/markets/summary",
                    params={"market": market},
                    headers={"Accept": "application/json"},
                    timeout=self.TIMEOUT,
                )
                resp.raise_for_status()
                payload = resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise TokenNotListed(self.VENUE, f"{token} not listed") from e
                raise VenueUnavailable(
                    self.VENUE, f"markets/summary HTTP error: {e}"
                ) from e
            except httpx.HTTPError as e:
                raise VenueUnavailable(
                    self.VENUE, f"markets/summary request failed: {e}"
                ) from e

            results = payload.get("results") or []
            if not results:
                raise TokenNotListed(self.VENUE, f"{token} returned no summary")

            summary = results[0]
            self._summary_cache[market] = (now, summary)
            return summary

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------

    async def get_funding_rate(self, token: str) -> FundingRate:
        summary = await self._get_market_summary(token)

        try:
            rate_per_period = float(summary["funding_rate"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"missing or invalid funding_rate field: {e}"
            ) from e

        apr = rate_per_period * (HOURS_PER_YEAR / FUNDING_PERIOD_HOURS)

        # Paradex funding is continuously accrued, but the next "boundary" for
        # comparison purposes is the next 8h UTC mark (00, 08, 16).
        now = datetime.now(timezone.utc)
        next_hour_boundary = ((now.hour // 8) + 1) * 8
        days_ahead = 0
        if next_hour_boundary >= 24:
            next_hour_boundary -= 24
            days_ahead = 1
        next_funding = (now + timedelta(days=days_ahead)).replace(
            hour=next_hour_boundary, minute=0, second=0, microsecond=0
        )

        return FundingRate(
            venue=self.VENUE,
            token=token.upper(),
            rate_per_period=rate_per_period,
            period_hours=FUNDING_PERIOD_HOURS,
            apr_annualized=apr,
            next_funding_at=next_funding,
        )

    # ------------------------------------------------------------------
    # Open interest
    # ------------------------------------------------------------------

    async def get_open_interest(self, token: str) -> OpenInterest:
        summary = await self._get_market_summary(token)

        try:
            oi_raw = float(summary["open_interest"])
            mark_price = float(summary["mark_price"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"missing or invalid OI/mark_price field: {e}"
            ) from e

        # Paradex docs are ambiguous about OI denomination. Heuristic: if
        # treating the raw value as base-asset produces an implausibly large
        # quantity, assume it was actually USD-denominated and reverse.
        # Empirically Paradex appears to report OI in base asset (contracts),
        # but we defensively normalize.
        if mark_price > 0:
            implied_base = oi_raw  # Assume base first
            implied_usd = oi_raw * mark_price
            if implied_base > OI_BASE_SANITY_CEILING:
                # Value was almost certainly already USD
                oi_usd = oi_raw
                oi_base = oi_raw / mark_price
            else:
                oi_base = implied_base
                oi_usd = implied_usd
        else:
            raise VenueUnavailable(self.VENUE, "mark_price is zero or negative")

        return OpenInterest(
            venue=self.VENUE,
            token=token.upper(),
            oi_base=oi_base,
            oi_usd=oi_usd,
            mark_price=mark_price,
        )

    # ------------------------------------------------------------------
    # Liquidations: NOT AVAILABLE
    # ------------------------------------------------------------------

    async def get_recent_liquidations(
        self, token: str, *, lookback_seconds: int = 60
    ) -> list[Liquidation]:
        """Paradex does not expose a public per-market liquidation feed.

        The /v1/liquidations REST endpoint requires JWT authentication and only
        returns the calling user's own liquidation history. There is no public
        WebSocket channel for market-wide liquidation events as of this writing.

        This is a real venue limitation, not an implementation gap. To get
        Paradex liquidation data you would need to either:
        - Run a Paradex Chain indexer reading on-chain liquidation events
        - Subscribe to a third-party data provider (Amberdata, Kaiko)
        """
        raise NotImplementedError(
            "Paradex does not expose public liquidations. "
            "Coverage.NOT_AVAILABLE; see docstring for details."
        )
