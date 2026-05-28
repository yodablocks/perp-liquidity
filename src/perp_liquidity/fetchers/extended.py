"""
Extended Exchange perpetual DEX client.

API reference:
- https://api.docs.extended.exchange

Endpoints used:
- GET /api/v1/info/markets/{market}/orderbook  -> order book
- GET /api/v1/info/markets                     -> all markets (funding + OI + markPrice per market)
- WS  wss://api.starknet.extended.exchange/stream.extended.exchange/v1/publicTrades/{market}
      -> public trades stream; tT field is "TRADE", "LIQUIDATION", or "DELEVERAGE"

Venue notes:
- Market name convention: {TOKEN}-USD (e.g. BTC-USD).
- Funding period is 1 hour (docs: "calculated every minute, applied once per hour").
- The markets list endpoint returns all venues in one call with fundingRate,
  nextFundingRate (ms epoch), openInterest (USD), openInterestBase (base asset),
  and markPrice all embedded in marketStats. We TTL-cache this for 2s so
  get_funding_rate + get_open_interest share one HTTP call.
- OI: we use openInterestBase * markPrice for consistency with other venues.
  openInterest (USD) is also available as a cross-check.
- Liquidations are available via the public WS trades stream. The REST trades
  endpoint only returns the last 50 trades with no pagination, so liquidations
  may not appear in quiet periods -- the WS stream is the correct path.
- WS side convention: S="SELL" on LIQUIDATION = long force-closed (liq_side="long");
  S="BUY" on LIQUIDATION = short force-closed (liq_side="short").
- The User-Agent header is required per the Extended API docs.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import httpx

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore

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


FUNDING_PERIOD_HOURS = 1.0
HOURS_PER_YEAR = 8760

API_BASE = "https://api.starknet.extended.exchange"
WS_BASE = "wss://api.starknet.extended.exchange/stream.extended.exchange/v1"

USER_AGENT = "perp-liquidity/0.1"
MARKETS_CACHE_TTL_SECONDS = 2.0


class ExtendedClient(PerpDEXClient):
    VENUE = "extended"
    COVERAGE = {
        "orderbook": Coverage.REST,
        "funding": Coverage.REST,
        "open_interest": Coverage.REST,
        # Public WS trades stream emits tT="LIQUIDATION" events.
        # REST trades endpoint caps at 50 with no pagination -- WS is correct path.
        "liquidations": Coverage.WS_TAIL,
    }

    def __init__(self, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        self._markets_cache: tuple[datetime, list] | None = None
        self._markets_lock = asyncio.Lock()

    @staticmethod
    def _market(token: str) -> str:
        """'BTC' -> 'BTC-USD'"""
        return f"{token.upper()}-USD"

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    async def get_orderbook(self, token: str) -> OrderBook:
        market = self._market(token)
        try:
            resp = await self.http.get(
                f"{API_BASE}/api/v1/info/markets/{market}/orderbook",
                headers={"User-Agent": USER_AGENT},
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

        ob_data = data.get("data", {})
        bids_raw = ob_data.get("bid") or []
        asks_raw = ob_data.get("ask") or []
        if not bids_raw or not asks_raw:
            raise TokenNotListed(self.VENUE, f"{token} returned empty orderbook")

        bids = [OrderBookLevel(price=float(lvl["price"]), qty=float(lvl["qty"])) for lvl in bids_raw]
        asks = [OrderBookLevel(price=float(lvl["price"]), qty=float(lvl["qty"])) for lvl in asks_raw]

        bids.sort(key=lambda b: b.price, reverse=True)
        asks.sort(key=lambda a: a.price)

        return OrderBook(venue=self.VENUE, token=token.upper(), bids=bids, asks=asks)

    # ------------------------------------------------------------------
    # Shared markets list (funding + OI + markPrice)
    # ------------------------------------------------------------------

    async def _get_markets(self) -> list:
        """Return all markets list with TTL cache."""
        async with self._markets_lock:
            now = datetime.now(timezone.utc)
            if (
                self._markets_cache is not None
                and (now - self._markets_cache[0]).total_seconds() < MARKETS_CACHE_TTL_SECONDS
            ):
                return self._markets_cache[1]

            try:
                resp = await self.http.get(
                    f"{API_BASE}/api/v1/info/markets",
                    headers={"User-Agent": USER_AGENT},
                    timeout=self.TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                raise VenueUnavailable(self.VENUE, f"markets request failed: {e}") from e

            markets = data.get("data") or []
            self._markets_cache = (now, markets)
            return markets

    async def _get_market_stats(self, token: str) -> dict:
        """Return marketStats for the given token."""
        markets = await self._get_markets()
        target = self._market(token)
        for m in markets:
            if m.get("name", "").upper() == target.upper():
                return m.get("marketStats", {})
        raise TokenNotListed(self.VENUE, f"{token} not in markets list")

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------

    async def get_funding_rate(self, token: str) -> FundingRate:
        stats = await self._get_market_stats(token)

        try:
            rate_per_period = float(stats["fundingRate"])
            next_funding_ms = int(stats["nextFundingRate"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(self.VENUE, f"missing or invalid funding field: {e}") from e

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
        stats = await self._get_market_stats(token)

        try:
            oi_base = float(stats["openInterestBase"])
            mark_price = float(stats["markPrice"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(self.VENUE, f"missing or invalid OI/markPrice field: {e}") from e

        return OpenInterest(
            venue=self.VENUE,
            token=token.upper(),
            oi_base=oi_base,
            oi_usd=oi_base * mark_price,
            mark_price=mark_price,
        )

    # ------------------------------------------------------------------
    # Liquidations (WS tail)
    # ------------------------------------------------------------------

    async def get_recent_liquidations(
        self, token: str, *, lookback_seconds: int = 60
    ) -> list[Liquidation]:
        """Subscribe to the public trades WS stream and capture liquidation events.

        Extended's WS publicTrades stream emits all trade types including
        tT="LIQUIDATION" and tT="DELEVERAGE". The REST trades endpoint is
        limited to 50 trades with no pagination, so the WS is the only
        reliable path for liquidation data.

        Side convention: S="SELL" on a liquidation = long was force-closed;
        S="BUY" = short was force-closed.
        """
        if websockets is None:
            raise VenueUnavailable(
                self.VENUE,
                "websockets package not installed (pip install websockets)",
            )

        market = self._market(token)
        captured: list[Liquidation] = []

        ws_url = f"{WS_BASE}/publicTrades/{market}"

        try:
            async with websockets.connect(
                ws_url,
                additional_headers={"User-Agent": USER_AGENT},
                ping_interval=20,
            ) as ws:
                deadline = asyncio.get_event_loop().time() + lookback_seconds
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    for trade in msg.get("data") or []:
                        if trade.get("tT") != "LIQUIDATION":
                            continue
                        try:
                            price = float(trade["p"])
                            qty = float(trade["q"])
                            ts_ms = int(trade["T"])
                            # S="SELL" = taker sold = long was liquidated
                            liq_side = "long" if trade.get("S") == "SELL" else "short"
                            captured.append(
                                Liquidation(
                                    venue=self.VENUE,
                                    token=token.upper(),
                                    side=liq_side,
                                    price=price,
                                    qty_base=qty,
                                    qty_usd=price * qty,
                                    occurred_at=datetime.fromtimestamp(
                                        ts_ms / 1000, tz=timezone.utc
                                    ),
                                )
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
        except Exception as e:
            raise VenueUnavailable(self.VENUE, f"WS liquidation tail failed: {e}") from e

        return captured
