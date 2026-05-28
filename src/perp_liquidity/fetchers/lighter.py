"""
Lighter perpetual DEX client.

API reference:
- https://apidocs.lighter.xyz/llms.txt  (comprehensive endpoint index)
- https://apidocs.lighter.xyz/reference/orderbookdetails.md
- https://apidocs.lighter.xyz/reference/funding-rates.md
- https://apidocs.lighter.xyz/reference/liquidations.md
- https://docs.lighter.xyz/perpetual-futures/funding

Endpoints used:
- GET /api/v1/orderBookDetails             -> metadata for ALL markets (market_id, OI, last price, taker fee)
- GET /api/v1/orderBookOrders?market_id=N  -> the actual orderbook for one market
- GET /api/v1/funding-rates                -> funding for ALL markets, all four tracked exchanges
- WS  wss://mainnet.zklighter.elliot.ai/stream channel `trade/{market_id}` -> liquidation trades

Venue notes:
- Funding period is 1 hour (matches Hyperliquid).
- Open interest in orderBookDetails is denominated in base asset (BTC for BTC market).
- The /funding-rates endpoint returns funding for FOUR exchanges (binance, bybit,
  hyperliquid, lighter). We filter to exchange=='lighter' for this client.
- Public liquidations are exposed via WebSocket ONLY. The REST /api/v1/liquidations
  endpoint is tagged "account" and requires auth + account_index — it returns
  the calling user's own liquidation history, NOT a market-wide feed. The
  public path is subscribing to the `trade/{market_id}` channel and reading
  the `liquidation_trades` array alongside regular trades.
- Fee reality vs marketing: the Lighter front-end advertises "zero fees," but
  the API schema returns 0.01% taker fees (1 bp) for standard accounts. The
  zero-fee narrative applies only to retail UI users on Premium accounts.
  The fee value embedded in our config/fees.yaml should reflect the API
  taker fee, since that is the only access pattern relevant to this tool.
"""

from __future__ import annotations

import asyncio
import json
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


FUNDING_PERIOD_HOURS = 1.0
HOURS_PER_YEAR = 8760

API_BASE = "https://mainnet.zklighter.elliot.ai"
WS_URL = "wss://mainnet.zklighter.elliot.ai/stream"

# Markets metadata changes rarely. A longer TTL is fine here than for prices.
MARKETS_CACHE_TTL_SECONDS = 30.0


class LighterClient(PerpDEXClient):
    VENUE = "lighter"
    COVERAGE = {
        "orderbook": Coverage.REST,
        "funding": Coverage.REST,
        "open_interest": Coverage.REST,
        # The REST /api/v1/liquidations endpoint exists but is per-account
        # (requires auth + account_index). For market-wide public liquidations
        # the only path is the WebSocket trade/{market_id} channel, which emits
        # a `liquidation_trades` array alongside regular trades.
        "liquidations": Coverage.WS_TAIL,
    }

    def __init__(self, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        # Two caches:
        # - markets: symbol -> market_id, OI, last_price, taker_fee (TTL 30s)
        # - funding: per-call, no cache (funding rates change every hour)
        self._markets_cache: tuple[datetime, dict[str, dict]] | None = None
        self._markets_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Markets metadata (shared cache for orderbook, OI, fees)
    # ------------------------------------------------------------------

    async def _get_markets(self) -> dict[str, dict]:
        """Return a map of {symbol_upper: market_detail} for all active perp markets.

        Cached for MARKETS_CACHE_TTL_SECONDS because every fetcher method needs
        the market_id and metadata, but the data rarely changes.
        """
        async with self._markets_lock:
            now = datetime.now(timezone.utc)
            if (
                self._markets_cache is not None
                and (now - self._markets_cache[0]).total_seconds()
                < MARKETS_CACHE_TTL_SECONDS
            ):
                return self._markets_cache[1]

            try:
                resp = await self.http.get(
                    f"{API_BASE}/api/v1/orderBookDetails",
                    params={"filter": "perp"},
                    headers={"Accept": "application/json"},
                    timeout=self.TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                raise VenueUnavailable(
                    self.VENUE, f"orderBookDetails request failed: {e}"
                ) from e

            details = data.get("order_book_details") or []
            if not details:
                raise VenueUnavailable(
                    self.VENUE, "orderBookDetails returned no perp markets"
                )

            markets = {
                m["symbol"].upper(): m
                for m in details
                if m.get("status") == "active" and m.get("symbol")
            }
            self._markets_cache = (now, markets)
            return markets

    async def _resolve_market(self, token: str) -> dict:
        markets = await self._get_markets()
        market = markets.get(token.upper())
        if market is None:
            raise TokenNotListed(self.VENUE, f"{token} not in active perp markets")
        return market

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    async def get_orderbook(self, token: str) -> OrderBook:
        market = await self._resolve_market(token)
        market_id = market["market_id"]

        try:
            resp = await self.http.get(
                f"{API_BASE}/api/v1/orderBookOrders",
                params={"market_id": market_id, "limit": 50},
                headers={"Accept": "application/json"},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise VenueUnavailable(
                self.VENUE, f"orderBookOrders request failed: {e}"
            ) from e

        bids_raw = data.get("bids") or []
        asks_raw = data.get("asks") or []
        if not bids_raw or not asks_raw:
            raise TokenNotListed(
                self.VENUE, f"{token} has empty orderbook (market_id={market_id})"
            )

        bids = [
            OrderBookLevel(
                price=float(level["price"]),
                qty=float(level["remaining_base_amount"]),
            )
            for level in bids_raw
        ]
        asks = [
            OrderBookLevel(
                price=float(level["price"]),
                qty=float(level["remaining_base_amount"]),
            )
            for level in asks_raw
        ]

        bids.sort(key=lambda b: b.price, reverse=True)
        asks.sort(key=lambda a: a.price)

        return OrderBook(venue=self.VENUE, token=token.upper(), bids=bids, asks=asks)

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------

    async def get_funding_rate(self, token: str) -> FundingRate:
        # Verify the market exists so we get a clean TokenNotListed if not
        await self._resolve_market(token)

        try:
            resp = await self.http.get(
                f"{API_BASE}/api/v1/funding-rates",
                headers={"Accept": "application/json"},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise VenueUnavailable(
                self.VENUE, f"funding-rates request failed: {e}"
            ) from e

        rates = data.get("funding_rates") or []
        token_upper = token.upper()

        # Filter to lighter-only entries for this token
        match = next(
            (
                r
                for r in rates
                if r.get("exchange") == "lighter"
                and r.get("symbol", "").upper() == token_upper
            ),
            None,
        )

        if match is None:
            raise VenueUnavailable(
                self.VENUE,
                f"funding rate for {token} not in /funding-rates response",
            )

        try:
            rate_per_period = float(match["rate"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"invalid funding rate field: {e}"
            ) from e

        apr = rate_per_period * (HOURS_PER_YEAR / FUNDING_PERIOD_HOURS)

        now = datetime.now(timezone.utc)
        next_funding = (now + timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0
        )

        return FundingRate(
            venue=self.VENUE,
            token=token_upper,
            rate_per_period=rate_per_period,
            period_hours=FUNDING_PERIOD_HOURS,
            apr_annualized=apr,
            next_funding_at=next_funding,
        )

    # ------------------------------------------------------------------
    # Open interest
    # ------------------------------------------------------------------

    async def get_open_interest(self, token: str) -> OpenInterest:
        market = await self._resolve_market(token)

        try:
            oi_base = float(market["open_interest"])
            last_price = float(market["last_trade_price"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"missing or invalid OI/price field: {e}"
            ) from e

        if last_price <= 0:
            raise VenueUnavailable(
                self.VENUE, "last_trade_price is zero or negative"
            )

        # Lighter does not expose a separate mark_price in orderBookDetails.
        # We use last_trade_price as the best public proxy. This is documented
        # transparently rather than hidden — using last vs mark introduces a
        # small skew for thin markets but it's well within measurement noise
        # for the cross-venue comparisons we care about.
        return OpenInterest(
            venue=self.VENUE,
            token=token.upper(),
            oi_base=oi_base,
            oi_usd=oi_base * last_price,
            mark_price=last_price,  # Proxy: last trade, not true mark
        )

    # ------------------------------------------------------------------
    # Liquidations (WS tail)
    # ------------------------------------------------------------------

    async def get_recent_liquidations(
        self, token: str, *, lookback_seconds: int = 60
    ) -> list[Liquidation]:
        """Subscribe to the trade/{market_id} WS channel and capture liquidation prints.

        Lighter exposes liquidations as part of the public `trade` WebSocket
        channel, which emits a `liquidation_trades` array alongside regular
        trades. There is no public REST endpoint for market-wide liquidations:
        /api/v1/liquidations is an authenticated per-account endpoint.

        We tail the channel for `lookback_seconds` and return whatever was
        captured (often zero on quiet markets — that's a real result, not an
        error).
        """
        try:
            import websockets
        except ImportError as e:
            raise VenueUnavailable(
                self.VENUE,
                "websockets package not installed (pip install websockets)",
            ) from e

        market = await self._resolve_market(token)
        market_id = market["market_id"]

        captured: list[Liquidation] = []

        subscribe_msg = json.dumps(
            {"type": "subscribe", "channel": f"trade/{market_id}"}
        )

        try:
            async with websockets.connect(WS_URL, ping_interval=20) as ws:
                await ws.send(subscribe_msg)

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

                    # We only care about updates that include liquidation trades.
                    # Subscribe confirmation messages and pure trade updates can
                    # be ignored.
                    liq_trades = msg.get("liquidation_trades") or []
                    if not liq_trades:
                        continue

                    for trade in liq_trades:
                        try:
                            price = float(trade.get("price", 0))
                            qty = float(
                                trade.get("size")
                                or trade.get("base_amount")
                                or 0
                            )
                            ts_raw = (
                                trade.get("transaction_time")
                                or trade.get("timestamp")
                                or trade.get("time")
                            )
                            if ts_raw is None or price <= 0 or qty <= 0:
                                continue

                            ts_int = int(ts_raw)
                            # Lighter uses microseconds in some places and ms
                            # in others. Auto-detect by magnitude.
                            if ts_int > 1e17:
                                occurred_at = datetime.fromtimestamp(
                                    ts_int / 1e9, tz=timezone.utc
                                )
                            elif ts_int > 1e14:
                                occurred_at = datetime.fromtimestamp(
                                    ts_int / 1e6, tz=timezone.utc
                                )
                            elif ts_int > 1e12:
                                occurred_at = datetime.fromtimestamp(
                                    ts_int / 1000, tz=timezone.utc
                                )
                            else:
                                occurred_at = datetime.fromtimestamp(
                                    ts_int, tz=timezone.utc
                                )

                            # Determine side. Lighter trade objects use is_ask:
                            #   is_ask=True  -> the trade hit the ask, meaning a
                            #                   buyer liquidation (short was forced
                            #                   to close by buying back)
                            #   is_ask=False -> the trade hit the bid, meaning a
                            #                   long was force-liquidated (sold)
                            is_ask = trade.get("is_ask")
                            if is_ask is None:
                                side_raw = (
                                    trade.get("side") or trade.get("direction") or ""
                                ).lower()
                                is_ask = side_raw in ("ask", "buy")
                            liq_side = "short" if is_ask else "long"

                            captured.append(
                                Liquidation(
                                    venue=self.VENUE,
                                    token=token.upper(),
                                    side=liq_side,
                                    price=price,
                                    qty_base=qty,
                                    qty_usd=price * qty,
                                    occurred_at=occurred_at,
                                )
                            )
                        except (TypeError, ValueError, KeyError):
                            # Malformed individual trade; skip but don't bail
                            continue
        except Exception as e:
            raise VenueUnavailable(
                self.VENUE, f"WS liquidation tail failed: {e}"
            ) from e

        return captured
