"""
Hyperliquid perpetual DEX client.

API reference:
- https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint
- https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket

Endpoints used:
- POST /info { "type": "l2Book", "coin": "BTC" }            -> order book
- POST /info { "type": "metaAndAssetCtxs" }                  -> funding + OI + mark price (all assets)
- WS  /ws subscribe { "type": "userEvents" / "trades" / ... } -> liquidations live stream

Venue notes:
- Funding period is 1 hour (different from Binance-style 8h venues).
- openInterest in metaAndAssetCtxs is denominated in BASE ASSET (e.g. BTC), not USD.
- Liquidations are not exposed via REST. We tail the WS trades stream and filter
  for trades with the `liquidation` flag, or use the dedicated `liquidations`
  subscription depending on payload shape.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

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


# Hyperliquid pays/charges funding every hour.
FUNDING_PERIOD_HOURS = 1.0
HOURS_PER_YEAR = 8760

REST_URL = "https://api.hyperliquid.xyz/info"
WS_URL = "wss://api.hyperliquid.xyz/ws"

# How long to cache metaAndAssetCtxs within one client instance. The endpoint
# returns data for ALL assets, so a small TTL avoids hammering it when callers
# ask for funding + OI on the same token back-to-back.
META_CACHE_TTL_SECONDS = 2.0


class HyperliquidClient(PerpDEXClient):
    VENUE = "hyperliquid"
    COVERAGE = {
        "orderbook": Coverage.REST,
        "funding": Coverage.REST,
        "open_interest": Coverage.REST,
        "liquidations": Coverage.WS_TAIL,
    }

    def __init__(self, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        self._meta_cache: tuple[datetime, list, list] | None = None
        self._meta_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    async def get_orderbook(self, token: str) -> OrderBook:
        try:
            resp = await self.http.post(
                REST_URL,
                json={"type": "l2Book", "coin": token.upper()},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise VenueUnavailable(self.VENUE, f"l2Book request failed: {e}") from e

        levels = data.get("levels")
        if not levels or len(levels) < 2:
            # Hyperliquid returns an empty `levels` array for unknown coins
            # rather than a 4xx. Distinguish "no liquidity" from "not listed".
            raise TokenNotListed(self.VENUE, f"{token} not listed or no liquidity")

        bids_raw, asks_raw = levels[0], levels[1]

        bids = [
            OrderBookLevel(price=float(level["px"]), qty=float(level["sz"]))
            for level in bids_raw
        ]
        asks = [
            OrderBookLevel(price=float(level["px"]), qty=float(level["sz"]))
            for level in asks_raw
        ]

        # Enforce the invariant declared in OrderBook's docstring.
        bids.sort(key=lambda b: b.price, reverse=True)
        asks.sort(key=lambda a: a.price)

        return OrderBook(venue=self.VENUE, token=token.upper(), bids=bids, asks=asks)

    # ------------------------------------------------------------------
    # Shared meta endpoint (funding + OI + mark price)
    # ------------------------------------------------------------------

    async def _get_asset_context(self, token: str) -> tuple[dict, dict]:
        """Return (universe_entry, context) for the given token.

        Wraps metaAndAssetCtxs with a tiny TTL cache so that asking for funding
        + OI in quick succession is one HTTP call, not two.
        """
        token_upper = token.upper()

        async with self._meta_lock:
            now = datetime.now(timezone.utc)
            if (
                self._meta_cache is not None
                and (now - self._meta_cache[0]).total_seconds() < META_CACHE_TTL_SECONDS
            ):
                universe, contexts = self._meta_cache[1], self._meta_cache[2]
            else:
                try:
                    resp = await self.http.post(
                        REST_URL,
                        json={"type": "metaAndAssetCtxs"},
                        timeout=self.TIMEOUT,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                except httpx.HTTPError as e:
                    raise VenueUnavailable(
                        self.VENUE, f"metaAndAssetCtxs failed: {e}"
                    ) from e

                if not isinstance(payload, list) or len(payload) != 2:
                    raise VenueUnavailable(
                        self.VENUE, f"unexpected metaAndAssetCtxs shape: {type(payload)}"
                    )

                universe = payload[0].get("universe", [])
                contexts = payload[1]
                self._meta_cache = (now, universe, contexts)

        # Match by index: universe[i] corresponds to contexts[i].
        for idx, entry in enumerate(universe):
            if entry.get("name", "").upper() == token_upper:
                if idx >= len(contexts):
                    raise VenueUnavailable(
                        self.VENUE,
                        f"universe/contexts length mismatch for {token}",
                    )
                return entry, contexts[idx]

        raise TokenNotListed(self.VENUE, f"{token} not in universe")

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------

    async def get_funding_rate(self, token: str) -> FundingRate:
        _, ctx = await self._get_asset_context(token)

        try:
            rate_per_period = float(ctx["funding"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"missing or invalid funding field: {e}"
            ) from e

        apr = rate_per_period * (HOURS_PER_YEAR / FUNDING_PERIOD_HOURS)

        # Next funding boundary is the next whole hour, UTC.
        now = datetime.now(timezone.utc)
        next_funding = (now + timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0
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
        _, ctx = await self._get_asset_context(token)

        try:
            oi_base = float(ctx["openInterest"])
            mark_price = float(ctx["markPx"])
        except (KeyError, TypeError, ValueError) as e:
            raise VenueUnavailable(
                self.VENUE, f"missing or invalid OI/markPx field: {e}"
            ) from e

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
        """Subscribe to the WS trades feed and capture liquidation prints.

        Hyperliquid does not expose a public liquidations REST endpoint. The
        WS `trades` channel emits each trade with a `users` field; liquidation
        trades are marked separately. We tail for `lookback_seconds` and
        return whatever was captured (often zero — that's a real result).
        """
        if websockets is None:
            raise VenueUnavailable(
                self.VENUE,
                "websockets package not installed (pip install websockets)",
            )

        token_upper = token.upper()
        captured: list[Liquidation] = []

        subscribe_msg = json.dumps(
            {
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": token_upper},
            }
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

                    if msg.get("channel") != "trades":
                        continue

                    for trade in msg.get("data", []):
                        # Hyperliquid marks liquidation trades with a `liq`-style flag.
                        # The field name has varied historically; check both.
                        is_liq = (
                            trade.get("liquidation") is True
                            or trade.get("isLiquidation") is True
                            or trade.get("hash") == "0x0000000000000000000000000000000000000000000000000000000000000000"
                        )
                        if not is_liq:
                            continue

                        price = float(trade.get("px", 0))
                        qty = float(trade.get("sz", 0))
                        # `side` in HL trades: "A" = ask filled (long got hit / short opened),
                        # "B" = bid filled. For a liquidation, the position that was forced
                        # to close is the opposite side of the print.
                        hl_side = trade.get("side", "")
                        liq_side = "long" if hl_side == "A" else "short"

                        captured.append(
                            Liquidation(
                                venue=self.VENUE,
                                token=token_upper,
                                side=liq_side,
                                price=price,
                                qty_base=qty,
                                qty_usd=price * qty,
                                occurred_at=datetime.fromtimestamp(
                                    trade.get("time", 0) / 1000, tz=timezone.utc
                                ),
                            )
                        )
        except Exception as e:
            raise VenueUnavailable(
                self.VENUE, f"WS liquidation tail failed: {e}"
            ) from e

        return captured
