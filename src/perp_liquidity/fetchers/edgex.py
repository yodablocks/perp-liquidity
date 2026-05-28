"""
EdgeX perpetual DEX client.

API reference:
- https://edgex-1.gitbook.io/edgex-documentation/api/public-api
- Base: https://pro.edgex.exchange

Endpoints used:
- GET /api/v1/public/meta/getMetaData          -> contract list (name -> contractId)
- GET /api/v1/public/quote/getDepth            -> order book (params: contractId, level)
- GET /api/v1/public/quote/getTicker           -> funding rate + OI + mark price (params: contractId)

Venue notes:
- Symbol convention: {TOKEN}USD (e.g. BTCUSD), no separator, no USDT suffix.
- Funding period is 4 hours (fundingRateIntervalMin=240, confirmed from live API 2026-05-27).
- Contract IDs are stable integers (BTC = 10000001). We fetch them once from
  getMetaData and cache permanently on the instance.
- getTicker returns funding rate, nextFundingTime, openInterest (base asset), and markPrice
  in a single call. We TTL-cache it 2s so get_funding_rate + get_open_interest back-to-back
  share one HTTP call.
- Liquidations are NOT_AVAILABLE. START_LIQUIDATING / FINISH_LIQUIDATING events only appear
  on the private WS channel (requires accountId + ECDSA signature). No public liquidation
  feed confirmed 2026-05-27.
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


FUNDING_PERIOD_HOURS = 4.0
HOURS_PER_YEAR = 8760

API_BASE = "https://pro.edgex.exchange"

# getTicker returns funding + OI + markPrice together. Cache briefly so
# get_funding_rate + get_open_interest back-to-back share one HTTP call.
TICKER_CACHE_TTL_SECONDS = 2.0


class EdgeXClient(PerpDEXClient):
    VENUE = "edgex"
    COVERAGE = {
        "orderbook": Coverage.REST,
        "funding": Coverage.REST,
        "open_interest": Coverage.REST,
        # Liquidations require ECDSA-signed private WS. No public feed.
        # START_LIQUIDATING / FINISH_LIQUIDATING on private channel only.
        # Confirmed 2026-05-27.
        "liquidations": Coverage.NOT_AVAILABLE,
    }

    def __init__(self, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        # Contract name -> contractId mapping. Fetched once, cached permanently.
        self._contract_map: dict[str, str] | None = None
        self._contract_map_lock = asyncio.Lock()
        # getTicker cache: contractId -> (fetched_at, data)
        self._ticker_cache: dict[str, tuple[datetime, dict]] = {}
        self._ticker_lock = asyncio.Lock()

    @staticmethod
    def _contract_name(token: str) -> str:
        """'BTC' -> 'BTCUSD'"""
        return f"{token.upper()}USD"

    # ------------------------------------------------------------------
    # Contract ID resolution (one-time fetch, cached for instance lifetime)
    # ------------------------------------------------------------------

    async def _get_contract_id(self, token: str) -> str:
        """Resolve token to EdgeX contractId, fetching metadata if needed."""
        async with self._contract_map_lock:
            if self._contract_map is None:
                try:
                    resp = await self.http.get(
                        f"{API_BASE}/api/v1/public/meta/getMetaData",
                        timeout=self.TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as e:
                    raise VenueUnavailable(
                        self.VENUE, f"getMetaData HTTP error: {e}"
                    ) from e
                except httpx.HTTPError as e:
                    raise VenueUnavailable(
                        self.VENUE, f"getMetaData request failed: {e}"
                    ) from e

                try:
                    # API shape: data is a dict with contractList key.
                    # Older shape wrapped it in a list: data[0]["contractList"].
                    raw = data["data"]
                    if isinstance(raw, list):
                        raw = raw[0]
                    contract_list = raw["contractList"]
                    self._contract_map = {
                        c["contractName"]: c["contractId"] for c in contract_list
                    }
                except (KeyError, IndexError, TypeError) as e:
                    raise VenueUnavailable(
                        self.VENUE, f"unexpected getMetaData shape: {e}"
                    ) from e

        name = self._contract_name(token)
        contract_id = self._contract_map.get(name)
        if contract_id is None:
            raise TokenNotListed(self.VENUE, f"{token} not listed (looked up as {name!r})")
        return contract_id

    # ------------------------------------------------------------------
    # Shared ticker (funding + OI + markPrice in one call)
    # ------------------------------------------------------------------

    async def _get_ticker(self, token: str) -> dict:
        """Fetch getTicker for token, with a short TTL cache."""
        contract_id = await self._get_contract_id(token)

        async with self._ticker_lock:
            now = datetime.now(timezone.utc)
            cached = self._ticker_cache.get(contract_id)
            if (
                cached is not None
                and (now - cached[0]).total_seconds() < TICKER_CACHE_TTL_SECONDS
            ):
                return cached[1]

            try:
                resp = await self.http.get(
                    f"{API_BASE}/api/v1/public/quote/getTicker",
                    params={"contractId": contract_id},
                    timeout=self.TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (400, 404):
                    raise TokenNotListed(self.VENUE, f"{token} not listed") from e
                raise VenueUnavailable(
                    self.VENUE, f"getTicker HTTP error: {e}"
                ) from e
            except httpx.HTTPError as e:
                raise VenueUnavailable(
                    self.VENUE, f"getTicker request failed: {e}"
                ) from e

            try:
                ticker = data["data"][0]
            except (KeyError, IndexError, TypeError) as e:
                raise VenueUnavailable(
                    self.VENUE, f"unexpected getTicker shape: {e}"
                ) from e

            self._ticker_cache[contract_id] = (now, ticker)
            return ticker

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    async def get_orderbook(self, token: str) -> OrderBook:
        contract_id = await self._get_contract_id(token)

        try:
            resp = await self.http.get(
                f"{API_BASE}/api/v1/public/quote/getDepth",
                params={"contractId": contract_id, "level": 200},
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (400, 404):
                raise TokenNotListed(self.VENUE, f"{token} not listed") from e
            raise VenueUnavailable(self.VENUE, f"getDepth HTTP error: {e}") from e
        except httpx.HTTPError as e:
            raise VenueUnavailable(self.VENUE, f"getDepth request failed: {e}") from e

        try:
            book = data["data"][0]
            bids_raw = book.get("bids") or []
            asks_raw = book.get("asks") or []
        except (KeyError, IndexError, TypeError) as e:
            raise VenueUnavailable(self.VENUE, f"unexpected getDepth shape: {e}") from e

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
            rate_per_period = float(ticker["fundingRate"])
            next_funding_ms = int(ticker["nextFundingTime"])
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
        """EdgeX liquidation events are only available on the private WS channel.

        The START_LIQUIDATING / FINISH_LIQUIDATING events require accountId and
        ECDSA signature authentication. No public liquidation feed exists.
        The public trades.{contractId} WS channel carries no isLiquidation flag.
        Both confirmed by live probing on 2026-05-27.
        """
        raise NotImplementedError(
            "EdgeX liquidations require private WS authentication. "
            "Coverage.NOT_AVAILABLE; see docstring for details."
        )
