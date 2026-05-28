"""
Tests for GRVTClient.

Uses httpx.MockTransport for all REST calls. No WS needed (liquidations
NOT_AVAILABLE -- trade messages have no is_liquidation flag).

API shapes confirmed from live probing 2026-05-28:
- Base: https://market-data.grvt.io/full/v1
- All endpoints are POST with JSON body (except ticker which accepts GET with query params).
- Book:    POST /book {"instrument": "BTC_USDT_Perp", "depth": 10}
  -> {"result": {"instrument": "BTC_USDT_Perp",
       "bids": [{"price": "73337.6", "size": "9.594", "num_orders": 17}, ...],
       "asks": [{"price": "73337.7", "size": "1.339", "num_orders": 3}, ...]}}
- Ticker:  GET /ticker?instrument=BTC_USDT_Perp
  -> {"result": {"instrument": "BTC_USDT_Perp", "mark_price": "73286.267...",
       "open_interest": "2683.355...", "funding_rate": "0.01",
       "next_funding_time": "1779984000000000000", ...}}
  All timestamps are nanoseconds. next_funding_time / 1e9 -> Unix seconds.
  open_interest is in base asset (BTC).
- Instrument: POST /instrument {"instrument": "BTC_USDT_Perp"}
  -> funding_interval_hours: 8 (confirmed from timestamp deltas too)
- Funding period: 8h (funding_interval_hours=8, delta between funding_time entries = 8h)
- Liquidations: NOT_AVAILABLE (trade messages have no is_liquidation flag,
  no dedicated liquidation endpoint exists). Confirmed 2026-05-28.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from perp_liquidity.fetchers.grvt import GRVTClient
from perp_liquidity.fetchers.base import (
    Coverage,
    FundingRate,
    OpenInterest,
    OrderBook,
    TokenNotListed,
    VenueUnavailable,
)


# ---------------------------------------------------------------------------
# Helpers / shared response shapes
# ---------------------------------------------------------------------------


def make_transport(*responses: tuple[int, dict]) -> httpx.MockTransport:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if not queue:
            raise RuntimeError(f"Unexpected request to {request.url}")
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


BOOK_RESPONSE = {
    "result": {
        "event_time": "1779970813250000000",
        "instrument": "BTC_USDT_Perp",
        "bids": [
            {"price": "73337.6", "size": "9.594", "num_orders": 17},
            {"price": "73337.5", "size": "0.272", "num_orders": 1},
        ],
        "asks": [
            {"price": "73337.7", "size": "1.339", "num_orders": 3},
            {"price": "73338.0", "size": "0.500", "num_orders": 1},
        ],
    }
}

TICKER_RESPONSE = {
    "result": {
        "event_time": "1779970951500000000",
        "instrument": "BTC_USDT_Perp",
        "mark_price": "73286.267209101",
        "index_price": "73317.112966795",
        "last_price": "73290.0",
        "best_bid_price": "73283.8",
        "best_ask_price": "73283.9",
        "funding_rate_8h_curr": "0.01",
        "funding_rate_8h_avg": "0.01",
        "open_interest": "2683.355941467",
        "funding_rate": "0.01",
        "next_funding_time": "1779984000000000000",
    }
}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_declares_all_four_dimensions():
    assert GRVTClient.COVERAGE["orderbook"] == Coverage.REST
    assert GRVTClient.COVERAGE["funding"] == Coverage.REST
    assert GRVTClient.COVERAGE["open_interest"] == Coverage.REST
    assert GRVTClient.COVERAGE["liquidations"] == Coverage.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orderbook_returns_sorted_bids_and_asks():
    transport = make_transport((200, BOOK_RESPONSE))
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        ob = await c.get_orderbook("BTC")

    assert isinstance(ob, OrderBook)
    assert ob.venue == "grvt"
    assert ob.token == "BTC"
    bid_prices = [b.price for b in ob.bids]
    assert bid_prices == sorted(bid_prices, reverse=True)
    ask_prices = [a.price for a in ob.asks]
    assert ask_prices == sorted(ask_prices)
    assert ob.bids[0].price < ob.asks[0].price


@pytest.mark.asyncio
async def test_orderbook_instrument_name_is_token_usdt_perp():
    """Token 'btc' must POST to /book with instrument 'BTC_USDT_Perp'."""
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen_bodies.append(json.loads(request.content))
        return httpx.Response(200, json=BOOK_RESPONSE)

    async with GRVTClient(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as c:
        await c.get_orderbook("btc")

    assert any(b.get("instrument") == "BTC_USDT_Perp" for b in seen_bodies)


@pytest.mark.asyncio
async def test_orderbook_raises_token_not_listed_on_empty_book():
    empty = {"result": {"instrument": "FAKE_USDT_Perp", "bids": [], "asks": []}}
    transport = make_transport((200, empty))
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_orderbook("FAKE")


@pytest.mark.asyncio
async def test_orderbook_raises_venue_unavailable_on_http_error():
    transport = make_transport((500, {"error": "internal"}))
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_orderbook("BTC")


@pytest.mark.asyncio
async def test_orderbook_raises_token_not_listed_on_404():
    transport = make_transport((404, {"error": "not found"}))
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_orderbook("FAKE")


# ---------------------------------------------------------------------------
# Funding rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ticker_uses_post_with_json_body():
    """GET /ticker returns 405; fetcher must POST with instrument in body."""
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen.append((request.method, json.loads(request.content) if request.content else {}))
        return httpx.Response(200, json=TICKER_RESPONSE)

    async with GRVTClient(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as c:
        await c.get_funding_rate("BTC")

    assert any(method == "POST" and body.get("instrument") == "BTC_USDT_Perp"
               for method, body in seen)


@pytest.mark.asyncio
async def test_funding_rate_annualizes_with_8h_period():
    transport = make_transport((200, TICKER_RESPONSE))
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        fr = await c.get_funding_rate("BTC")

    assert isinstance(fr, FundingRate)
    assert fr.venue == "grvt"
    assert fr.token == "BTC"
    assert fr.period_hours == 8.0
    expected_apr = 0.01 * (8760 / 8.0)
    assert abs(fr.apr_annualized - expected_apr) < 1e-9
    assert fr.rate_per_period == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_funding_rate_next_funding_parsed_from_nanoseconds():
    transport = make_transport((200, TICKER_RESPONSE))
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        fr = await c.get_funding_rate("BTC")

    # 1779984000000000000 ns = 1779984000 s
    expected = datetime.fromtimestamp(1779984000000000000 / 1e9, tz=timezone.utc)
    assert fr.next_funding_at == expected


@pytest.mark.asyncio
async def test_funding_rate_raises_venue_unavailable_on_http_error():
    transport = make_transport((502, {"error": "bad gateway"}))
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_funding_rate("BTC")


@pytest.mark.asyncio
async def test_funding_rate_raises_token_not_listed_on_404():
    transport = make_transport((404, {"error": "not found"}))
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_funding_rate("FAKE")


# ---------------------------------------------------------------------------
# Open interest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_interest_uses_base_asset_and_mark_price():
    transport = make_transport((200, TICKER_RESPONSE))
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        oi = await c.get_open_interest("BTC")

    assert isinstance(oi, OpenInterest)
    assert oi.venue == "grvt"
    assert oi.token == "BTC"
    assert oi.oi_base == pytest.approx(2683.355941467)
    expected_usd = 2683.355941467 * 73286.267209101
    assert oi.oi_usd == pytest.approx(expected_usd, rel=1e-6)
    assert oi.mark_price == pytest.approx(73286.267209101)


@pytest.mark.asyncio
async def test_open_interest_shares_ticker_call_with_funding():
    """funding + OI back-to-back should only make one ticker call (TTL cache)."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if "ticker" in str(request.url):
            call_count += 1
        return httpx.Response(200, json=TICKER_RESPONSE)

    async with GRVTClient(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as c:
        await c.get_funding_rate("BTC")
        await c.get_open_interest("BTC")

    assert call_count == 1, f"Expected 1 ticker call, got {call_count}"


@pytest.mark.asyncio
async def test_open_interest_raises_venue_unavailable_on_http_error():
    transport = make_transport((500, {"error": "internal"}))
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_open_interest("BTC")


# ---------------------------------------------------------------------------
# Liquidations: NOT_AVAILABLE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_liquidations_raises_not_implemented():
    """GRVT trade messages have no is_liquidation flag; no public liq feed."""
    transport = make_transport()
    async with GRVTClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(NotImplementedError):
            await c.get_recent_liquidations("BTC")
