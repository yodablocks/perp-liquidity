"""
Tests for ApeXClient.

Uses httpx.MockTransport for all REST calls. No WS needed (liquidations
are NOT_AVAILABLE -- private account stream only).

API shapes confirmed from live probing 2026-05-28:
- Base URL: https://omni.apex.exchange
- Depth:   GET /api/v3/depth?symbol=BTCUSDT
  -> {"data": {"a": [["price", "size"], ...], "b": [["price", "size"], ...]}}
  Note: "a" = asks, "b" = bids
- Ticker:  GET /api/v3/ticker?symbol=BTCUSDT
  -> {"data": [{"fundingRate": "-0.00000594", "markPrice": "73325.96",
       "openInterest": "1541.044", "nextFundingTime": "2026-05-28T10:00:00Z",
       "indexPrice": "73369.06", "lastPrice": "73319.7", ...}]}
  openInterest is in base asset (BTC). nextFundingTime is ISO 8601 string.
- Funding history: GET /api/v3/history-funding?symbol=BTC-USDT&limit=3
  -> {"data": {"historyFunds": [{"rate": "0.00001250", "fundingTime": 1779958800000, ...}]}}
  Timestamp delta confirms 1h funding period (1779958800000 - 1779955200000 = 3600s).
- Liquidations: NOT_AVAILABLE (only in private account WS stream, requires auth).
- Symbol convention: {TOKEN}USDT for depth/ticker (e.g. BTCUSDT),
  {TOKEN}-USDT with hyphen for history-funding (e.g. BTC-USDT).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from perp_liquidity.fetchers.apex import ApeXClient
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


DEPTH_RESPONSE = {
    "data": {
        "a": [
            ["73319.6", "7.299"],
            ["73321.1", "0.549"],
        ],
        "b": [
            ["73319.5", "2.100"],
            ["73318.0", "5.000"],
        ],
    }
}

TICKER_RESPONSE = {
    "data": [
        {
            "symbol": "BTCUSDT",
            "fundingRate": "-0.00000594",
            "highPrice24h": "76138.9",
            "indexPrice": "73369.06",
            "lastPrice": "73319.7",
            "lowPrice24h": "72692.1",
            "nextFundingTime": "2026-05-28T10:00:00Z",
            "openInterest": "1541.044",
            "oraclePrice": "",
            "markPrice": "73325.96",
            "predictedFundingRate": "0.0000125",
        }
    ],
    "timeCost": 2244226,
}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_declares_all_four_dimensions():
    assert ApeXClient.COVERAGE["orderbook"] == Coverage.REST
    assert ApeXClient.COVERAGE["funding"] == Coverage.REST
    assert ApeXClient.COVERAGE["open_interest"] == Coverage.REST
    assert ApeXClient.COVERAGE["liquidations"] == Coverage.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orderbook_returns_sorted_bids_and_asks():
    transport = make_transport((200, DEPTH_RESPONSE))
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        ob = await c.get_orderbook("BTC")

    assert isinstance(ob, OrderBook)
    assert ob.venue == "apex"
    assert ob.token == "BTC"
    bid_prices = [b.price for b in ob.bids]
    assert bid_prices == sorted(bid_prices, reverse=True)
    ask_prices = [a.price for a in ob.asks]
    assert ask_prices == sorted(ask_prices)
    assert ob.bids[0].price < ob.asks[0].price


@pytest.mark.asyncio
async def test_orderbook_symbol_is_uppercased_with_usdt_suffix():
    """Token 'btc' must hit the endpoint as 'BTCUSDT'."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=DEPTH_RESPONSE)

    async with ApeXClient(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as c:
        await c.get_orderbook("btc")

    assert any("BTCUSDT" in u for u in seen)


@pytest.mark.asyncio
async def test_orderbook_raises_token_not_listed_on_empty_book():
    empty = {"data": {"a": [], "b": []}}
    transport = make_transport((200, empty))
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_orderbook("FAKE")


@pytest.mark.asyncio
async def test_orderbook_raises_venue_unavailable_on_http_error():
    transport = make_transport((500, {"error": "internal"}))
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_orderbook("BTC")


@pytest.mark.asyncio
async def test_orderbook_raises_token_not_listed_on_404():
    transport = make_transport((404, {"error": "not found"}))
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_orderbook("FAKE")


# ---------------------------------------------------------------------------
# Funding rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_funding_rate_annualizes_with_1h_period():
    transport = make_transport((200, TICKER_RESPONSE))
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        fr = await c.get_funding_rate("BTC")

    assert isinstance(fr, FundingRate)
    assert fr.venue == "apex"
    assert fr.token == "BTC"
    assert fr.period_hours == 1.0
    expected_apr = -0.00000594 * (8760 / 1.0)
    assert abs(fr.apr_annualized - expected_apr) < 1e-9
    assert fr.rate_per_period == pytest.approx(-0.00000594)


@pytest.mark.asyncio
async def test_funding_rate_next_funding_parsed_from_iso_string():
    transport = make_transport((200, TICKER_RESPONSE))
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        fr = await c.get_funding_rate("BTC")

    expected = datetime(2026, 5, 28, 10, 0, 0, tzinfo=timezone.utc)
    assert fr.next_funding_at == expected


@pytest.mark.asyncio
async def test_funding_rate_raises_token_not_listed_on_404():
    transport = make_transport((404, {"error": "not found"}))
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_funding_rate("FAKE")


@pytest.mark.asyncio
async def test_funding_rate_raises_venue_unavailable_on_http_error():
    transport = make_transport((502, {"error": "bad gateway"}))
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_funding_rate("BTC")


# ---------------------------------------------------------------------------
# Open interest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_interest_uses_base_asset_and_mark_price():
    transport = make_transport((200, TICKER_RESPONSE))
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        oi = await c.get_open_interest("BTC")

    assert isinstance(oi, OpenInterest)
    assert oi.venue == "apex"
    assert oi.token == "BTC"
    assert oi.oi_base == pytest.approx(1541.044)
    expected_usd = 1541.044 * 73325.96
    assert oi.oi_usd == pytest.approx(expected_usd, rel=1e-6)
    assert oi.mark_price == pytest.approx(73325.96)


@pytest.mark.asyncio
async def test_open_interest_shares_ticker_call_with_funding():
    """funding + OI back-to-back should only make one ticker call (TTL cache)."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if "ticker" in str(request.url):
            call_count += 1
        return httpx.Response(200, json=TICKER_RESPONSE)

    async with ApeXClient(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as c:
        await c.get_funding_rate("BTC")
        await c.get_open_interest("BTC")

    assert call_count == 1, f"Expected 1 ticker call, got {call_count}"


@pytest.mark.asyncio
async def test_open_interest_raises_venue_unavailable_on_http_error():
    transport = make_transport((500, {"error": "internal"}))
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_open_interest("BTC")


# ---------------------------------------------------------------------------
# Liquidations: NOT_AVAILABLE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_liquidations_raises_not_implemented():
    """ApeX liquidations are only in the private account WS stream."""
    transport = make_transport()
    async with ApeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(NotImplementedError):
            await c.get_recent_liquidations("BTC")
