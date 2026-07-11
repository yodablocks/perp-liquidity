"""
Tests for ExtendedClient.

Uses httpx.MockTransport for REST calls. WS liquidation tail is tested with a
fake websockets server via a monkeypatched connect function.

API shape confirmed from live probing 2026-05-27:
- Orderbook: GET /api/v1/info/markets/BTC-USD/orderbook
  -> {"status":"OK","data":{"market":"BTC-USD","bid":[{"qty":"...","price":"..."}],"ask":[...]}}
- Markets list: GET /api/v1/info/markets
  -> {"status":"OK","data":[{...,"name":"BTC-USD","marketStats":{
       "fundingRate":"0.000013","nextFundingRate":1779908400000,
       "openInterest":"96465777","openInterestBase":"1289.07","markPrice":"74839.5",...}}]}
- WS trades: wss://api.starknet.extended.exchange/stream.extended.exchange/v1/publicTrades/BTC-USD
  -> {"ts":...,"data":[{"m":"BTC-USD","S":"BUY","tT":"TRADE","T":...,"p":"...","q":"...","i":...}],"seq":...}
  tT can be TRADE, LIQUIDATION, or DELEVERAGE
  S: "SELL" on a LIQUIDATION = long was force-closed (liq_side="long")
  S: "BUY"  on a LIQUIDATION = short was force-closed (liq_side="short")
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from perp_liquidity.fetchers.extended import ExtendedClient
from perp_liquidity.fetchers.base import (
    Coverage,
    FundingRate,
    OpenInterest,
    OrderBook,
    TokenNotListed,
    VenueUnavailable,
)


# ---------------------------------------------------------------------------
# Fixtures / shared response shapes
# ---------------------------------------------------------------------------

ORDERBOOK_RESPONSE = {
    "status": "OK",
    "data": {
        "market": "BTC-USD",
        "bid": [
            {"qty": "17.07654", "price": "74846"},
            {"qty": "2.50000", "price": "74845"},
        ],
        "ask": [
            {"qty": "4.11472", "price": "74847"},
            {"qty": "1.00000", "price": "74848"},
        ],
    },
}

# marketStats entry embedded in the markets list
BTC_MARKET_STATS = {
    "fundingRate": "0.000013",
    "nextFundingRate": 1779908400000,
    "openInterest": "96465777.457609",
    "openInterestBase": "1289.07192",
    "markPrice": "74839.54806",
    "lastPrice": "74880",
}

MARKETS_RESPONSE = {
    "status": "OK",
    "data": [
        {
            "name": "BTC-USD",
            "type": "PERPETUAL",
            "active": True,
            "status": "ACTIVE",
            "marketStats": BTC_MARKET_STATS,
        },
        {
            "name": "ETH-USD",
            "type": "PERPETUAL",
            "active": True,
            "status": "ACTIVE",
            "marketStats": {
                "fundingRate": "0.000005",
                "nextFundingRate": 1779908400000,
                "openInterest": "5000000",
                "openInterestBase": "1666.67",
                "markPrice": "3000.0",
                "lastPrice": "3001",
            },
        },
    ],
}


def make_transport(*responses: tuple[int, dict]) -> httpx.MockTransport:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if not queue:
            raise RuntimeError(f"Unexpected request to {request.url}")
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_declares_all_four_dimensions():
    assert ExtendedClient.COVERAGE["orderbook"] == Coverage.REST
    assert ExtendedClient.COVERAGE["funding"] == Coverage.REST
    assert ExtendedClient.COVERAGE["open_interest"] == Coverage.REST
    assert ExtendedClient.COVERAGE["liquidations"] == Coverage.WS_TAIL


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orderbook_returns_sorted_bids_and_asks():
    transport = make_transport((200, ORDERBOOK_RESPONSE))
    async with ExtendedClient(httpx.AsyncClient(transport=transport)) as c:
        ob = await c.get_orderbook("BTC")

    assert isinstance(ob, OrderBook)
    assert ob.venue == "extended"
    assert ob.token == "BTC"
    bid_prices = [b.price for b in ob.bids]
    assert bid_prices == sorted(bid_prices, reverse=True)
    ask_prices = [a.price for a in ob.asks]
    assert ask_prices == sorted(ask_prices)
    assert ob.bids[0].price < ob.asks[0].price


@pytest.mark.asyncio
async def test_orderbook_market_name_is_token_dash_usd():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=ORDERBOOK_RESPONSE)

    async with ExtendedClient(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as c:
        await c.get_orderbook("btc")

    assert any("BTC-USD" in u for u in seen)


@pytest.mark.asyncio
async def test_orderbook_raises_token_not_listed_on_empty_book():
    empty = {"status": "OK", "data": {"market": "FAKE-USD", "bid": [], "ask": []}}
    transport = make_transport((200, empty))
    async with ExtendedClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_orderbook("FAKE")


@pytest.mark.asyncio
async def test_orderbook_raises_venue_unavailable_on_http_error():
    transport = make_transport((500, {"status": "ERROR"}))
    async with ExtendedClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_orderbook("BTC")


@pytest.mark.asyncio
async def test_orderbook_raises_token_not_listed_on_404():
    transport = make_transport((404, {"status": "NOT_FOUND"}))
    async with ExtendedClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_orderbook("FAKE")


# ---------------------------------------------------------------------------
# Funding rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_funding_rate_annualizes_with_1h_period():
    transport = make_transport((200, MARKETS_RESPONSE))
    async with ExtendedClient(httpx.AsyncClient(transport=transport)) as c:
        fr = await c.get_funding_rate("BTC")

    assert isinstance(fr, FundingRate)
    assert fr.venue == "extended"
    assert fr.token == "BTC"
    assert fr.period_hours == 1.0
    expected_apr = 0.000013 * (8760 / 1.0)
    assert abs(fr.apr_annualized - expected_apr) < 1e-9
    assert fr.rate_per_period == pytest.approx(0.000013)


@pytest.mark.asyncio
async def test_funding_rate_next_funding_parsed_from_ms_epoch():
    transport = make_transport((200, MARKETS_RESPONSE))
    async with ExtendedClient(httpx.AsyncClient(transport=transport)) as c:
        fr = await c.get_funding_rate("BTC")

    expected = datetime.fromtimestamp(1779908400000 / 1000, tz=timezone.utc)
    assert fr.next_funding_at == expected


@pytest.mark.asyncio
async def test_funding_rate_raises_token_not_listed_for_unknown_market():
    transport = make_transport((200, MARKETS_RESPONSE))
    async with ExtendedClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_funding_rate("XYZ")


@pytest.mark.asyncio
async def test_funding_rate_raises_venue_unavailable_on_http_error():
    transport = make_transport((502, {"status": "ERROR"}))
    async with ExtendedClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_funding_rate("BTC")


# ---------------------------------------------------------------------------
# Open interest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_interest_uses_base_asset_and_mark_price():
    transport = make_transport((200, MARKETS_RESPONSE))
    async with ExtendedClient(httpx.AsyncClient(transport=transport)) as c:
        oi = await c.get_open_interest("BTC")

    assert isinstance(oi, OpenInterest)
    assert oi.venue == "extended"
    assert oi.token == "BTC"
    assert oi.oi_base == pytest.approx(1289.07192)
    expected_usd = 1289.07192 * 74839.54806
    assert oi.oi_usd == pytest.approx(expected_usd, rel=1e-6)
    assert oi.mark_price == pytest.approx(74839.54806)


@pytest.mark.asyncio
async def test_open_interest_shares_http_call_with_funding(monkeypatch):
    """funding + OI back-to-back should only make one HTTP call (TTL cache)."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if "markets" in str(request.url) and "orderbook" not in str(request.url):
            call_count += 1
        return httpx.Response(200, json=MARKETS_RESPONSE)

    transport = httpx.MockTransport(handler)
    async with ExtendedClient(httpx.AsyncClient(transport=transport)) as c:
        await c.get_funding_rate("BTC")
        await c.get_open_interest("BTC")

    assert call_count == 1, f"Expected 1 markets call, got {call_count}"


# ---------------------------------------------------------------------------
# Liquidations (WS tail)
# ---------------------------------------------------------------------------


def _make_ws_messages(*batches: list[dict]) -> list[str]:
    """Encode trade batches as JSON WS messages."""
    msgs = []
    for i, trades in enumerate(batches):
        msgs.append(json.dumps({"ts": 1779907000000 + i, "data": trades, "seq": i + 1}))
    return msgs


@pytest.mark.asyncio
async def test_liquidations_captures_long_liquidation_from_ws():
    """S='SELL' on tT='LIQUIDATION' means a long was force-closed."""
    liq_trade = {
        "m": "BTC-USD", "S": "SELL", "tT": "LIQUIDATION",
        "T": 1779907000000, "p": "74500", "q": "0.5", "i": 999,
    }
    messages = _make_ws_messages([liq_trade])

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.recv = AsyncMock(side_effect=messages + [asyncio.TimeoutError()])

    with patch("perp_liquidity.fetchers.extended.websockets.connect", return_value=mock_ws):
        async with ExtendedClient(httpx.AsyncClient(transport=make_transport())) as c:
            liqs = await c.get_recent_liquidations("BTC", lookback_seconds=5)

    assert len(liqs) == 1
    assert liqs[0].side == "long"
    assert liqs[0].price == pytest.approx(74500.0)
    assert liqs[0].qty_base == pytest.approx(0.5)
    assert liqs[0].qty_usd == pytest.approx(74500.0 * 0.5)
    assert liqs[0].venue == "extended"
    assert liqs[0].token == "BTC"


@pytest.mark.asyncio
async def test_liquidations_captures_short_liquidation_from_ws():
    """S='BUY' on tT='LIQUIDATION' means a short was force-closed."""
    liq_trade = {
        "m": "BTC-USD", "S": "BUY", "tT": "LIQUIDATION",
        "T": 1779907000000, "p": "75000", "q": "1.0", "i": 1000,
    }
    messages = _make_ws_messages([liq_trade])

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.recv = AsyncMock(side_effect=messages + [asyncio.TimeoutError()])

    with patch("perp_liquidity.fetchers.extended.websockets.connect", return_value=mock_ws):
        async with ExtendedClient(httpx.AsyncClient(transport=make_transport())) as c:
            liqs = await c.get_recent_liquidations("BTC", lookback_seconds=5)

    assert len(liqs) == 1
    assert liqs[0].side == "short"


@pytest.mark.asyncio
async def test_liquidations_ignores_regular_trades():
    """tT='TRADE' messages must not be included in results."""
    regular_trade = {
        "m": "BTC-USD", "S": "BUY", "tT": "TRADE",
        "T": 1779907000000, "p": "75000", "q": "0.1", "i": 1001,
    }
    messages = _make_ws_messages([regular_trade])

    mock_ws = AsyncMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.recv = AsyncMock(side_effect=messages + [asyncio.TimeoutError()])

    with patch("perp_liquidity.fetchers.extended.websockets.connect", return_value=mock_ws):
        async with ExtendedClient(httpx.AsyncClient(transport=make_transport())) as c:
            liqs = await c.get_recent_liquidations("BTC", lookback_seconds=5)

    assert liqs == []


@pytest.mark.asyncio
async def test_liquidations_raises_venue_unavailable_on_ws_failure():
    with patch(
        "perp_liquidity.fetchers.extended.websockets.connect",
        side_effect=Exception("connection refused"),
    ):
        async with ExtendedClient(httpx.AsyncClient(transport=make_transport())) as c:
            with pytest.raises(VenueUnavailable):
                await c.get_recent_liquidations("BTC", lookback_seconds=5)
