"""
Tests for EdgeXClient.

Uses httpx.MockTransport for all REST calls. No WS needed (liquidations
are NOT_AVAILABLE -- private channel only, requires ECDSA auth).

API shapes confirmed from live probing 2026-05-27:
- Meta:     GET /api/v1/public/meta/getMetaData
  -> {"code":"SUCCESS","data":[{"global":{...},"contractList":[{"contractId":"10000001","contractName":"BTCUSD",...}]}]}
  Note: data is a list with one element containing contractList.
- Depth:    GET /api/v1/public/quote/getDepth?contractId=10000001&level=200
  -> {"code":"SUCCESS","data":[{"contractId":"10000001","contractName":"BTCUSD",
       "asks":[{"price":"75031.7","size":"0.130"},...],
       "bids":[{"price":"75030.1","size":"0.200"},...]}]}
- Ticker:   GET /api/v1/public/quote/getTicker?contractId=10000001
  -> {"code":"SUCCESS","data":[{"contractId":"10000001","markPrice":"75081.0...",
       "openInterest":"4064.812","fundingRate":"0.00005000",
       "fundingTime":"1779912000000","nextFundingTime":"1779926400000"}]}
  fundingRateIntervalMin=240 => 4h funding period
  openInterest is in base asset (BTC)
- Liquidations: NOT_AVAILABLE (START_LIQUIDATING on private WS only, confirmed 2026-05-27)
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from perp_liquidity.fetchers.edgex import EdgeXClient
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

# getMetaData returns a list with one element containing contractList
META_RESPONSE = {
    "code": "SUCCESS",
    "data": [
        {
            "global": {"appName": "edgeX"},
            "contractList": [
                {
                    "contractId": "10000001",
                    "contractName": "BTCUSD",
                    "tickSize": "0.1",
                    "stepSize": "0.001",
                },
                {
                    "contractId": "10000002",
                    "contractName": "ETHUSD",
                    "tickSize": "0.01",
                    "stepSize": "0.01",
                },
            ],
        }
    ],
}

DEPTH_RESPONSE = {
    "code": "SUCCESS",
    "data": [
        {
            "startVersion": "4037435251",
            "endVersion": "4037435293",
            "level": 200,
            "contractId": "10000001",
            "contractName": "BTCUSD",
            "asks": [
                {"price": "75031.7", "size": "0.130"},
                {"price": "75033.0", "size": "5.377"},
            ],
            "bids": [
                {"price": "75030.1", "size": "0.200"},
                {"price": "75029.5", "size": "1.500"},
            ],
        }
    ],
}

TICKER_RESPONSE = {
    "code": "SUCCESS",
    "data": [
        {
            "contractId": "10000001",
            "contractName": "BTCUSD",
            "lastPrice": "75058.4",
            "markPrice": "75081.09802",
            "indexPrice": "75063.68",
            "openInterest": "4064.812",
            "fundingRate": "0.00005000",
            "fundingTime": "1779912000000",
            "nextFundingTime": "1779926400000",
        }
    ],
}


def make_transport(*responses: tuple[int, dict]) -> httpx.MockTransport:
    """Return a MockTransport that plays back responses in order."""
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
    assert EdgeXClient.COVERAGE["orderbook"] == Coverage.REST
    assert EdgeXClient.COVERAGE["funding"] == Coverage.REST
    assert EdgeXClient.COVERAGE["open_interest"] == Coverage.REST
    assert EdgeXClient.COVERAGE["liquidations"] == Coverage.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Contract ID resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_id_resolved_from_meta():
    """BTC -> contractId 10000001 via getMetaData contractList."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "getMetaData" in str(request.url):
            return httpx.Response(200, json=META_RESPONSE)
        return httpx.Response(200, json=DEPTH_RESPONSE)

    async with EdgeXClient(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as c:
        await c.get_orderbook("BTC")

    assert any("getMetaData" in u for u in seen)
    assert any("10000001" in u for u in seen)


@pytest.mark.asyncio
async def test_contract_id_cached_across_calls():
    """getMetaData should only be called once even if multiple methods use it."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if "getMetaData" in str(request.url):
            call_count += 1
            return httpx.Response(200, json=META_RESPONSE)
        if "getDepth" in str(request.url):
            return httpx.Response(200, json=DEPTH_RESPONSE)
        if "getTicker" in str(request.url):
            return httpx.Response(200, json=TICKER_RESPONSE)
        raise RuntimeError(f"Unexpected: {request.url}")

    async with EdgeXClient(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as c:
        await c.get_orderbook("BTC")
        await c.get_funding_rate("BTC")
        await c.get_open_interest("BTC")

    assert call_count == 1, f"Expected 1 getMetaData call, got {call_count}"


@pytest.mark.asyncio
async def test_unknown_token_raises_token_not_listed():
    """Token not in contractList raises TokenNotListed."""
    transport = make_transport((200, META_RESPONSE))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_orderbook("XYZ")


@pytest.mark.asyncio
async def test_contract_id_resolved_from_meta_dict_shape():
    """API changed: data is now a dict directly, not a list wrapper."""
    meta_dict_shape = {
        "code": "SUCCESS",
        "data": {
            "global": {"appName": "edgeX"},
            "contractList": [
                {"contractId": "10000001", "contractName": "BTCUSD"},
                {"contractId": "10000002", "contractName": "ETHUSD"},
            ],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "getMetaData" in str(request.url):
            return httpx.Response(200, json=meta_dict_shape)
        return httpx.Response(200, json=DEPTH_RESPONSE)

    async with EdgeXClient(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as c:
        ob = await c.get_orderbook("BTC")

    assert ob.venue == "edgex"
    assert ob.token == "BTC"


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orderbook_returns_sorted_bids_and_asks():
    transport = make_transport((200, META_RESPONSE), (200, DEPTH_RESPONSE))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        ob = await c.get_orderbook("BTC")

    assert isinstance(ob, OrderBook)
    assert ob.venue == "edgex"
    assert ob.token == "BTC"
    bid_prices = [b.price for b in ob.bids]
    assert bid_prices == sorted(bid_prices, reverse=True)
    ask_prices = [a.price for a in ob.asks]
    assert ask_prices == sorted(ask_prices)
    assert ob.bids[0].price < ob.asks[0].price


@pytest.mark.asyncio
async def test_orderbook_token_uppercased():
    """Token 'btc' -> 'BTC' in the returned OrderBook."""
    transport = make_transport((200, META_RESPONSE), (200, DEPTH_RESPONSE))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        ob = await c.get_orderbook("btc")

    assert ob.token == "BTC"


@pytest.mark.asyncio
async def test_orderbook_raises_token_not_listed_on_empty_book():
    empty = {
        "code": "SUCCESS",
        "data": [{"contractId": "10000001", "contractName": "BTCUSD", "asks": [], "bids": []}],
    }
    transport = make_transport((200, META_RESPONSE), (200, empty))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_orderbook("BTC")


@pytest.mark.asyncio
async def test_orderbook_raises_venue_unavailable_on_http_error():
    transport = make_transport((200, META_RESPONSE), (500, {"code": "ERROR"}))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_orderbook("BTC")


@pytest.mark.asyncio
async def test_orderbook_raises_venue_unavailable_on_meta_http_error():
    transport = make_transport((503, {"code": "ERROR"}))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_orderbook("BTC")


# ---------------------------------------------------------------------------
# Funding rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_funding_rate_annualizes_with_4h_period():
    transport = make_transport((200, META_RESPONSE), (200, TICKER_RESPONSE))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        fr = await c.get_funding_rate("BTC")

    assert isinstance(fr, FundingRate)
    assert fr.venue == "edgex"
    assert fr.token == "BTC"
    assert fr.period_hours == 4.0
    expected_apr = 0.00005 * (8760 / 4.0)
    assert abs(fr.apr_annualized - expected_apr) < 1e-9
    assert fr.rate_per_period == pytest.approx(0.00005)


@pytest.mark.asyncio
async def test_funding_rate_next_funding_parsed_from_ms_epoch():
    transport = make_transport((200, META_RESPONSE), (200, TICKER_RESPONSE))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        fr = await c.get_funding_rate("BTC")

    expected = datetime.fromtimestamp(1779926400000 / 1000, tz=timezone.utc)
    assert fr.next_funding_at == expected


@pytest.mark.asyncio
async def test_funding_rate_raises_venue_unavailable_on_http_error():
    transport = make_transport((200, META_RESPONSE), (502, {"code": "ERROR"}))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_funding_rate("BTC")


# ---------------------------------------------------------------------------
# Open interest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_interest_uses_base_asset_and_mark_price():
    transport = make_transport((200, META_RESPONSE), (200, TICKER_RESPONSE))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        oi = await c.get_open_interest("BTC")

    assert isinstance(oi, OpenInterest)
    assert oi.venue == "edgex"
    assert oi.token == "BTC"
    assert oi.oi_base == pytest.approx(4064.812)
    expected_usd = 4064.812 * 75081.09802
    assert oi.oi_usd == pytest.approx(expected_usd, rel=1e-6)
    assert oi.mark_price == pytest.approx(75081.09802)


@pytest.mark.asyncio
async def test_open_interest_shares_ticker_call_with_funding():
    """funding + OI back-to-back should only make one getTicker call (TTL cache)."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if "getMetaData" in str(request.url):
            return httpx.Response(200, json=META_RESPONSE)
        if "getTicker" in str(request.url):
            call_count += 1
            return httpx.Response(200, json=TICKER_RESPONSE)
        raise RuntimeError(f"Unexpected: {request.url}")

    async with EdgeXClient(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as c:
        await c.get_funding_rate("BTC")
        await c.get_open_interest("BTC")

    assert call_count == 1, f"Expected 1 getTicker call, got {call_count}"


@pytest.mark.asyncio
async def test_open_interest_raises_venue_unavailable_on_http_error():
    transport = make_transport((200, META_RESPONSE), (500, {"code": "ERROR"}))
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_open_interest("BTC")


# ---------------------------------------------------------------------------
# Liquidations: NOT_AVAILABLE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_liquidations_raises_not_implemented():
    """EdgeX liquidations require ECDSA-signed private WS channel."""
    transport = make_transport()  # no requests expected
    async with EdgeXClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(NotImplementedError):
            await c.get_recent_liquidations("BTC")
