"""
Tests for AsterClient.

Uses httpx.MockTransport to avoid real network calls. Each test injects
a fake response matching Aster's actual API shapes (confirmed from live probing).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from perp_liquidity.fetchers.aster import AsterClient
from perp_liquidity.fetchers.base import (
    Coverage,
    FundingRate,
    OpenInterest,
    OrderBook,
    TokenNotListed,
    VenueUnavailable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_transport(*responses: tuple[int, dict]) -> httpx.MockTransport:
    """Return a MockTransport that plays back responses in order."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if not queue:
            raise RuntimeError(f"Unexpected request to {request.url}")
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


DEPTH_RESPONSE = {
    "lastUpdateId": 999,
    "E": 1779901278000,
    "T": 1779901278000,
    "bids": [["75259.2", "0.039"], ["75258.0", "1.200"], ["75257.0", "2.500"]],
    "asks": [["75259.3", "0.008"], ["75260.0", "0.500"], ["75261.0", "1.000"]],
}

PREMIUM_INDEX_RESPONSE = {
    "symbol": "BTCUSDT",
    "markPrice": "75274.03357971",
    "indexPrice": "75309.12500000",
    "estimatedSettlePrice": "75184.42624241",
    "lastFundingRate": "0.00010000",
    "interestRate": "0.00010000",
    "nextFundingTime": 1779926400000,
    "time": 1779901278000,
}

OPEN_INTEREST_RESPONSE = {
    "symbol": "BTCUSDT",
    "openInterest": "5487.840",
    "time": 1779900791113,
}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_declares_all_four_dimensions():
    assert AsterClient.COVERAGE["orderbook"] == Coverage.REST
    assert AsterClient.COVERAGE["funding"] == Coverage.REST
    assert AsterClient.COVERAGE["open_interest"] == Coverage.REST
    assert AsterClient.COVERAGE["liquidations"] == Coverage.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orderbook_returns_sorted_bids_and_asks():
    transport = make_transport((200, DEPTH_RESPONSE))
    async with AsterClient(httpx.AsyncClient(transport=transport)) as c:
        ob = await c.get_orderbook("BTC")

    assert isinstance(ob, OrderBook)
    assert ob.venue == "aster"
    assert ob.token == "BTC"
    # bids descending
    bid_prices = [b.price for b in ob.bids]
    assert bid_prices == sorted(bid_prices, reverse=True)
    # asks ascending
    ask_prices = [a.price for a in ob.asks]
    assert ask_prices == sorted(ask_prices)
    # best bid < best ask
    assert ob.bids[0].price < ob.asks[0].price


@pytest.mark.asyncio
async def test_orderbook_symbol_is_uppercased_with_usdt_suffix():
    """Token 'btc' must hit the endpoint as 'BTCUSDT'."""
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json=DEPTH_RESPONSE)

    transport = httpx.MockTransport(handler)
    async with AsterClient(httpx.AsyncClient(transport=transport)) as c:
        await c.get_orderbook("btc")

    assert any("BTCUSDT" in u for u in seen_urls)


@pytest.mark.asyncio
async def test_orderbook_raises_token_not_listed_on_empty_book():
    empty = {"lastUpdateId": 1, "E": 0, "T": 0, "bids": [], "asks": []}
    transport = make_transport((200, empty))
    async with AsterClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(TokenNotListed):
            await c.get_orderbook("FAKE")


@pytest.mark.asyncio
async def test_orderbook_raises_venue_unavailable_on_http_error():
    transport = make_transport((500, {"msg": "internal error"}))
    async with AsterClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_orderbook("BTC")


# ---------------------------------------------------------------------------
# Funding rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_funding_rate_annualizes_with_8h_period():
    transport = make_transport((200, PREMIUM_INDEX_RESPONSE))
    async with AsterClient(httpx.AsyncClient(transport=transport)) as c:
        fr = await c.get_funding_rate("BTC")

    assert isinstance(fr, FundingRate)
    assert fr.venue == "aster"
    assert fr.token == "BTC"
    assert fr.period_hours == 8.0
    expected_apr = 0.00010000 * (8760 / 8.0)
    assert abs(fr.apr_annualized - expected_apr) < 1e-9
    assert fr.rate_per_period == pytest.approx(0.00010000)


@pytest.mark.asyncio
async def test_funding_rate_next_funding_at_is_parsed_from_ms_epoch():
    transport = make_transport((200, PREMIUM_INDEX_RESPONSE))
    async with AsterClient(httpx.AsyncClient(transport=transport)) as c:
        fr = await c.get_funding_rate("BTC")

    expected = datetime.fromtimestamp(1779926400000 / 1000, tz=timezone.utc)
    assert fr.next_funding_at == expected


@pytest.mark.asyncio
async def test_funding_rate_raises_venue_unavailable_on_http_error():
    transport = make_transport((502, {"msg": "bad gateway"}))
    async with AsterClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_funding_rate("BTC")


# ---------------------------------------------------------------------------
# Open interest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_interest_normalizes_to_usd_using_mark_price():
    # OI and markPrice come from two different endpoints; both must be called.
    transport = make_transport(
        (200, OPEN_INTEREST_RESPONSE),
        (200, PREMIUM_INDEX_RESPONSE),
    )
    async with AsterClient(httpx.AsyncClient(transport=transport)) as c:
        oi = await c.get_open_interest("BTC")

    assert isinstance(oi, OpenInterest)
    assert oi.venue == "aster"
    assert oi.token == "BTC"
    assert oi.oi_base == pytest.approx(5487.840)
    expected_usd = 5487.840 * 75274.03357971
    assert oi.oi_usd == pytest.approx(expected_usd, rel=1e-6)
    assert oi.mark_price == pytest.approx(75274.03357971)


@pytest.mark.asyncio
async def test_open_interest_raises_venue_unavailable_on_oi_http_error():
    transport = make_transport((500, {"msg": "error"}))
    async with AsterClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(VenueUnavailable):
            await c.get_open_interest("BTC")


# ---------------------------------------------------------------------------
# Liquidations: NOT_AVAILABLE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_liquidations_raises_not_implemented():
    """Aster forceOrders endpoint is out of maintenance; WS emits nothing."""
    transport = make_transport()  # no requests expected
    async with AsterClient(httpx.AsyncClient(transport=transport)) as c:
        with pytest.raises(NotImplementedError):
            await c.get_recent_liquidations("BTC")
