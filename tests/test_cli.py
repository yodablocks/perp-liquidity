"""
Tests for CLI orchestration layer.

Tests inject fake PerpDEXClient subclasses into run_analysis() to avoid
any real network calls. format_report() is tested separately as a pure
string function.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from perp_liquidity.fetchers.base import (
    FundingRate,
    OpenInterest,
    OrderBook,
    OrderBookLevel,
    TokenNotListed,
    VenueUnavailable,
)
from perp_liquidity.cli import AnalysisReport, run_analysis
from perp_liquidity.output import format_report


# ---------------------------------------------------------------------------
# Fake fetcher helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ob(venue: str, token: str) -> OrderBook:
    return OrderBook(
        venue=venue,
        token=token,
        bids=[OrderBookLevel(price=99_000.0, qty=10.0)],
        asks=[OrderBookLevel(price=101_000.0, qty=10.0)],
        fetched_at=_NOW,
    )


def _fr(venue: str, token: str, rate: float = 0.001) -> FundingRate:
    return FundingRate(
        venue=venue,
        token=token,
        rate_per_period=rate,
        period_hours=8.0,
        apr_annualized=rate * (8760 / 8.0),
        fetched_at=_NOW,
    )


def _oi(venue: str, token: str, oi_usd: float = 1_000_000.0) -> OpenInterest:
    return OpenInterest(
        venue=venue,
        token=token,
        oi_base=oi_usd / 100_000.0,
        oi_usd=oi_usd,
        mark_price=100_000.0,
        fetched_at=_NOW,
    )


def make_success_client(venue: str, oi_usd: float = 1_000_000.0, rate: float = 0.001):
    """Return a fake client class that returns valid data for a given venue."""
    from perp_liquidity.fetchers.base import Coverage, PerpDEXClient

    class FakeClient(PerpDEXClient):
        VENUE = venue
        COVERAGE = {
            "orderbook": Coverage.REST,
            "funding": Coverage.REST,
            "open_interest": Coverage.REST,
            "liquidations": Coverage.NOT_AVAILABLE,
        }

        async def get_orderbook(self, token: str) -> OrderBook:
            return _ob(self.VENUE, token.upper())

        async def get_funding_rate(self, token: str) -> FundingRate:
            return _fr(self.VENUE, token.upper(), rate)

        async def get_open_interest(self, token: str) -> OpenInterest:
            return _oi(self.VENUE, token.upper(), oi_usd)

        async def get_recent_liquidations(self, token, *, lookback_seconds=60):
            raise NotImplementedError

    FakeClient.__name__ = f"Fake_{venue}"
    return FakeClient


def make_token_not_listed_client(venue: str):
    from perp_liquidity.fetchers.base import Coverage, PerpDEXClient

    class FakeClient(PerpDEXClient):
        VENUE = venue
        COVERAGE = {
            "orderbook": Coverage.REST,
            "funding": Coverage.REST,
            "open_interest": Coverage.REST,
            "liquidations": Coverage.NOT_AVAILABLE,
        }

        async def get_orderbook(self, token: str) -> OrderBook:
            raise TokenNotListed(self.VENUE, f"{token} not listed")

        async def get_funding_rate(self, token: str) -> FundingRate:
            raise TokenNotListed(self.VENUE, f"{token} not listed")

        async def get_open_interest(self, token: str) -> OpenInterest:
            raise TokenNotListed(self.VENUE, f"{token} not listed")

        async def get_recent_liquidations(self, token, *, lookback_seconds=60):
            raise NotImplementedError

    FakeClient.__name__ = f"Fake_notlisted_{venue}"
    return FakeClient


def make_unavailable_client(venue: str):
    from perp_liquidity.fetchers.base import Coverage, PerpDEXClient

    class FakeClient(PerpDEXClient):
        VENUE = venue
        COVERAGE = {
            "orderbook": Coverage.REST,
            "funding": Coverage.REST,
            "open_interest": Coverage.REST,
            "liquidations": Coverage.NOT_AVAILABLE,
        }

        async def get_orderbook(self, token: str) -> OrderBook:
            raise VenueUnavailable(self.VENUE, "connection refused")

        async def get_funding_rate(self, token: str) -> FundingRate:
            raise VenueUnavailable(self.VENUE, "connection refused")

        async def get_open_interest(self, token: str) -> OpenInterest:
            raise VenueUnavailable(self.VENUE, "connection refused")

        async def get_recent_liquidations(self, token, *, lookback_seconds=60):
            raise NotImplementedError

    FakeClient.__name__ = f"Fake_unavailable_{venue}"
    return FakeClient


# ---------------------------------------------------------------------------
# AnalysisReport type
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_analysis_returns_analysis_report():
    clients = [make_success_client("hl")]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    assert isinstance(report, AnalysisReport)


@pytest.mark.asyncio
async def test_report_token_uppercased():
    clients = [make_success_client("hl")]
    report = await run_analysis("btc", fetcher_classes=clients, clip_usds=[1_000])
    assert report.token == "BTC"


# ---------------------------------------------------------------------------
# Successful fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_analysis_populates_slippage():
    clients = [make_success_client("hl")]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000, 10_000])
    # 1 venue * 2 clips * 2 sides = 4
    assert len(report.slippage) == 4


@pytest.mark.asyncio
async def test_run_analysis_populates_funding():
    clients = [make_success_client("hl"), make_success_client("lt", rate=0.002)]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    assert len(report.funding) == 2
    assert report.funding[0].rank == 1


@pytest.mark.asyncio
async def test_run_analysis_populates_oi():
    clients = [make_success_client("hl", oi_usd=2_000_000), make_success_client("lt", oi_usd=500_000)]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    assert len(report.oi) == 2
    assert report.oi[0].open_interest.venue == "hl"


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_not_listed_goes_to_not_listed():
    clients = [make_success_client("hl"), make_token_not_listed_client("px")]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    assert "px" in report.not_listed
    assert len(report.funding) == 1  # only hl succeeded


@pytest.mark.asyncio
async def test_venue_unavailable_goes_to_failures():
    clients = [make_success_client("hl"), make_unavailable_client("lt")]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    assert any(v == "lt" for v, _ in report.failures)
    assert len(report.funding) == 1


@pytest.mark.asyncio
async def test_all_venues_fail_returns_empty_results():
    clients = [make_unavailable_client("hl"), make_unavailable_client("px")]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    assert report.slippage == []
    assert report.funding == []
    assert report.oi == []
    assert len(report.failures) == 2


@pytest.mark.asyncio
async def test_run_analysis_never_raises_on_venue_error():
    """run_analysis absorbs all venue errors; caller always gets a report."""
    clients = [make_unavailable_client("hl")]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    assert isinstance(report, AnalysisReport)


# ---------------------------------------------------------------------------
# format_report (output.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_format_report_contains_venue_names():
    clients = [make_success_client("hl"), make_success_client("lt")]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    output = format_report(report)
    assert "hl" in output
    assert "lt" in output


@pytest.mark.asyncio
async def test_format_report_contains_token():
    clients = [make_success_client("hl")]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    output = format_report(report)
    assert "BTC" in output


@pytest.mark.asyncio
async def test_format_report_notes_failures():
    clients = [make_success_client("hl"), make_unavailable_client("px")]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    output = format_report(report)
    assert "px" in output


@pytest.mark.asyncio
async def test_format_report_returns_string():
    clients = [make_success_client("hl")]
    report = await run_analysis("BTC", fetcher_classes=clients, clip_usds=[1_000])
    assert isinstance(format_report(report), str)
