"""
Tests for liquidations analyzer.

Pure function tests -- no IO. Tests use hand-crafted Liquidation objects.

Conventions:
  - summarize_liquidations takes a list[Liquidation] and returns LiquidationSummary
  - side='long' means a long was force-closed (forced sell)
  - side='short' means a short was force-closed (forced buy)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from perp_liquidity.fetchers.base import Liquidation
from perp_liquidity.analyzers.liquidations import LiquidationSummary, summarize_liquidations


def make_liq(
    venue: str = "hyperliquid",
    side: str = "long",
    qty_usd: float = 10_000.0,
    price: float = 70_000.0,
    occurred_at: datetime | None = None,
) -> Liquidation:
    if occurred_at is None:
        occurred_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Liquidation(
        venue=venue,
        token="BTC",
        side=side,
        price=price,
        qty_base=qty_usd / price,
        qty_usd=qty_usd,
        occurred_at=occurred_at,
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def test_summarize_returns_summary():
    liqs = [make_liq()]
    result = summarize_liquidations(liqs)
    assert isinstance(result, LiquidationSummary)


def test_summarize_empty_returns_zero_summary():
    result = summarize_liquidations([])
    assert result.total_usd == 0.0
    assert result.count == 0
    assert result.long_usd == 0.0
    assert result.short_usd == 0.0
    assert result.largest_usd == 0.0


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------


def test_total_usd_sums_all():
    liqs = [make_liq(qty_usd=10_000), make_liq(qty_usd=5_000), make_liq(qty_usd=3_000)]
    result = summarize_liquidations(liqs)
    assert result.total_usd == pytest.approx(18_000.0)


def test_count_is_number_of_events():
    liqs = [make_liq() for _ in range(5)]
    result = summarize_liquidations(liqs)
    assert result.count == 5


# ---------------------------------------------------------------------------
# Long / short split
# ---------------------------------------------------------------------------


def test_long_usd_sums_long_side_only():
    liqs = [
        make_liq(side="long", qty_usd=10_000),
        make_liq(side="short", qty_usd=4_000),
        make_liq(side="long", qty_usd=6_000),
    ]
    result = summarize_liquidations(liqs)
    assert result.long_usd == pytest.approx(16_000.0)
    assert result.short_usd == pytest.approx(4_000.0)


def test_long_count_and_short_count():
    liqs = [
        make_liq(side="long"),
        make_liq(side="long"),
        make_liq(side="short"),
    ]
    result = summarize_liquidations(liqs)
    assert result.long_count == 2
    assert result.short_count == 1


def test_all_longs():
    liqs = [make_liq(side="long", qty_usd=5_000) for _ in range(3)]
    result = summarize_liquidations(liqs)
    assert result.short_usd == pytest.approx(0.0)
    assert result.short_count == 0


# ---------------------------------------------------------------------------
# Largest liquidation
# ---------------------------------------------------------------------------


def test_largest_usd_is_max_single_event():
    liqs = [
        make_liq(qty_usd=10_000),
        make_liq(qty_usd=500_000),
        make_liq(qty_usd=25_000),
    ]
    result = summarize_liquidations(liqs)
    assert result.largest_usd == pytest.approx(500_000.0)


def test_largest_liq_event_stored():
    """largest_event is the full Liquidation object for the biggest hit."""
    big = make_liq(qty_usd=500_000)
    liqs = [make_liq(qty_usd=10_000), big, make_liq(qty_usd=25_000)]
    result = summarize_liquidations(liqs)
    assert result.largest_event is big


def test_largest_event_none_when_empty():
    result = summarize_liquidations([])
    assert result.largest_event is None


# ---------------------------------------------------------------------------
# Per-venue breakdown
# ---------------------------------------------------------------------------


def test_by_venue_groups_by_venue():
    liqs = [
        make_liq(venue="hyperliquid", qty_usd=10_000),
        make_liq(venue="lighter", qty_usd=5_000),
        make_liq(venue="hyperliquid", qty_usd=3_000),
    ]
    result = summarize_liquidations(liqs)
    assert result.by_venue["hyperliquid"].total_usd == pytest.approx(13_000.0)
    assert result.by_venue["lighter"].total_usd == pytest.approx(5_000.0)


def test_by_venue_empty_when_no_liqs():
    result = summarize_liquidations([])
    assert result.by_venue == {}


def test_by_venue_count_correct():
    liqs = [
        make_liq(venue="hyperliquid"),
        make_liq(venue="hyperliquid"),
        make_liq(venue="extended"),
    ]
    result = summarize_liquidations(liqs)
    assert result.by_venue["hyperliquid"].count == 2
    assert result.by_venue["extended"].count == 1
