"""
Tests for funding rate analyzer.

Pure function tests -- no IO. Tests use hand-crafted FundingRate objects.

Conventions:
  - rank_funding returns a list sorted by apr_annualized ascending (most negative first)
  - FundingRankRow wraps the original FundingRate with rank (1=best) and market_share_pct
  - detect_flips returns venues where sign of rate_per_period differs from the rest
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from perp_liquidity.fetchers.base import FundingRate
from perp_liquidity.analyzers.funding import FundingRankRow, rank_funding, detect_flips


def make_fr(venue: str, rate: float, period_hours: float = 8.0) -> FundingRate:
    apr = rate * (8760 / period_hours)
    return FundingRate(
        venue=venue,
        token="BTC",
        rate_per_period=rate,
        period_hours=period_hours,
        apr_annualized=apr,
        next_funding_at=None,
        fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# rank_funding
# ---------------------------------------------------------------------------


def test_rank_funding_returns_list_of_rank_rows():
    rates = [make_fr("hl", 0.01), make_fr("px", -0.002)]
    rows = rank_funding(rates)
    assert all(isinstance(r, FundingRankRow) for r in rows)


def test_rank_funding_sorted_by_apr_ascending():
    """Most negative APR first (rank 1)."""
    rates = [
        make_fr("hl", 0.01),
        make_fr("px", -0.002),
        make_fr("lt", 0.005),
    ]
    rows = rank_funding(rates)
    aprs = [r.funding_rate.apr_annualized for r in rows]
    assert aprs == sorted(aprs)


def test_rank_funding_rank_1_is_most_negative():
    rates = [make_fr("hl", 0.01), make_fr("px", -0.002)]
    rows = rank_funding(rates)
    assert rows[0].rank == 1
    assert rows[0].funding_rate.venue == "px"


def test_rank_funding_rank_increments():
    rates = [make_fr("a", 0.01), make_fr("b", 0.02), make_fr("c", 0.03)]
    rows = rank_funding(rates)
    assert [r.rank for r in rows] == [1, 2, 3]


def test_rank_funding_spread_bps():
    """spread_bps = (max_apr - min_apr) * 10_000 on the last row."""
    rates = [make_fr("hl", 0.001), make_fr("px", 0.01)]
    rows = rank_funding(rates)
    apr_low = rates[0].apr_annualized  # hl
    apr_high = rates[1].apr_annualized  # px
    # spread_bps available on each row as a property or field
    expected = (apr_high - apr_low) * 10_000
    assert rows[-1].spread_bps == pytest.approx(expected)
    assert rows[0].spread_bps == pytest.approx(expected)


def test_rank_funding_empty_returns_empty():
    assert rank_funding([]) == []


def test_rank_funding_single_item():
    rows = rank_funding([make_fr("hl", 0.01)])
    assert len(rows) == 1
    assert rows[0].rank == 1
    assert rows[0].spread_bps == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# detect_flips
# ---------------------------------------------------------------------------


def test_detect_flips_returns_venues_with_opposite_sign():
    """If most venues are positive, flips are the negative ones."""
    rates = [
        make_fr("hl", 0.01),
        make_fr("lt", 0.005),
        make_fr("px", -0.002),  # flip
        make_fr("ax", 0.003),
    ]
    flips = detect_flips(rates)
    assert set(flips) == {"px"}


def test_detect_flips_empty_when_all_same_sign():
    rates = [make_fr("hl", 0.01), make_fr("lt", 0.005), make_fr("px", 0.002)]
    assert detect_flips(rates) == []


def test_detect_flips_empty_when_empty_input():
    assert detect_flips([]) == []


def test_detect_flips_empty_when_single_item():
    assert detect_flips([make_fr("hl", 0.01)]) == []


def test_detect_flips_majority_negative_flags_positives():
    rates = [
        make_fr("hl", -0.01),
        make_fr("lt", -0.005),
        make_fr("px", 0.002),  # flip
        make_fr("ax", -0.003),
    ]
    flips = detect_flips(rates)
    assert set(flips) == {"px"}


def test_detect_flips_exact_split_returns_empty():
    """50/50 split: no majority, no flips flagged."""
    rates = [make_fr("hl", 0.01), make_fr("px", -0.002)]
    assert detect_flips(rates) == []
