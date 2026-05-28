"""
Tests for open interest analyzer.

Pure function tests -- no IO. Tests use hand-crafted OpenInterest objects.

Conventions:
  - rank_open_interest returns list sorted by oi_usd descending (largest first)
  - OIRankRow wraps the original OpenInterest with rank and market_share_pct
  - market_share_pct sums to 100.0 across all rows
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from perp_liquidity.fetchers.base import OpenInterest
from perp_liquidity.analyzers.open_interest import OIRankRow, rank_open_interest


def make_oi(venue: str, oi_usd: float, mark_price: float = 100_000.0) -> OpenInterest:
    oi_base = oi_usd / mark_price
    return OpenInterest(
        venue=venue,
        token="BTC",
        oi_base=oi_base,
        oi_usd=oi_usd,
        mark_price=mark_price,
        fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# rank_open_interest
# ---------------------------------------------------------------------------


def test_rank_oi_returns_list_of_rank_rows():
    ois = [make_oi("hl", 2_000_000), make_oi("px", 500_000)]
    rows = rank_open_interest(ois)
    assert all(isinstance(r, OIRankRow) for r in rows)


def test_rank_oi_sorted_by_usd_descending():
    ois = [
        make_oi("px", 500_000),
        make_oi("hl", 2_000_000),
        make_oi("lt", 1_000_000),
    ]
    rows = rank_open_interest(ois)
    usd_values = [r.open_interest.oi_usd for r in rows]
    assert usd_values == sorted(usd_values, reverse=True)


def test_rank_oi_rank_1_is_largest():
    ois = [make_oi("hl", 2_000_000), make_oi("px", 500_000)]
    rows = rank_open_interest(ois)
    assert rows[0].rank == 1
    assert rows[0].open_interest.venue == "hl"


def test_rank_oi_rank_increments():
    ois = [make_oi("a", 3_000), make_oi("b", 2_000), make_oi("c", 1_000)]
    rows = rank_open_interest(ois)
    assert [r.rank for r in rows] == [1, 2, 3]


def test_rank_oi_market_share_sums_to_100():
    ois = [make_oi("hl", 2_000_000), make_oi("px", 500_000), make_oi("lt", 500_000)]
    rows = rank_open_interest(ois)
    total = sum(r.market_share_pct for r in rows)
    assert total == pytest.approx(100.0, abs=1e-6)


def test_rank_oi_market_share_proportional():
    ois = [make_oi("hl", 3_000_000), make_oi("px", 1_000_000)]
    rows = rank_open_interest(ois)
    # hl = 75%, px = 25%
    assert rows[0].market_share_pct == pytest.approx(75.0)
    assert rows[1].market_share_pct == pytest.approx(25.0)


def test_rank_oi_empty_returns_empty():
    assert rank_open_interest([]) == []


def test_rank_oi_single_item_100_pct():
    rows = rank_open_interest([make_oi("hl", 1_000_000)])
    assert len(rows) == 1
    assert rows[0].rank == 1
    assert rows[0].market_share_pct == pytest.approx(100.0)


def test_rank_oi_total_usd_on_row():
    """Each row should expose total_usd (sum across all venues)."""
    ois = [make_oi("hl", 3_000_000), make_oi("px", 1_000_000)]
    rows = rank_open_interest(ois)
    assert rows[0].total_usd == pytest.approx(4_000_000)
    assert rows[1].total_usd == pytest.approx(4_000_000)
