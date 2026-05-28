"""
Tests for slippage analyzer.

Pure function tests -- no IO, no fixtures. Tests use hand-crafted OrderBook
objects to verify exact arithmetic.

Conventions:
  - clip_size is in USD
  - slippage_bps is always positive (worse = larger)
  - mid_price = (best_bid + best_ask) / 2
  - effective_price = weighted-average fill price across consumed levels
  - side='buy' walks asks (ascending); side='sell' walks bids (descending)
  - partial=True when book depth is insufficient to fill the full clip
"""

from __future__ import annotations

import pytest

from perp_liquidity.fetchers.base import OrderBook, OrderBookLevel
from perp_liquidity.analyzers.slippage import SlippageResult, compute_slippage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_book(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    venue: str = "test",
    token: str = "BTC",
) -> OrderBook:
    """Build an OrderBook from (price, qty) tuples.

    Caller is responsible for sorting: bids descending, asks ascending.
    """
    return OrderBook(
        venue=venue,
        token=token,
        bids=[OrderBookLevel(price=p, qty=q) for p, q in bids],
        asks=[OrderBookLevel(price=p, qty=q) for p, q in asks],
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def test_slippage_result_is_a_dataclass():
    book = make_book(bids=[(99.0, 1.0)], asks=[(101.0, 1.0)])
    result = compute_slippage(book, clip_usd=50.0, side="buy")
    assert isinstance(result, SlippageResult)
    assert result.venue == "test"
    assert result.token == "BTC"
    assert result.side == "buy"


# ---------------------------------------------------------------------------
# Buy side (walks asks ascending)
# ---------------------------------------------------------------------------


def test_buy_single_level_full_fill():
    """Buy $100 with a single ask at 100.0 qty=2. Fill entire clip on first level."""
    book = make_book(bids=[(99.0, 1.0)], asks=[(100.0, 2.0)])
    result = compute_slippage(book, clip_usd=100.0, side="buy")

    assert result.clip_usd == 100.0
    assert result.filled_usd == pytest.approx(100.0)
    assert result.effective_price == pytest.approx(100.0)
    assert result.partial is False
    # mid = (99 + 100) / 2 = 99.5
    # slippage = (100.0 - 99.5) / 99.5 * 10000
    expected_bps = (100.0 - 99.5) / 99.5 * 10_000
    assert result.slippage_bps == pytest.approx(expected_bps)


def test_buy_partial_last_level():
    """Buy $150 across two ask levels; second level only partially consumed."""
    # Level 1: 100.0, qty=1 -> fills $100 (1 BTC * $100)
    # Level 2: 101.0, qty=5 -> need $50 more = 50/101 BTC
    book = make_book(bids=[(99.0, 1.0)], asks=[(100.0, 1.0), (101.0, 5.0)])
    result = compute_slippage(book, clip_usd=150.0, side="buy")

    assert result.filled_usd == pytest.approx(150.0)
    assert result.partial is False
    # effective_price = total_usd_spent / total_btc_bought
    #   = 150 / (1 + 50/101)
    btc_from_l1 = 1.0
    btc_from_l2 = 50.0 / 101.0
    total_btc = btc_from_l1 + btc_from_l2
    expected_price = 150.0 / total_btc
    assert result.effective_price == pytest.approx(expected_price, rel=1e-9)


def test_buy_exact_level_boundary():
    """Buy exactly depletes first level, second level not touched."""
    # Level 1: price=100, qty=1 -> $100 exactly
    book = make_book(bids=[(99.0, 1.0)], asks=[(100.0, 1.0), (110.0, 5.0)])
    result = compute_slippage(book, clip_usd=100.0, side="buy")

    assert result.filled_usd == pytest.approx(100.0)
    assert result.effective_price == pytest.approx(100.0)
    assert result.partial is False


def test_buy_insufficient_depth_returns_partial():
    """Book exhausted before clip filled: partial=True, filled_usd < clip_usd."""
    # Only $50 of asks available
    book = make_book(bids=[(99.0, 1.0)], asks=[(100.0, 0.5)])
    result = compute_slippage(book, clip_usd=200.0, side="buy")

    assert result.partial is True
    assert result.filled_usd == pytest.approx(50.0)
    assert result.clip_usd == 200.0


def test_buy_empty_asks_returns_partial_with_zero_fill():
    """No asks at all: partial=True, filled_usd=0, slippage_bps=0."""
    book = make_book(bids=[(99.0, 1.0)], asks=[])
    result = compute_slippage(book, clip_usd=1000.0, side="buy")

    assert result.partial is True
    assert result.filled_usd == pytest.approx(0.0)
    assert result.slippage_bps == pytest.approx(0.0)


def test_buy_slippage_bps_is_positive():
    """Buy always pays above mid: slippage_bps >= 0."""
    book = make_book(bids=[(99.0, 10.0)], asks=[(101.0, 10.0)])
    result = compute_slippage(book, clip_usd=500.0, side="buy")
    assert result.slippage_bps > 0


# ---------------------------------------------------------------------------
# Sell side (walks bids descending)
# ---------------------------------------------------------------------------


def test_sell_single_level_full_fill():
    """Sell $100 with best bid at 100.0 qty=2. Full fill on first level."""
    book = make_book(bids=[(100.0, 2.0)], asks=[(101.0, 1.0)])
    result = compute_slippage(book, clip_usd=100.0, side="sell")

    assert result.filled_usd == pytest.approx(100.0)
    assert result.effective_price == pytest.approx(100.0)
    assert result.partial is False
    # mid = (100 + 101) / 2 = 100.5
    # sell slippage: (100.5 - 100.0) / 100.5 * 10000 (mid above effective)
    expected_bps = (100.5 - 100.0) / 100.5 * 10_000
    assert result.slippage_bps == pytest.approx(expected_bps)


def test_sell_partial_last_level():
    """Sell $150 across two bid levels."""
    # Level 1: 100.0, qty=1 -> $100
    # Level 2: 99.0, qty=5 -> need $50 more = 50/99 BTC
    book = make_book(bids=[(100.0, 1.0), (99.0, 5.0)], asks=[(101.0, 1.0)])
    result = compute_slippage(book, clip_usd=150.0, side="sell")

    assert result.filled_usd == pytest.approx(150.0)
    assert result.partial is False
    btc_from_l1 = 1.0
    btc_from_l2 = 50.0 / 99.0
    total_btc = btc_from_l1 + btc_from_l2
    expected_price = 150.0 / total_btc
    assert result.effective_price == pytest.approx(expected_price, rel=1e-9)


def test_sell_insufficient_depth_returns_partial():
    book = make_book(bids=[(100.0, 0.5)], asks=[(101.0, 1.0)])
    result = compute_slippage(book, clip_usd=200.0, side="sell")

    assert result.partial is True
    assert result.filled_usd == pytest.approx(50.0)


def test_sell_empty_bids_returns_partial_with_zero_fill():
    book = make_book(bids=[], asks=[(101.0, 1.0)])
    result = compute_slippage(book, clip_usd=1000.0, side="sell")

    assert result.partial is True
    assert result.filled_usd == pytest.approx(0.0)
    assert result.slippage_bps == pytest.approx(0.0)


def test_sell_slippage_bps_is_positive():
    """Sell always receives below mid: slippage_bps >= 0."""
    book = make_book(bids=[(99.0, 10.0)], asks=[(101.0, 10.0)])
    result = compute_slippage(book, clip_usd=500.0, side="sell")
    assert result.slippage_bps > 0


# ---------------------------------------------------------------------------
# Mid price
# ---------------------------------------------------------------------------


def test_mid_price_stored_on_result():
    book = make_book(bids=[(99.0, 1.0)], asks=[(101.0, 1.0)])
    result = compute_slippage(book, clip_usd=50.0, side="buy")
    assert result.mid_price == pytest.approx(100.0)


def test_mid_price_is_none_when_one_side_empty():
    book = make_book(bids=[], asks=[(101.0, 1.0)])
    result = compute_slippage(book, clip_usd=50.0, side="buy")
    assert result.mid_price is None


# ---------------------------------------------------------------------------
# Multi-clip helper
# ---------------------------------------------------------------------------


def test_compute_slippage_multi_returns_list_ordered_by_clip():
    """compute_slippage_multi returns one result per clip, buy+sell."""
    from perp_liquidity.analyzers.slippage import compute_slippage_multi

    book = make_book(bids=[(99.0, 100.0)], asks=[(101.0, 100.0)])
    results = compute_slippage_multi(book, clip_usds=[1_000, 10_000], sides=["buy", "sell"])

    assert len(results) == 4  # 2 clips * 2 sides
    clips = [r.clip_usd for r in results]
    sides = [r.side for r in results]
    assert set(clips) == {1_000, 10_000}
    assert set(sides) == {"buy", "sell"}


def test_compute_slippage_multi_default_clips_and_sides():
    """Default clips=[1000,10000,100000,500000], sides=['buy','sell']."""
    from perp_liquidity.analyzers.slippage import compute_slippage_multi, DEFAULT_CLIPS

    book = make_book(bids=[(99.0, 10_000.0)], asks=[(101.0, 10_000.0)])
    results = compute_slippage_multi(book)

    assert len(results) == len(DEFAULT_CLIPS) * 2
