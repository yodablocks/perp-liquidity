"""Slippage analyzer: compute effective fill cost vs mid price."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from perp_liquidity.fetchers.base import OrderBook

DEFAULT_CLIPS: list[float] = [1_000, 10_000, 100_000, 500_000]


@dataclass(frozen=True)
class SlippageResult:
    venue: str
    token: str
    side: Literal["buy", "sell"]
    clip_usd: float
    filled_usd: float
    effective_price: float  # weighted-average fill price (0.0 if filled_usd == 0)
    mid_price: float | None
    slippage_bps: float  # always >= 0; larger = worse
    partial: bool  # True when book exhausted before clip filled


def compute_slippage(
    book: OrderBook,
    clip_usd: float,
    side: Literal["buy", "sell"],
) -> SlippageResult:
    """Walk the order book and compute slippage for a given clip size.

    Args:
        book: OrderBook snapshot (bids descending, asks ascending).
        clip_usd: Target fill size in USD.
        side: 'buy' walks asks; 'sell' walks bids.

    Returns:
        SlippageResult with effective_price, slippage_bps, and whether the
        fill was partial (book depth insufficient).
    """
    levels = book.asks if side == "buy" else book.bids
    mid = book.mid_price

    if not levels or clip_usd <= 0:
        return SlippageResult(
            venue=book.venue,
            token=book.token,
            side=side,
            clip_usd=clip_usd,
            filled_usd=0.0,
            effective_price=0.0,
            mid_price=mid,
            slippage_bps=0.0,
            partial=True,
        )

    remaining_usd = clip_usd
    total_usd_spent = 0.0
    total_base_filled = 0.0

    for level in levels:
        if remaining_usd <= 0:
            break
        available_usd = level.price * level.qty
        consumed_usd = min(remaining_usd, available_usd)
        consumed_base = consumed_usd / level.price
        total_usd_spent += consumed_usd
        total_base_filled += consumed_base
        remaining_usd -= consumed_usd

    partial = remaining_usd > 1e-9  # floating-point tolerance
    filled_usd = total_usd_spent

    if total_base_filled == 0:
        return SlippageResult(
            venue=book.venue,
            token=book.token,
            side=side,
            clip_usd=clip_usd,
            filled_usd=0.0,
            effective_price=0.0,
            mid_price=mid,
            slippage_bps=0.0,
            partial=True,
        )

    effective_price = total_usd_spent / total_base_filled

    if mid is None or mid == 0:
        slippage_bps = 0.0
    elif side == "buy":
        slippage_bps = (effective_price - mid) / mid * 10_000
    else:
        slippage_bps = (mid - effective_price) / mid * 10_000

    return SlippageResult(
        venue=book.venue,
        token=book.token,
        side=side,
        clip_usd=clip_usd,
        filled_usd=filled_usd,
        effective_price=effective_price,
        mid_price=mid,
        slippage_bps=max(0.0, slippage_bps),
        partial=partial,
    )


def compute_slippage_multi(
    book: OrderBook,
    clip_usds: list[float] | None = None,
    sides: list[Literal["buy", "sell"]] | None = None,
) -> list[SlippageResult]:
    """Run compute_slippage for every (clip, side) combination.

    Defaults to DEFAULT_CLIPS and both sides.
    """
    if clip_usds is None:
        clip_usds = DEFAULT_CLIPS
    if sides is None:
        sides = ["buy", "sell"]

    return [
        compute_slippage(book, clip_usd=clip, side=side)
        for clip in clip_usds
        for side in sides
    ]
