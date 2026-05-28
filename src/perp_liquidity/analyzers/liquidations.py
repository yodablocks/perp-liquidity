"""Liquidations analyzer: aggregate events from WS_TAIL venues."""

from __future__ import annotations

from dataclasses import dataclass, field

from perp_liquidity.fetchers.base import Liquidation


@dataclass
class LiquidationSummary:
    count: int
    total_usd: float
    long_usd: float
    short_usd: float
    long_count: int
    short_count: int
    largest_usd: float
    largest_event: Liquidation | None
    by_venue: dict[str, "LiquidationSummary"] = field(default_factory=dict)


def summarize_liquidations(liqs: list[Liquidation]) -> LiquidationSummary:
    """Aggregate liquidation events into a summary.

    by_venue contains a nested LiquidationSummary per venue (without
    further nesting -- by_venue is empty on nested summaries).
    """
    if not liqs:
        return LiquidationSummary(
            count=0,
            total_usd=0.0,
            long_usd=0.0,
            short_usd=0.0,
            long_count=0,
            short_count=0,
            largest_usd=0.0,
            largest_event=None,
            by_venue={},
        )

    total_usd = sum(l.qty_usd for l in liqs)
    long_liqs = [l for l in liqs if l.side == "long"]
    short_liqs = [l for l in liqs if l.side == "short"]
    largest = max(liqs, key=lambda l: l.qty_usd)

    # Per-venue breakdown (no further nesting)
    venues: dict[str, list[Liquidation]] = {}
    for l in liqs:
        venues.setdefault(l.venue, []).append(l)

    by_venue = {
        venue: LiquidationSummary(
            count=len(vliqs),
            total_usd=sum(l.qty_usd for l in vliqs),
            long_usd=sum(l.qty_usd for l in vliqs if l.side == "long"),
            short_usd=sum(l.qty_usd for l in vliqs if l.side == "short"),
            long_count=sum(1 for l in vliqs if l.side == "long"),
            short_count=sum(1 for l in vliqs if l.side == "short"),
            largest_usd=max(l.qty_usd for l in vliqs),
            largest_event=max(vliqs, key=lambda l: l.qty_usd),
        )
        for venue, vliqs in venues.items()
    }

    return LiquidationSummary(
        count=len(liqs),
        total_usd=total_usd,
        long_usd=sum(l.qty_usd for l in long_liqs),
        short_usd=sum(l.qty_usd for l in short_liqs),
        long_count=len(long_liqs),
        short_count=len(short_liqs),
        largest_usd=largest.qty_usd,
        largest_event=largest,
        by_venue=by_venue,
    )
