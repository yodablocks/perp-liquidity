"""Open interest analyzer: cross-venue ranking with market share."""

from __future__ import annotations

from dataclasses import dataclass

from perp_liquidity.fetchers.base import OpenInterest


@dataclass(frozen=True)
class OIRankRow:
    open_interest: OpenInterest
    rank: int
    market_share_pct: float  # this venue's oi_usd / total_usd * 100
    total_usd: float  # sum of oi_usd across all venues in the ranking


def rank_open_interest(ois: list[OpenInterest]) -> list[OIRankRow]:
    """Rank venues by USD open interest, descending (largest = rank 1).

    Returns an empty list for empty input.
    market_share_pct values sum to 100.0 across all rows.
    """
    if not ois:
        return []

    sorted_ois = sorted(ois, key=lambda o: o.oi_usd, reverse=True)
    total_usd = sum(o.oi_usd for o in sorted_ois)

    return [
        OIRankRow(
            open_interest=oi,
            rank=i + 1,
            market_share_pct=(oi.oi_usd / total_usd * 100) if total_usd > 0 else 0.0,
            total_usd=total_usd,
        )
        for i, oi in enumerate(sorted_ois)
    ]
