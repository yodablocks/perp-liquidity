"""Funding rate analyzer: cross-venue ranking and flip detection."""

from __future__ import annotations

from dataclasses import dataclass

from perp_liquidity.fetchers.base import FundingRate


@dataclass(frozen=True)
class FundingRankRow:
    funding_rate: FundingRate
    rank: int
    spread_bps: float  # max_apr - min_apr across all venues, in bps (same on every row)


def rank_funding(rates: list[FundingRate]) -> list[FundingRankRow]:
    """Rank venues by annualized funding rate, ascending (most negative = rank 1).

    Returns an empty list for empty input.
    spread_bps is the full range (max - min) across all venues, expressed in bps.
    """
    if not rates:
        return []

    sorted_rates = sorted(rates, key=lambda r: r.apr_annualized)
    min_apr = sorted_rates[0].apr_annualized
    max_apr = sorted_rates[-1].apr_annualized
    spread_bps = (max_apr - min_apr) * 10_000

    return [
        FundingRankRow(funding_rate=fr, rank=i + 1, spread_bps=spread_bps)
        for i, fr in enumerate(sorted_rates)
    ]


def detect_flips(rates: list[FundingRate]) -> list[str]:
    """Return venue names whose funding sign differs from the majority.

    A venue is a 'flip' if its rate_per_period has the opposite sign from
    more than half the venues. Ties (50/50 split) return no flips.
    """
    if len(rates) <= 1:
        return []

    positives = [r for r in rates if r.rate_per_period >= 0]
    negatives = [r for r in rates if r.rate_per_period < 0]

    if len(positives) == len(negatives):
        return []

    if len(positives) > len(negatives):
        # Majority positive: negatives are flips
        return [r.venue for r in negatives]
    else:
        # Majority negative: positives are flips
        return [r.venue for r in positives]
