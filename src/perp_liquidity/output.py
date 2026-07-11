"""Text and CSV output formatters for AnalysisReport."""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perp_liquidity.cli import AnalysisReport
    from perp_liquidity.analyzers.slippage import SlippageResult


def format_report(report: "AnalysisReport") -> str:
    lines: list[str] = []
    ts = report.fetched_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"Perp Liquidity Snapshot — {report.token}  [{ts}]")
    lines.append("=" * 72)

    # --- Funding ---
    lines.append("\nFunding Rate (APR, annualized)  [rank 1 = most negative]")
    lines.append(f"  {'Venue':<14} {'Period':>8} {'Rate/period':>14} {'APR':>12}")
    lines.append("  " + "-" * 52)
    for row in report.funding:
        fr = row.funding_rate
        lines.append(
            f"  {fr.venue:<14} {fr.period_hours:>6.0f}h"
            f"  {fr.rate_per_period:>12.6f}"
            f"  {fr.apr_annualized * 100:>10.2f}%"
            f"  #{row.rank}"
        )
    if report.funding:
        spread = report.funding[0].spread_bps
        lines.append(f"\n  Cross-venue spread: {spread:.1f} bps APR")

    # --- Open Interest ---
    lines.append("\nOpen Interest (USD)")
    lines.append(f"  {'Venue':<14} {'OI (USD)':>16} {'Share':>8} {'Rank':>6}")
    lines.append("  " + "-" * 48)
    for oi_row in report.oi:
        oi = oi_row.open_interest
        lines.append(
            f"  {oi.venue:<14} ${oi.oi_usd:>14,.0f}"
            f"  {oi_row.market_share_pct:>6.1f}%"
            f"  #{oi_row.rank}"
        )
    if report.oi:
        total = report.oi[0].total_usd
        lines.append(f"\n  Total tracked OI: ${total:,.0f}")

    # --- Slippage ---
    if report.slippage:
        clip_usds = sorted({r.clip_usd for r in report.slippage})
        lines.append("\nSlippage (bps vs mid)  [* = partial fill]")
        header_clips = "  ".join(f"${int(c):>7,}" for c in clip_usds)
        lines.append(f"  {'Venue':<14}  {'Side':<5}  {header_clips}")
        lines.append("  " + "-" * (14 + 2 + 5 + 2 + 12 * len(clip_usds)))

        # group by (venue, side)

        keyed: dict[tuple[str, str], dict[float, "SlippageResult"]] = {}
        for r in report.slippage:
            key = (r.venue, r.side)
            keyed.setdefault(key, {})[r.clip_usd] = r

        venues_seen = []
        for (venue, side), by_clip in sorted(keyed.items()):
            if venue not in venues_seen:
                venues_seen.append(venue)
            clip_cols = []
            for c in clip_usds:
                cell = by_clip.get(c)
                if cell is None:
                    clip_cols.append(f"{'n/a':>9}")
                elif cell.partial:
                    val = f"{cell.slippage_bps:.3f}" if cell.slippage_bps < 0.01 else f"{cell.slippage_bps:.2f}"
                    clip_cols.append(f"{val + '*':>9}")
                else:
                    val = f"{cell.slippage_bps:.3f}" if cell.slippage_bps < 0.01 else f"{cell.slippage_bps:.2f}"
                    clip_cols.append(f"{val:>9}")
            lines.append(f"  {venue:<14}  {side:<5}  {'  '.join(clip_cols)}")

    # --- Failures / not listed ---
    if report.not_listed:
        lines.append(f"\nToken not listed: {', '.join(report.not_listed)}")
    if report.failures:
        lines.append("\nVenue errors:")
        for venue, msg in report.failures:
            lines.append(f"  {venue}: {msg}")

    return "\n".join(lines)


def write_csv(report: "AnalysisReport", path: str) -> None:
    """Write slippage, funding, and OI results to separate CSV files under path/."""
    os.makedirs(path, exist_ok=True)
    token = report.token
    ts = report.fetched_at.strftime("%Y%m%d_%H%M%S")

    # Slippage CSV
    if report.slippage:
        slippage_path = os.path.join(path, f"{token}_slippage_{ts}.csv")
        with open(slippage_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["venue", "token", "side", "clip_usd", "filled_usd",
                        "effective_price", "mid_price", "slippage_bps", "partial"])
            for r in report.slippage:
                w.writerow([r.venue, r.token, r.side, r.clip_usd, r.filled_usd,
                             r.effective_price, r.mid_price, r.slippage_bps, r.partial])

    # Funding CSV
    if report.funding:
        funding_path = os.path.join(path, f"{token}_funding_{ts}.csv")
        with open(funding_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "venue", "token", "rate_per_period", "period_hours",
                        "apr_annualized", "spread_bps", "next_funding_at"])
            for row in report.funding:
                fr = row.funding_rate
                w.writerow([row.rank, fr.venue, fr.token, fr.rate_per_period,
                             fr.period_hours, fr.apr_annualized, row.spread_bps,
                             fr.next_funding_at])

    # OI CSV
    if report.oi:
        oi_path = os.path.join(path, f"{token}_oi_{ts}.csv")
        with open(oi_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "venue", "token", "oi_base", "oi_usd",
                        "mark_price", "market_share_pct", "total_usd"])
            for oi_row in report.oi:
                oi = oi_row.open_interest
                w.writerow([oi_row.rank, oi.venue, oi.token, oi.oi_base, oi.oi_usd,
                             oi.mark_price, oi_row.market_share_pct, oi_row.total_usd])
