"""CLI entry point: perp-liquidity --token BTC [--clip 1000,10000,100000,500000] [--csv PATH]"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from perp_liquidity.fetchers.base import (
    FetcherError,
    PerpDEXClient,
    TokenNotListed,
)
from perp_liquidity.analyzers.slippage import (
    SlippageResult,
    compute_slippage_multi,
    DEFAULT_CLIPS,
)
from perp_liquidity.analyzers.funding import FundingRankRow, rank_funding
from perp_liquidity.analyzers.open_interest import OIRankRow, rank_open_interest


# ---------------------------------------------------------------------------
# Default fetcher list (imported lazily to allow test injection)
# ---------------------------------------------------------------------------


def _default_fetchers() -> list[type[PerpDEXClient]]:
    from perp_liquidity.fetchers.hyperliquid import HyperliquidClient
    from perp_liquidity.fetchers.paradex import ParadexClient
    from perp_liquidity.fetchers.lighter import LighterClient
    from perp_liquidity.fetchers.aster import AsterClient
    from perp_liquidity.fetchers.extended import ExtendedClient
    from perp_liquidity.fetchers.edgex import EdgeXClient
    from perp_liquidity.fetchers.apex import ApeXClient
    from perp_liquidity.fetchers.grvt import GRVTClient

    return [
        HyperliquidClient,
        ParadexClient,
        LighterClient,
        AsterClient,
        ExtendedClient,
        EdgeXClient,
        ApeXClient,
        GRVTClient,
    ]


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnalysisReport:
    token: str
    slippage: list[SlippageResult]
    funding: list[FundingRankRow]
    oi: list[OIRankRow]
    failures: list[tuple[str, str]]  # (venue, error_message)
    not_listed: list[str]  # venues where token is not traded
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Async orchestrator
# ---------------------------------------------------------------------------


async def _fetch_one(
    cls: type[PerpDEXClient],
    token: str,
) -> tuple[str, object | None, object | None, object | None, str | None, str | None]:
    """Fetch orderbook, funding, OI for one venue.

    Returns (venue, orderbook, funding_rate, open_interest, failure_msg, not_listed_venue).
    Exactly one of (results tuple) or (failure_msg or not_listed_venue) will be set.
    """
    venue = cls.VENUE
    async with cls() as client:
        try:
            ob = await client.get_orderbook(token)
        except TokenNotListed:
            return venue, None, None, None, None, venue
        except FetcherError as e:
            return venue, None, None, None, str(e), None

        try:
            fr = await client.get_funding_rate(token)
        except (NotImplementedError, FetcherError):
            fr = None

        try:
            oi = await client.get_open_interest(token)
        except (NotImplementedError, FetcherError):
            oi = None

    return venue, ob, fr, oi, None, None


async def run_analysis(
    token: str,
    fetcher_classes: list[type[PerpDEXClient]] | None = None,
    clip_usds: list[float] | None = None,
) -> AnalysisReport:
    """Fetch data from all venues concurrently and run analyzers.

    Args:
        token: Token symbol (e.g. 'BTC'). Case-insensitive; stored uppercased.
        fetcher_classes: Override the default 8-venue list (used by tests).
        clip_usds: USD clip sizes for slippage. Defaults to DEFAULT_CLIPS.

    Returns:
        AnalysisReport with results from all venues that responded successfully.
        VenueUnavailable errors go to report.failures; TokenNotListed to report.not_listed.
        Never raises.
    """
    token = token.upper()
    if fetcher_classes is None:
        fetcher_classes = _default_fetchers()
    if clip_usds is None:
        clip_usds = list(DEFAULT_CLIPS)

    tasks = [_fetch_one(cls, token) for cls in fetcher_classes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    orderbooks = []
    funding_rates = []
    open_interests = []
    failures: list[tuple[str, str]] = []
    not_listed: list[str] = []

    for res in results:
        if isinstance(res, Exception):
            # asyncio.gather with return_exceptions=True: unexpected exception
            failures.append(("unknown", str(res)))
            continue
        venue, ob, fr, oi, fail_msg, not_listed_venue = res
        if not_listed_venue:
            not_listed.append(not_listed_venue)
        elif fail_msg:
            failures.append((venue, fail_msg))
        else:
            if ob is not None:
                orderbooks.append(ob)
            if fr is not None:
                funding_rates.append(fr)
            if oi is not None:
                open_interests.append(oi)

    slippage_results: list[SlippageResult] = []
    for ob in orderbooks:
        slippage_results.extend(compute_slippage_multi(ob, clip_usds=clip_usds))

    return AnalysisReport(
        token=token,
        slippage=slippage_results,
        funding=rank_funding(funding_rates),
        oi=rank_open_interest(open_interests),
        failures=failures,
        not_listed=not_listed,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="perp-liquidity",
        description="Compare liquidity across perpetual DEX venues.",
    )
    parser.add_argument("--token", required=True, help="Token symbol, e.g. BTC")
    parser.add_argument(
        "--clip",
        default=",".join(str(int(c)) for c in DEFAULT_CLIPS),
        help="Comma-separated USD clip sizes (default: 1000,10000,100000,500000)",
    )
    parser.add_argument("--csv", metavar="PATH", help="Write CSV output to this path")
    args = parser.parse_args(argv)

    clip_usds = [float(x.strip()) for x in args.clip.split(",")]
    report = asyncio.run(run_analysis(args.token, clip_usds=clip_usds))

    from perp_liquidity.output import format_report, write_csv

    print(format_report(report))

    if args.csv:
        write_csv(report, args.csv)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
