# perp-liquidity

Cross-venue liquidity comparison across 8 perpetual DEX venues: order book depth, slippage, funding rates, and open interest — from public APIs, via the command line.

Built as a portfolio piece for a quant trading application. No dashboard, no AI insights, no execution — just clean data from the order book.

## Venues

| Venue | Orderbook | Funding | OI | Liquidations | Funding period |
|---|---|---|---|---|---|
| Hyperliquid | ✅ | ✅ | ✅ | ✅ WS tail | 1h |
| Paradex | ✅ | ✅ | ✅ | ❌ auth required | 8h |
| Lighter | ✅ | ✅ | ✅ | ✅ WS tail | 1h |
| Aster | ✅ | ✅ | ✅ | ❌ no public feed | 8h |
| Extended | ✅ | ✅ | ✅ | ✅ WS tail | 1h |
| EdgeX | ✅ | ✅ | ✅ | ❌ private WS only | 4h |
| ApeX | ✅ | ✅ | ✅ | ❌ private WS only | 1h |
| GRVT | ✅ | ✅ | ✅ | ❌ no liquidation flag | 8h |

Documented gaps are preferred over fake coverage. Each fetcher declares what it actually supports via a `Coverage` enum.

## Install

```bash
git clone https://github.com/yodablocks/perp-liquidity
cd perp-liquidity
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
perp-liquidity --token BTC
```

Optional flags:

```bash
# Custom clip sizes (USD)
perp-liquidity --token BTC --clip 10000,100000,500000

# Write CSV output to a directory
perp-liquidity --token BTC --csv out/
```

## Sample output

```
Perp Liquidity Snapshot — BTC  [2026-05-28 14:27:18 UTC]
========================================================================

Funding Rate (APR, annualized)  [rank 1 = most negative]
  Venue            Period    Rate/period          APR
  ----------------------------------------------------
  aster               8h      0.000019        2.07%  #1
  paradex             8h      0.000100       10.95%  #2
  hyperliquid         1h      0.000013       10.95%  #3
  edgex               4h      0.000050       10.95%  #4
  apex                1h      0.000013       10.95%  #5
  extended            1h      0.000013       11.39%  #6
  lighter             1h      0.000096       84.10%  #7
  grvt                8h      0.010000     1095.00%  #8

  Cross-venue spread: 109292.9 bps APR

Open Interest (USD)
  Venue                  OI (USD)    Share   Rank
  ------------------------------------------------
  hyperliquid    $ 2,391,367,560    65.5%  #1
  aster          $   406,508,822    11.1%  #2
  edgex          $   302,853,176     8.3%  #3
  grvt           $   196,111,898     5.4%  #4
  lighter        $   143,128,524     3.9%  #5
  apex           $   110,294,228     3.0%  #6
  extended       $    93,766,168     2.6%  #7
  paradex        $     7,039,467     0.2%  #8

  Total tracked OI: $3,651,069,841

Slippage (bps vs mid)  [* = partial fill]
  Venue           Side   $  1,000  $ 10,000  $100,000  $500,000
  -----------------------------------------------------------------------
  apex            buy         0.12       0.12       0.12       0.12
  apex            sell        0.12       0.12       0.12       0.13
  aster           buy         0.34       0.36       0.36       0.50
  aster           sell       0.007      0.007      0.008       0.43
  edgex           buy         0.12       0.16       0.26       0.43
  edgex           sell        0.12       0.23       0.26       0.34
  extended        buy         0.07       0.07       0.07       0.07
  extended        sell        0.07       0.07       0.07       0.16
  grvt            buy        0.007      0.007      0.007       0.15
  grvt            sell       0.007      0.007      0.007       0.14
  hyperliquid     buy         0.20       0.70       0.83       1.12
  hyperliquid     sell        0.07       0.07       0.07       0.07
  lighter         buy        0.007       0.30       1.28       1.91
  lighter         sell       0.007       0.14       0.40       0.82
  paradex         buy         0.56       1.13       1.86       3.56
  paradex         sell        0.56       0.56       1.88      11.35
```

## Methodology

### Slippage

Walk the order book levels in execution order (asks for buy, bids for sell), accumulating USD notional until the clip size is filled. Effective price is the weighted average fill price across all consumed levels. Slippage in bps is measured against the mid price at snapshot time.

```
slippage_bps (buy)  = (effective_price - mid) / mid × 10,000
slippage_bps (sell) = (mid - effective_price) / mid × 10,000
```

A `*` suffix means the book was exhausted before the clip was filled — the reported slippage covers only the partial fill.

### Funding rate

Each venue has a different funding interval (1h, 4h, or 8h). Rates are annualized using:

```
APR = rate_per_period × (8760 / period_hours)
```

This makes cross-venue comparison meaningful regardless of settlement frequency. The raw `rate_per_period` is also shown.

> **Note on GRVT:** GRVT's funding rate of 1095% APR reflects its rate cap being hit during bootstrapping, not a typical market condition. Treat it as a signal of low liquidity rather than a tradeable opportunity.

### Open interest

All venues report OI in base asset (e.g. BTC). USD conversion uses the current mark price from the same API call.

## Architecture

```
src/perp_liquidity/
├── fetchers/
│   ├── base.py           # PerpDEXClient ABC, dataclasses, Coverage enum, error hierarchy
│   ├── hyperliquid.py
│   ├── paradex.py
│   ├── lighter.py
│   ├── aster.py
│   ├── extended.py
│   ├── edgex.py
│   ├── apex.py
│   └── grvt.py
├── analyzers/
│   ├── slippage.py       # compute_slippage, compute_slippage_multi
│   ├── funding.py        # rank_funding, detect_flips
│   └── open_interest.py  # rank_open_interest
├── cli.py                # run_analysis, main()
└── output.py             # format_report, write_csv
```

Each fetcher is an async context manager backed by `httpx.AsyncClient`. The CLI fetches all venues concurrently via `asyncio.gather` and absorbs per-venue errors — a single unavailable venue does not abort the run.

Error hierarchy: `FetcherError` → `VenueUnavailable` (API failure) | `TokenNotListed` (token not traded there).

## Known limitations

- **ApeX geo-block**: `omni.apex.exchange` resolves to `127.0.0.1` from some ISPs (confirmed on French residential connections). Workaround: add the real IP to `/etc/hosts`:
  ```bash
  # get real IP
  dig @8.8.8.8 omni.apex.exchange +short
  # add to /etc/hosts (requires sudo)
  sudo sh -c 'echo "157.185.129.119 omni.apex.exchange" >> /etc/hosts'
  # remove after use
  sudo sed -i '' '/omni\.apex\.exchange/d' /etc/hosts
  ```
- **Liquidations**: only 3 venues (Hyperliquid, Lighter, Extended) expose liquidations publicly. The other 5 require authenticated private streams. Gaps are declared explicitly via `Coverage.NOT_AVAILABLE`.
- **Lighter OI**: uses `last_trade_price` as a mark price proxy (no dedicated mark price endpoint).
- **Snapshot-only**: this tool takes one-shot snapshots. It is not a streaming feed and does not persist history.
- **Public APIs only**: no Kaiko, Amberdata, or authenticated endpoints.

## Development

```bash
# Run all tests (124 tests)
pytest

# Run a specific venue
pytest tests/test_hyperliquid.py -v

# Type-check
mypy src/

# Lint
ruff check src/ tests/
```

Tests use `httpx.MockTransport` for all REST calls — no real network required. WS venues (Hyperliquid, Lighter, Extended liquidations) are tested via async mock fixtures.
