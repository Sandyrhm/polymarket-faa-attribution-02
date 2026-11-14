# Polymarket FAA attribution quantile forecaster

Entry for make.fun bounty `polymarket-faa-attribution-02`: forecast the
Yes price of every market of Polymarket's FAA outage attribution family
at every minute of the 4 hours after a checkpoint, as five quantiles
(5%, 25%, 50%, 75%, 95%).

## Method

`main.py` (Python 3.13, stdlib only, deterministic, offline) forecasts
each market independently:

- **Level:** the last price at the checkpoint.
- **Median path:** the level, flat. Two drift terms exist but are off for
  this final: the cross-market sum drift (the median pulled toward a
  cross-market sum of 1) only applies when a case holds >= 5 markets,
  i.e. a single-outcome family like the practice's NVIDIA event — the
  final's 4 FAA markets are four independent Yes/No questions, so no
  sum drift applies there. The proportional decay toward 0 was tested
  and rejected (the markets sit flat, they do not decay).
- **Volatility:** trimmed, exponentially-weighted RMS of the 1-minute
  price changes over the last 720 minutes (half-life 60 min) — the
  largest 5% of squared changes are dropped first, then changes are
  winsorized at ±0.05 and weighted by recency. Trimming matters: when a
  market is first listed its price sits near 0.5, then settles near
  0.01, and that one launch collapse would otherwise dominate any RMS.
  Floored at 0.0003, capped at 0.02, multiplied by 0.6.
- **Spread:** `sigma * sqrt(h)` for horizon `h` minutes.
- **Quantiles:** asymmetric — realised moves skew down (the book usually
  carries an overround, and fresh listings collapse from their opening
  price), so the downside bands are wider:
  z = −3.0 / −0.9 / 0 / +0.4 / +2.0. Flipping the tails for low-priced
  markets and widening the upside tail were both tested and lost on
  every validation set, and a jump stress test showed wider tails buy
  almost nothing against a real jump (a jump to 0.50 or 1.00 dwarfs any
  reasonable spread), so the entry stays sharp.

## Local validation

No local grader exists for this competition (program/v1, server-graded),
so the practice cases were scored locally against answers re-fetched from
Polymarket's public price history (`GET /clob.polymarket.com/prices-history`,
fidelity=1), gridded exactly as the rules describe. That scorer
reproduces the house baseline's official practice score (0.363589)
exactly, so it is the same measure the server uses on the practice.

| case | this entry | house baseline |
|------|-----------|----------------|
| p00 | 0.411 | 0.536 |
| p01 | 0.304 | 0.380 |
| p02 | 0.363 | 0.408 |
| p03 | 0.229 | 0.369 |
| p04 | 0.199 | 0.299 |
| p05 | 0.105 | 0.238 |
| **mean** | **0.258** | **0.364** |

The practice is another event's instance, so the entry was also
validated on pseudo-cases cut from the FAA markets' own live history
(checkpoints 02:00–05:00Z Nov 14, the flat regime the final's
checkpoints live in): **0.134** there, the best of all tested variants
(next best 0.140). Ideas tested and rejected on one or both sets:
flipped tails for low prices, widened upside tails, proportional decay
toward 0, a relative volatility floor, longer lookback windows and
half-lives, heavier trimming, median-of-last-5 as the level, and wider
sigma multipliers.

## Run

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

One case takes ~0.1 s. Output is `/output/<id>.out` per case:
`{"forecasts": [{"id": ..., "q": [[...5 quantiles...] x 240]}]}`.
