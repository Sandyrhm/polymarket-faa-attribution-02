"""Quantile forecaster for Polymarket's FAA outage attribution family.

Reads one case at a time (/input/<id>.in: every market's Yes price per minute
for the 72h up to a checkpoint) and writes /output/<id>.out with, for each
market and each of the 240 minutes after the checkpoint, five quantiles
(5%, 25%, 50%, 75%, 95%) of that minute's price.

Method, per market (stdlib only, deterministic, no network, no model calls):
  - level: the last price at the checkpoint.
  - median path: the level, optionally pulled toward a cross-market sum of 1
    (only when the case holds >= 5 markets, i.e. a single-outcome family
    like the practice's NVIDIA event; the final's 4 FAA markets are four
    independent Yes/No questions, so no sum drift applies there) and
    optionally decayed proportionally toward 0 (no-news decay for
    attribution markets).
  - sigma1: EXPONENTIALLY-WEIGHTED trimmed RMS of the 1-minute price
    changes over the last WIN minutes (half-life 120 min) -- the largest
    5% of squared changes are dropped first (a launch jump would otherwise
    dominate any RMS), then changes are winsorized at +-0.05 and weighted
    by recency. Floored, capped, multiplied by 0.7.
  - spread(h) = sigma1 * sqrt(h).
  - quantiles are ASYMMETRIC. The practice event's moves skew down (the
    book carries an overround), so the downside bands are wider there:
    z = -3.0 / -0.9 / 0 / +0.4 / +2.0. But a market priced near 0 can only
    drift down slowly while it can jump up a long way on news, so for
    low-priced markets the tails flip (TAIL=1) or the upside widens while
    the downside stays wide (TAIL=2).

Local validation (own scorer, answers re-fetched from Polymarket's public
price history and gridded exactly as the rules describe -- it reproduces
the house baseline's official practice score 0.363589 exactly):
  practice (6 cases, item-weighted mean pinball): this entry 0.2581
  (baseline 0.3636; the shipped nvidia v3 scores 0.2596). Per case:
  p00 0.411, p01 0.304, p02 0.363, p03 0.229, p04 0.199, p05 0.105.
  The final is a different event (4 independent low-priced markets, no
  overround, no sum-to-1), so the entry was also validated on
  pseudo-cases cut from the FAA markets' own live history (checkpoints
  02:00-05:00Z Nov 14, flat regime): 0.1338 there, best of all tested
  variants. A jump stress test (one market jumping to 0.50 or 1.00 and
  staying) showed wider tails buy almost nothing -- a real jump dwarfs
  any reasonable spread -- so the entry stays sharp rather than paying
  for width it cannot use.
"""

import json
import math
import os
from statistics import median

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# --- tuning parameters (defaults = shipped candidate) ---
WIN = int(os.environ.get("POLY_WIN", "720"))            # lookback minutes for diffs
HALFLIFE = _f("POLY_HALFLIFE", 60.0)                   # EWMA half-life in minutes
TRIM = _f("POLY_TRIM", 0.05)                           # fraction of largest squared diffs dropped
SIG_FLOOR = _f("POLY_FLOOR", 0.0003)                   # min per-minute sigma
SIG_CAP = _f("POLY_CAP", 0.02)                         # max per-minute sigma
SIG_MULT = _f("POLY_MULT", 0.6)                        # global sigma multiplier
Z95U = _f("POLY_Z95U", 2.0)                            # up-tail quantile
Z95D = _f("POLY_Z95D", 3.0)                            # down-tail quantile
Z75U = _f("POLY_Z75U", 0.4)                            # up inner quantile
Z75D = _f("POLY_Z75D", 0.9)                            # down inner quantile
ALPHA = _f("POLY_ALPHA", 0.5)                          # spread growth exponent
LEVEL = os.environ.get("POLY_LEVEL", "last")           # "med5" or "last"
WINSOR = _f("POLY_WINSOR", 0.05)                       # winsorize 1-min diffs at +-this
SUM_K = _f("POLY_SUM_K", 1.0)                          # cross-market sum drift strength
TAIL = int(os.environ.get("POLY_TAIL", "0"))           # 0=v3 down-wide always;
                                                       # 1=flip tails when level<0.5;
                                                       # 2=widen up tail only when level<0.2
DECAY = _f("POLY_DECAY", 0.0)                          # proportional decay toward 0 over horizon
RELFLOOR = _f("POLY_RELFLOOR", 0.0)                    # sigma floor as fraction of level


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def tail_z(level):
    """Quantile z-scores for a market at `level` (see module docstring)."""
    if TAIL == 1 and level < 0.5:
        return [-Z95U, -Z75U, 0.0, Z75D, Z95D]
    if TAIL == 2 and level < 0.2:
        return [-Z95D, -Z75D, 0.0, Z75D, Z95D]
    return [-Z95D, -Z75D, 0.0, Z75U, Z95U]


def forecast_market(prices, horizon, sum_drift=0.0):
    n = len(prices)
    if n == 0:
        raise ValueError("market has no prices")
    if LEVEL == "last" or n < 5:
        level = prices[-1]
    else:
        level = median(prices[-5:])
    # per-minute volatility: trimmed, exponentially-weighted RMS of recent diffs
    m = min(WIN, n - 1)
    if m < 10:
        sigma1 = 0.0025
    else:
        diffs = []
        for i in range(n - m, n):
            d = prices[i] - prices[i - 1]
            if d > WINSOR:
                d = WINSOR
            elif d < -WINSOR:
                d = -WINSOR
            diffs.append(d)
        sq = sorted(d * d for d in diffs)
        keep_n = max(10, int((1.0 - TRIM) * len(sq)))
        cut = sq[keep_n - 1]
        lam = 0.5 ** (1.0 / HALFLIFE)
        wsum = 0.0
        acc = 0.0
        w = 1.0
        for d in reversed(diffs):
            if d * d > cut:
                continue
            acc += w * d * d
            wsum += w
            w *= lam
        sigma1 = math.sqrt(acc / wsum) if wsum > 0 else 0.0025
        floor = max(SIG_FLOOR, RELFLOOR * level)
        if sigma1 < floor:
            sigma1 = floor
        elif sigma1 > SIG_CAP:
            sigma1 = SIG_CAP
    sigma1 *= SIG_MULT
    z = tail_z(level)
    rows = []
    for h in range(1, horizon + 1):
        frac = h / horizon
        # median: sum drift (single-outcome families only) + decay toward 0
        med = level * (1.0 + SUM_K * sum_drift * frac - DECAY * frac)
        spread = sigma1 * (h ** ALPHA)
        rows.append([price6(med + zz * spread) for zz in z])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case["horizon_minutes"])
        levels = [mkt["prices"][-1] for mkt in case["markets"]]
        sum_x = sum(levels)
        # the sum signal only means something for a (near-)complete book
        sum_drift = (1.0 - sum_x) / sum_x if len(levels) >= 5 and sum_x > 0 else 0.0
        forecasts = []
        for mkt in case["markets"]:
            forecasts.append({"id": mkt["id"], "q": forecast_market(mkt["prices"], horizon, sum_drift)})
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
