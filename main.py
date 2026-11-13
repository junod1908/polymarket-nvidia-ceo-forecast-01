"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 240 minutes ahead.

Entry for make.fun competition polymarket-nvidia-ceo-forecast-01
(program/v1: Python 3.13, standard library only, no network).

Model
-----
Each market's price is a step process: it sits still for many minutes, ticks
by small amounts, and occasionally jumps.  The house baseline assumes every
market moves like a random walk with the same sigma of 0.0025 per
sqrt-minute.  These markets are in fact far quieter than that in their body
and differ a lot from one another, so this entry estimates each market's own
volatility from its own price history in the case file:

* take the market's minute-to-minute price changes over its recent history
  (the last 1,441 prices, i.e. the last 24 hours of changes);
* sigma = the 90th percentile of |change| divided by 1.644854 (the 90th
  percentile of |N(0, 1)|).  A high quantile of the absolute changes is a
  robust scale estimate here: the median change is 0 (most minutes do not
  move) and the mean square is dominated by a few wild early prints, so
  neither the median nor the RMS is usable;
* floor sigma at 0.0004 so a market that has barely moved still gets a
  small, honest spread;
* the median forecast is the last price (persistence: these prices show no
  exploitable short-horizon drift -- a shrunk recent-trend term was scored
  on the practice and made the loss clearly worse), and the other quantiles
  fan out around it,

      q_tau(h) = clamp(x + z_tau * sigma * h**0.44, 0, 1),

  with z the standard normal quantiles of the five scored levels.  The
  exponent is 0.44, a little under the random walk's 0.5: on the practice
  cases the realised spreads grow slightly slower than sqrt(h) (prices
  that jump tend to come part of the way back), and 0.44 scored best of
  the exponents tried, robustly across the neighbouring values.  The two
  outer quantiles are widened by a further factor 1.2: minute changes are
  fat-tailed, so the 5% and 95% levels of the h-minute displacement sit
  further out than a normal body fitted at the 90th percentile implies,
  and the widening paid for itself at every horizon on the practice.

The program reads every *.in file in $MAKEFUN_INPUT_DIR (else /input) in
name order and writes one <case>.out per case to $MAKEFUN_OUTPUT_DIR (else
/output).  It is deterministic: fixed arithmetic, sorted file order, no
randomness, no clock.
"""

import json
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in row order.
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level, to 6 decimals.
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Minutes of price history used for the volatility estimate (the last day).
HISTORY_MINUTES = 1441
# The 90th percentile of |N(0, 1)|: turns a |change| quantile into a sigma.
ABS_Q90 = 1.644854
# No market is treated as perfectly still.
SIGMA_FLOOR = 0.0004
# Spreads grow like h**SPREAD_EXPONENT (a touch slower than a random walk).
SPREAD_EXPONENT = 0.44
# Extra width on the 5% and 95% quantiles for the fat tails of the moves.
OUTER_FACTOR = 1.2


def quantile(sorted_vals, t):
    """The t-quantile of an ascending list, linearly interpolated."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = t * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def market_sigma(prices):
    """Robust per-market volatility: a high quantile of |minute changes|."""
    seg = prices[-HISTORY_MINUTES:]
    diffs = [abs(seg[i + 1] - seg[i]) for i in range(len(seg) - 1)]
    if not diffs:
        return SIGMA_FLOOR
    diffs.sort()
    return max(quantile(diffs, 0.90) / ABS_Q90, SIGMA_FLOOR)


def price6(v):
    """Clamp to [0, 1] and round to 6 decimals (never -0.0)."""
    v = round(min(1.0, max(0.0, v)), 6)
    return v + 0.0 if v != 0 else 0.0


def forecast_market(prices, horizon):
    """240 rows of five quantiles around the market's last price."""
    if prices:
        x = prices[-1]
        sigma = market_sigma(prices)
    else:
        x, sigma = 0.5, SIGMA_FLOOR
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * h ** SPREAD_EXPONENT
        rows.append([
            price6(x + Z[0] * OUTER_FACTOR * spread),
            price6(x + Z[1] * spread),
            price6(x),
            price6(x + Z[3] * spread),
            price6(x + Z[4] * OUTER_FACTOR * spread),
        ])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        if list(case["quantiles"]) != TAUS:
            raise ValueError(f"{name} asks for quantiles {case['quantiles']}, not {TAUS}")
        horizon = int(case["horizon_minutes"])
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
            for m in case["markets"]
        ]
        text = json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n"
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
