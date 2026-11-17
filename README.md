# fib-liquidity-sweep-backtester

**Does an intraday Fibonacci-retracement + liquidity-sweep setup (SMC / ICT style) survive realistic costs and an out-of-sample test?**

A single-file Python backtester for the setup on gold (XAUUSD), Bitcoin and Ethereum, using 1-minute data from 2019 to 2026. Every result is in **R net of costs**, measured on the initial stop. The verdict is published whether it is positive or not.

---

## The setup

**Context (higher timeframe, 15m by default)**
- Swings are N-bar fractals (N = 3), confirmed only after the N bars on the right have closed. Nothing looks into the future.
- The active impulse is the last confirmed swing high and swing low: high → low means shorts, low → high means longs.
- Minimum impulse size: 3 × ATR(14) of the higher timeframe. The impulse is invalidated if price returns beyond its origin (100%).

**Fibonacci (0 = end of impulse, 1 = origin)**
- The sweep must reach the 0.618 zone. The take-profit is at the −0.382 extension (or a fixed R with `--tp-r`).

**Trigger (lower timeframe, 5m by default)**
- A wick sweeps a liquidity level and the candle body closes back inside, in the direction of the trade.

**Execution**
- Entry at the open of the next candle. Stop beyond the wick plus a buffer.
- If stop and target are both touched in the same candle, the stop is assumed.
- Spread and commission are subtracted from every trade: gold in $ per ounce, crypto as a % of price (0.10% round trip by default).

## What it measures

- Expectancy, win rate, profit factor and maximum drawdown in net R, split **in-sample / out-of-sample**.
- MFE / MAE of every trade, and a "TP curve": the net expectancy if the target were k R, letting each trade run to its original stop.
- Monte Carlo (2,000 reshuffles) of the maximum drawdown.
- A **ladder** of filters applied one at a time (cost ≤ 10% of risk, higher-timeframe trend, key liquidity levels, sessions, higher timeframes), with how many calendar years end positive.

## Results

1-minute bid data, 2019 – 2026: five in-sample years and three out-of-sample years.

| Market · configuration | Trades | Net R per trade (in-sample) | Net R per trade (out-of-sample) | Positive years |
|---|---|---|---|---|
| Gold · base setup 15m / 5m | 1,727 | −0.19 | −0.13 | 2 / 8 |
| Gold · 1h / 15m | 503 | −0.03 | **+0.81** | 6 / 8 |
| Gold · 1h / 15m + 4h trend | 254 | −0.33 | **+1.42** | 5 / 8 |
| Bitcoin · base setup 15m / 5m | 2,617 | −0.51 | −0.59 | 0 / 8 |
| Bitcoin · best variant (4h / 1h + daily trend) | 171 | −0.15 | +0.21 | 3 / 8 |
| Ethereum · base setup 15m / 5m | 2,384 | −0.49 | −0.43 | 0 / 8 |
| Ethereum · best variant (4h / 1h + daily trend) | 176 | −0.32 | −0.07 | 2 / 8 |

## Verdict

- **The setup as usually taught loses money** in all three markets once costs are included. On 5-minute triggers, costs alone take a large share of each R.
- **On gold, moving up to 1h / 15m looks strong out of sample but loses in sample.** An edge that appears in only one of the two periods is not robust, so the variant is **reported, not adopted**.
- **On Bitcoin and Ethereum no variant holds up** in both periods.
- Useful by-product: the ladder shows that most of the apparent edge in short-timeframe versions disappears once trades whose cost exceeds 10% of the risk are removed.

## Usage

```bash
pip install pandas numpy

python fib_liquidity_sweep_backtester.py --synthetic                                  # smoke test
python fib_liquidity_sweep_backtester.py --data xau_m1.csv --split 2025-07-01          # one run
python fib_liquidity_sweep_backtester.py --data xau_m1.csv --split 2025-07-01 --ladder # filter ladder
python fib_liquidity_sweep_backtester.py --asset btc --data btc_a.csv btc_b.csv \
       --split 2024-01-01 --ladder h1                                                  # crypto, higher timeframes
```

Main options: `--asset xau|btc|eth`, `--ltf` / `--htf`, `--liquidity fractal|pdhl|asia|key`, `--trend-tf 4h`, `--max-cost-frac 0.1`, `--tp-r 2`, `--session 7-17`.

Outputs go to `resultados/`: `trades.csv` (every trade with MFE / MAE), `resumen.json`, the ladder tables and an equity chart. Market data is not included in the repository.

<sub>Code comments and option help are in Spanish; documentation is in English.</sub>

## Disclaimer

Research only. Not financial advice and not a trading signal.

---
*Author: Ivan Alonso · [GitHub](https://github.com/ivanalonso-dev) · [LinkedIn](https://www.linkedin.com/in/ivanalonsodev) · [X](https://x.com/ivanalonsodev)*
