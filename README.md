# Mean Reversion Pairs Trading — EWC / EWA

A statistical-arbitrage study that trades the spread between two closely related country ETFs: **EWC** (iShares MSCI Canada) and **EWA** (iShares MSCI Australia). The idea is market-neutral: instead of betting on whether either ETF goes up or down, the strategy bets that the *relationship* between them mean-reverts after it stretches too far in either direction.

The project covers the full path from raw price data to a backtested strategy:

1. Pull daily prices for both ETFs.
2. Confirm they move together over the long run (cointegration tests).
3. Estimate a time-varying hedge ratio with a Kalman filter.
4. Build the spread, measure how fast it reverts (half-life), and turn it into a rolling z-score.
5. Trade the z-score with simple entry/exit thresholds and check the equity curve.

There are two parallel implementations of the strategy logic: a research notebook with a hand-written trade simulation, and a `backtrader` backtest. There is also a small Interactive Brokers connectivity test for live/paper data.

## Overview

The research lives in `ewc_ewa_linear_test.ipynb`. It downloads the price history, runs the statistical tests, fits the Kalman filter, and runs a self-contained trade loop that tracks cash, long/short holdings, and equity over time. It also produces the diagnostic plots stored in `images/`.

The `backtrader` version (`backtest_ewc_ewa_pair_trade.py`) reuses the same maths but runs it through a proper event-driven backtest engine. The Kalman hedge ratio and the moving z-score are pre-computed and fed into the engine as extra data lines, so the strategy class only has to react to the z-score on each bar.

## How it works

**Data.** Daily `Open` prices for EWC and EWA are pulled from Yahoo Finance with `yfinance` (default window `2019-01-01` to `2022-01-01`) and saved to `Stock_data/EWC.csv` and `Stock_data/EWA.csv`.

**Cointegration.** The notebook uses the Johansen test (`coint_johansen`), the Engle-Granger test (`coint`), and the Augmented Dickey-Fuller test (`adfuller`) to check that EWC and EWA share a stable long-run relationship and that the resulting spread is stationary.

**Dynamic hedge ratio.** Rather than a single static OLS hedge ratio, a Kalman filter (`pykalman`) estimates a slope and intercept that drift over time. EWA is treated as the observation, `[EWC, 1]` as the observation matrix, and the state `[slope, intercept]` follows a random walk. The process noise is controlled by `delta = 1e-5`.

```python
delta = 1e-5
trans_cov = delta / (1 - delta) * np.eye(2)
obs_mat = np.vstack([ewc["Open"], np.ones(ewc["Open"].shape)]).T[:, np.newaxis]

kf = KalmanFilter(
    n_dim_obs=1,
    n_dim_state=2,
    initial_state_mean=np.zeros(2),
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_mat,
    observation_covariance=1.0,
    transition_covariance=trans_cov,
)

state_means, state_covs = kf.filter(ewa["Open"].values)
slope = state_means[:, 0]
intercept = state_means[:, 1]
```

**Spread and z-score.** The spread is `EWA["Open"] - slope * EWC["Open"]`. The half-life of mean reversion is estimated by regressing the spread's first difference on its lagged level (an AR(1)-style OLS fit) and taking `-ln(2) / coefficient`. The notebook then uses a lookback window derived from the half-life (`round(halflife) * 6`) to compute a moving standard deviation and a moving z-score:

```
z = (spread - intercept) / moving_std
```

**Trading rules.** When the z-score drops below `-entryScore` the spread is "too low", so the strategy goes long the spread (long EWA, short EWC). When it rises above `+entryScore` it goes short the spread (short EWA, long EWC). Positions are closed when the z-score returns to roughly zero (within a small exit band). In the notebook the entry threshold is `1.7` and the simulation starts with `2000` in cash; in the `backtrader` script the thresholds are `entryLong = -1.5`, `entryShort = 1.8`, and an exit band of `0.2`, starting from `20000`.

## Features

- Daily ETF data download and CSV caching via `yfinance`.
- Johansen, Engle-Granger, and ADF tests for cointegration and stationarity.
- Kalman-filter dynamic linear regression for a time-varying hedge ratio.
- Half-life estimation to size the z-score lookback window.
- Rolling moving standard deviation and moving z-score.
- A hand-written trade simulation in the notebook with cash/holdings/equity tracking and an equity curve.
- A `backtrader` backtest with a custom pandas data feed, an IBKR-style fixed commission model, hedge-ratio-based position sizing, and trade/PnL logging.
- An Interactive Brokers connectivity test for streaming live data.
- Pre-rendered diagnostic charts in `images/`.

## Tech Stack

- **Language:** Python 3
- **Notebook:** Jupyter
- **Data:** `yfinance` (Yahoo Finance)
- **Numerics / data:** NumPy, pandas
- **Statistics:** statsmodels (`coint_johansen`, `coint`, `adfuller`, `OLS`)
- **Filtering:** `pykalman`
- **Backtesting:** `backtrader`
- **Brokerage / live data:** Interactive Brokers TWS API via `IbPy2` and `backtrader`'s `IBStore`
- **Plotting:** matplotlib

## Project Structure

```
.
├── ewc_ewa_linear_test.ipynb        # Main research notebook: data, tests, Kalman filter, trade sim
├── backtest_ewc_ewa_pair_trade.py   # backtrader implementation of the pair-trading strategy
├── back_testing.py                  # Minimal backtrader template strategy
├── paper_trade_test.py              # Interactive Brokers live/paper data connectivity test
├── requirements                     # Environment dependency list (pip freeze)
└── images/                          # Diagnostic plots and a notebook to generate them
    ├── Johansen Eigen value demo.png
    ├── Linear Regession Hedge Ratio Example.png
    ├── Linear Regression DEMO.png
    ├── Linear Regression vs Kalman Filter.png
    ├── Standard deviation demo.png
    ├── backtrader result.png
    ├── staionary example.png
    ├── stationary plot example.png
    ├── medium article image.png
    └── image_generation.ipynb
```

The `Stock_data/` directory referenced by the code is not committed. It is created when you run the first notebook cell, which downloads the data and writes `Stock_data/EWC.csv` and `Stock_data/EWA.csv`. The `backtrader` scripts read from those same files.

## Installation

```bash
git clone https://github.com/wittyswayam/quant-mean-reversion-model.git
cd quant-mean-reversion-model

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

The committed `requirements` file is a full environment freeze and includes packages this project does not import (for example torch/torchvision/torchaudio). The actual dependencies are:

```text
yfinance
pandas
numpy
matplotlib
statsmodels
pykalman
backtrader
IbPy2          # only needed for paper_trade_test.py
jupyter        # only needed to run the notebook
```

Install them directly:

```bash
pip install yfinance pandas numpy matplotlib statsmodels pykalman backtrader IbPy2 jupyter
```

## Running the project

### 1. Get the data

From the project root, create the data folder and run the first notebook cell (or the equivalent snippet) to download prices:

```bash
mkdir -p Stock_data
```

```python
import yfinance as yf

ewc = yf.Ticker("EWC").history(start="2019-01-01", end="2022-01-01")
ewa = yf.Ticker("EWA").history(start="2019-01-01", end="2022-01-01")

ewc.to_csv("Stock_data/EWC.csv")
ewa.to_csv("Stock_data/EWA.csv")
```

### 2. Research notebook

```bash
jupyter notebook ewc_ewa_linear_test.ipynb
```

Run the cells top to bottom: download/load data, plot prices, run the cointegration and ADF tests, fit the Kalman filter, build the spread and z-score, then run the trade simulation and plot the equity curve. The final cell prints the number of trades and flags any losing ("bad") trades.

### 3. backtrader backtest

```bash
python backtest_ewc_ewa_pair_trade.py
```

This loads `Stock_data/EWC.csv` and `Stock_data/EWA.csv`, recomputes the Kalman hedge ratio and moving z-score, wraps them in a custom pandas data feed, and runs the `bollinger_pair_trade` strategy through `backtrader`. The first ~20 bars are skipped (the moving standard deviation is undefined until the lookback window fills, so data is filtered to dates after `2019-03-01`). It prints the starting and final portfolio value and the total return. Uncomment the `cerebro.plot(...)` lines to see the chart.

The `back_testing.py` file is a minimal `backtrader` template strategy (buy after consecutive down closes, exit after a fixed number of bars). It is a scaffold for getting `backtrader` running rather than part of the pairs strategy.

### 4. Interactive Brokers connectivity test

`paper_trade_test.py` connects to a running Interactive Brokers TWS / IB Gateway instance and streams data:

```bash
python paper_trade_test.py
```

It expects TWS or IB Gateway to be listening on `localhost:7497` (the default paper-trading port) and subscribes to `USD.JPY` forex data resampled to 15-second bars, printing each tick as it arrives. This is a standalone connectivity check and does not trade the EWC/EWA pair.

## Configuration

The main knobs are set inline in the code:

| Setting | Where | Default |
|---|---|---|
| Tickers | notebook / scripts | `EWC`, `EWA` |
| Date range | notebook download cell | `2019-01-01` → `2022-01-01` |
| Kalman process noise | `delta` | `1e-5` |
| Lookback window | notebook | `round(halflife) * 6` |
| Lookback window | `backtest_ewc_ewa_pair_trade.py` | `20` |
| Entry threshold | notebook (`entryScore`) | `1.7` |
| Entry thresholds | backtrader (`entryLong`, `entryShort`) | `-1.5`, `1.8` |
| Exit band | backtrader (`exitScoreBuffer`) | `0.2` |
| Starting cash | notebook | `2000` |
| Starting cash | backtrader | `20000` |
| IB port | `paper_trade_test.py` | `7497` |

The notebook and the `backtrader` script were tuned independently, so their thresholds and lookback windows differ. Treat them as two experiments on the same idea rather than one shared configuration.

## Screenshots

Diagnostic plots are in the `images/` folder, including:

- `Linear Regression vs Kalman Filter.png` — static OLS hedge ratio versus the Kalman estimate.
- `Johansen Eigen value demo.png` — cointegration test output.
- `stationary plot example.png` / `staionary example.png` — the mean-reverting spread.
- `Standard deviation demo.png` — spread with the moving z-score overlaid.
- `backtrader result.png` — equity curve from the backtest.

```markdown
![Kalman vs OLS hedge ratio](images/Linear%20Regression%20vs%20Kalman%20Filter.png)
![Backtest result](images/backtrader%20result.png)
```

## Notes and limitations

- Results are in-sample over 2019–2022 and are meant to demonstrate the method, not to represent realistic returns.
- The notebook simulation ignores transaction costs, slippage, and borrowing/short costs. The `backtrader` script adds a fixed IBKR-style commission model but still assumes fills at the recorded price.
- The strategy assumes the EWC/EWA cointegration holds. If that relationship breaks (for example through a commodity-cycle regime shift), the spread can stop reverting and the tests should be re-run.
- The committed `requirements` file is a full environment dump; see the Installation section for the packages that are actually used.

## Future improvements

- Add stop-losses and a maximum holding period so positions don't sit in a non-reverting spread.
- Re-run cointegration on a rolling basis and pause trading when it fails.
- Extend from a single pair to a portfolio of cointegrated pairs with risk-based allocation.
- Reconcile the notebook and `backtrader` parameters and add out-of-sample / walk-forward testing.
- Wire the strategy into the Interactive Brokers connection for paper trading on the actual ETF pair.

## License

No license file is included in the repository. Without one, default copyright applies and reuse rights are unspecified. If you intend this to be open source, add a `LICENSE` file (for example MIT) to make the terms explicit.
