# 📉 Mean Reversion Strategies in Quantitative Finance  

A full statistical arbitrage pipeline that builds a **pairs trading strategy** between the Canadian equity ETF **EWC** and the Australian equity ETF **EWA**, using **Johansen cointegration tests**, **Kalman-filter-based dynamic linear regression**, and a **z-score mean-reversion trading engine**.[1]

The notebook demonstrates how to:
- Detect a stable long-term relationship (cointegration) between two ETFs.
- Estimate a **time-varying hedge ratio** with a **Kalman filter**.
- Construct a **mean-reverting spread**.
- Trade the spread using **z-score thresholds**, with backtested equity curves.[1]

***

## 📑 Quick Navigation

- [🎯 Executive Summary](#executive-summary)  
- [🔍 Trading Problem](#trading-problem)  
- [📈 Data & Instruments](#data--instruments)  
- [🏗️ Model Architecture](#model-architecture)  
- [⚙️ Methodology & Math](#methodology--math)  
- [🚀 Getting Started](#getting-started)  
- [📊 Backtest Results](#backtest-results)  
- [🔄 Strategy Workflow](#strategy-workflow)  
- [📈 Performance & Sensitivity](#performance--sensitivity)  
- [🔮 Extensions & Future Work](#extensions--future-work)  

***

## 🎯 Executive Summary

This project tackles **market-neutral trading** between two highly related country ETFs, **EWC (Canada)** and **EWA (Australia)**, by exploiting **relative mispricing** instead of outright price direction.[1]

Core components:

1. **Cointegration Testing**  
   - Verify that EWC and EWA share a **long-run equilibrium** using **Johansen** and **ADF** tests.[1]

2. **Dynamic Hedge Ratio via Kalman Filter**  
   - Replace static OLS regression with a **state-space model**:
     \[
     \text{EWA}_t = \beta_t \cdot \text{EWC}_t + \alpha_t + \epsilon_t
     \]
     \[
     \begin{bmatrix}\beta_t \\ \alpha_t\end{bmatrix}
       = \begin{bmatrix}\beta_{t-1} \\ \alpha_{t-1}\end{bmatrix} + \eta_t
     \]
   - \(\beta_t, \alpha_t\) evolve over time and are estimated via **Kalman filtering**.[1]

3. **Mean-Reverting Spread & Z-Score**  
   - Build a spread:
     \[
     s_t = \text{EWA}_t - \beta_t \cdot \text{EWC}_t - \alpha_t
     \]
   - Normalize via **rolling z-score** with window = **half-life** \(\approx 6\) days.[1]

4. **Pairs Trading Engine**  
   - Enter positions when z-score is extreme (e.g. \(|z| > 1.7\)).  
   - Exit when spread reverts (z-score near 0).[1]

**Key Outcome:** Over the sample 2019–2022, the simulated strategy produces a **smooth, upward-sloping equity curve** with **multiple profitable trades** and no losing trades under the specific threshold chosen in the notebook.[1]

***

## 🔍 Trading Problem

### Objective: Relative Value Between EWC and EWA

Instead of predicting whether EWC or EWA will rise or fall, the notebook focuses on their **price relationship**:

| Aspect                 | Challenge                                      | Approach                                       |
|------------------------|------------------------------------------------|-----------------------------------------------|
| Long-run relationship  | Are EWC and EWA co-moving over time?          | Johansen cointegration test                   |
| Non-stationary prices  | Price levels trend and drift                  | Work on **spread** that is stationary         |
| Time-varying link      | Relationship changes over regimes             | Kalman filter with random-walk state          |
| Entry/exit timing      | When to enter/exit trades                     | Z-score thresholds on the spread              |
| Risk control           | Avoid persistent divergence                   | Market-neutral long/short hedged positions    |[1]

### Instruments & Universe

- **EWC** – iShares MSCI Canada ETF (ticker: `EWC`)  
- **EWA** – iShares MSCI Australia ETF (ticker: `EWA`)  
- **Price Field** – Open prices (can be adapted to Close/Mid).[1]

***

## 📈 Data & Instruments

### Data Source & Period

- **Source:** Yahoo Finance via `yfinance`.[1]
- **Date Range:** `2019-01-01` to `2022-01-01`.[1]
- **Frequency:** Daily (trading days).  
- **Saved As:** `StockdataEWC.csv`, `StockdataEWA.csv` for reproducibility.[1]

### Data Preparation

1. **Download**
   ```python
   ewc = yf.Ticker("EWC").history(start="2019-01-01", end="2022-01-01")
   ewa = yf.Ticker("EWA").history(start="2019-01-01", end="2022-01-01")
   ewc.to_csv("StockdataEWC.csv")
   ewa.to_csv("StockdataEWA.csv")
   ```

2. **Load & Align**
   - Load from CSV with `pandas.read_csv`.  
   - Align indices and extract `Open` columns for both ETFs.[1]

3. **Visualization**
   - Plot EWC and EWA open prices on the same figure to show co-movement and relative level over time.[1]

***

## 🏗️ Model Architecture

The core of the notebook is a **state-space regression** with a **Kalman filter**, surrounded by tests and a trading engine.

### High-Level Architecture

```text
Yahoo Finance Price Data
        ↓
Data Loading & Cleaning (pandas)
        ↓
Cointegration Tests (Johansen, ADF)
        ↓
Dynamic Linear Regression (Kalman Filter)
        ↓
Time-varying Hedge Ratio (β_t, α_t)
        ↓
Mean-Reverting Spread s_t
        ↓
Rolling Z-Score z_t
        ↓
Trading Rules (entry/exit)
        ↓
Backtest & Equity Curve
```

### Component Overview

| Component                 | Input                      | Process                                      | Output                          |
|---------------------------|----------------------------|----------------------------------------------|---------------------------------|
| Data Loader               | Yahoo Finance, CSV        | Download, align, select columns              | EWC/EWA price series            |
| Johansen Cointegration    | Price matrix [EWC, EWA]   | Test rank of cointegration                   | Eigenvalues, test statistics    |
| ADF Test                  | Spread series             | Stationarity check                            | ADF statistic, p-value          |
| Kalman Filter Regression  | EWC, EWA prices           | State-space modelling of β_t, α_t            | Time series of β_t, α_t         |
| Spread Construction       | β_t, α_t, prices          | EWA - β_t·EWC - α_t                          | Mean-reverting spread s_t       |
| Z-Score Normalization     | Spread s_t                | Rolling mean/std (half-life window)          | z_t                             |
| Trading Engine            | z_t, prices               | Entry/exit, P&L computation                  | Trade log, equity curve         |[1]

***

## ⚙️ Methodology & Math

### 1. Cointegration: Johansen & ADF

**Johansen Test**  
Given a vector \(X_t = [\text{EWC}_t, \text{EWA}_t]^T\), the Johansen test checks for one or more cointegrating vectors \(\beta\) such that:
\[
\beta^T X_t \sim \text{stationary}
\]

The notebook:
- Builds a 2-column matrix of prices.
- Calls `coint_johansen` with appropriate lag and deterministic terms.
- Examines trace and eigenvalue statistics to confirm rank 1 cointegration.[1]

**ADF Test on Spread**  
Once a spread candidate \(s_t\) is derived, the ADF test checks:
- Null: \(s_t\) has a unit root (non-stationary).
- Alternative: \(s_t\) is stationary.

Reported ADF statistic is highly negative (approx. -654), with near-zero p-value, supporting stationarity.[1]

### 2. Static OLS (Baseline)

Initial hedge ratio estimation uses classic OLS:
\[
\text{EWA}_t = \beta \cdot \text{EWC}_t + \alpha + \epsilon_t
\]

Implemented via `statsmodels.regression.linear_model.OLS`:
- `endog` = EWA prices.
- `exog`  = EWC prices with constant term.[1]

This static model is only for comparison; trading uses the dynamic β_t from Kalman filtering.

### 3. Dynamic Linear Regression via Kalman Filter

**State Vector:**
\[
\theta_t =
\begin{bmatrix}
\beta_t \\
\alpha_t
\end{bmatrix}
\]

**Observation Equation:**
\[
y_t = H_t \theta_t + v_t
\quad\text{with}\quad
y_t = \text{EWA}_t,\quad
H_t = [\text{EWC}_t, 1]
\]

**State Equation (Random Walk):**
\[
\theta_t = \theta_{t-1} + w_t
\]

**Noise Terms:**
- \(v_t \sim \mathcal{N}(0, R)\)
- \(w_t \sim \mathcal{N}(0, Q)\)

In code:
```python
delta = 1e-5
trans_cov = (delta / (1 - delta)) * np.eye(2)   # Q
obs_mat   = np.vstack([ewc_open, np.ones_like(ewc_open)]).T[:, np.newaxis, :]

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

state_means, state_covs = kf.filter(ewa_open.values)
slope = state_means[:, 0]
intercept = state_means[:, 1]
```

### 4. Mean-Reverting Spread

Spread:
\[
s_t = \text{EWA}_t - \beta_t \cdot \text{EWC}_t - \alpha_t
\]

The notebook:
- Computes this spread series.
- Plots it with fitted mean and ±2 standard deviation bands to illustrate mean reversion.[1]

### 5. Half-Life of Mean Reversion

To derive the optimal lookback window:

1. Fit an AR(1) to the spread:
   \[
   \Delta s_t = \phi s_{t-1} + \epsilon_t
   \]

2. Half-life:
   \[
   \text{half-life} = -\frac{\ln(2)}{\phi}
   \]

Estimated half-life is ≈ 6 periods.[1]
This value is used as the **rolling window** for the z-score.

### 6. Z-Score Calculation

For each time \(t\), compute rolling mean and standard deviation over window \(h\) (half-life):

\[
\mu_t = \frac{1}{h}\sum_{i=t-h+1}^t s_i,\quad
\sigma_t = \sqrt{\frac{1}{h-1}\sum_{i=t-h+1}^t (s_i - \mu_t)^2}
\]

\[
z_t = \frac{s_t - \mu_t}{\sigma_t}
\]

This standardizes spread deviations to a common scale.[1]

### 7. Trading Rules

Let \(z_t\) be the z-score at time \(t\), and th be the entry threshold (e.g. 1.7).

**Entry:**

- **Long Spread (Long EWA, Short EWC)**  
  - Condition: \(z_t < -\text{th}\)  
  - Intuition: Spread is “too low”; expect it to rise.

- **Short Spread (Short EWA, Long EWC)**  
  - Condition: \(z_t > \text{th}\)  
  - Intuition: Spread is “too high”; expect it to fall.

**Exit:**

- When \(z_t\) crosses back toward 0 (e.g. \(|z_t| < 0.1\)), close the position.[1]

Portfolio is assumed:
- Initial capital: 2000 (units of base currency).
- Full reallocation on each entry (no leverage modelling beyond the hedge).[1]

***

## 🚀 Getting Started

### 1. Clone & Environment

```bash
git clone https://github.com/wittyswayam/quant-mean-reversion-model.git
cd quant-mean-reversion-model
```

Create environment:
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Minimal `requirements.txt` (based on notebook imports):[1]
```text
yfinance
pandas
numpy
matplotlib
statsmodels
pykalman
```

### 2. Run the Notebook

```bash
jupyter notebook ewc_ewa_linear_test.ipynb
# or:
jupyter lab ewc_ewa_linear_test.ipynb
```

Execute cells in order:

1. Data download and CSV save.
2. Imports and data loading.
3. Plot price series.
4. Cointegration tests (Johansen + ADF).
5. Kalman filter model and parameter estimation.
6. Spread and z-score calculation.
7. Trading simulation and equity curve plotting.[1]

### 3. Configuration Hooks

Inside the notebook you can easily modify:

- **Date Range:**  
  `start="YYYY-MM-DD", end="YYYY-MM-DD"` for different regimes.[1]
- **Tickers:**  
  Replace `"EWC"` and `"EWA"` with other candidate pairs.
- **Kalman Parameters:**  
  `delta` and noise covariances for smoother or more reactive β_t.
- **Z-Score Thresholds:**  
  Adjust entry level (e.g. 1.5, 2.0) to tune trade frequency vs robustness.
- **Capital & Positioning:**  
  Change initial capital, sizing, and transaction cost assumptions.

***

## 📊 Backtest Results

### Visual Diagnostics

The notebook produces multiple diagnostic plots:[1]

1. **Price Series Plot**  
   - EWC and EWA open prices over time.

2. **Dynamic Hedge Ratio Plot**  
   - Time series of **slope (β_t)** and **intercept (α_t)** from Kalman filter.  
   - Shows slowly varying relationship across the 3-year window.

3. **Spread Plot**  
   - Mean-reverting spread with estimated mean and ±2σ bands.  
   - Clear oscillation around zero confirms tradable deviations.

4. **Z-Score Plot**  
   - z-score of spread with horizontal lines at ±threshold.  
   - Visualizes entry (crossing beyond threshold) and exit (reversion).

5. **Equity Curve Plot**  
   - Cumulative equity starting from 2000 capital.  
   - Shows monotonic growth across multiple trades.

### Trade Statistics (As Implemented)

From the notebook logic:[1]

- **Sample Period:** 2019–2022  
- **Number of Trades:** 28  
- **Losing Trades:** 0 (for the chosen 1.7 threshold)  
- **Entry Threshold:** 1.7 (selected visually after experimentation)  
- **Exit Rule:** z-score back toward 0 ± small band  

These results are **in-sample** and do not include:
- Transaction costs
- Slippage
- Borrow constraints / shorting costs

They are meant as a **methodology demonstration**, not production-ready PnL.

***

## 🔄 Strategy Workflow

### End-to-End Workflow

```text
START
  │
  ├── 1. Load Historical Data (EWC & EWA)
  │      • Download via yfinance
  │      • Save / load from CSV
  │
  ├── 2. Verify Cointegration
  │      • Johansen test on [EWC, EWA]
  │      • ADF test on candidate spread
  │      • Proceed only if stationary spread detected
  │
  ├── 3. Estimate Dynamic Hedge Ratio
  │      • Build state-space model:
  │          θ_t = [β_t, α_t]'
  │      • KalmanFilter.filter(EWA_prices)
  │      • Extract β_t, α_t over time
  │
  ├── 4. Construct Spread & Z-Score
  │      • s_t = EWA_t - β_t·EWC_t - α_t
  │      • Compute half-life of s_t via AR(1)
  │      • Rolling mean, std (window = half-life)
  │      • z_t = (s_t - μ_t) / σ_t
  │
  ├── 5. Generate Trading Signals
  │      • If z_t > +th → short EWA / long EWC
  │      • If z_t < -th → long EWA / short EWC
  │      • If |z_t| < small_band → close positions
  │
  ├── 6. Backtest Simulation
  │      • Track position, cost basis, realized P&L
  │      • Update equity curve over time
  │
  └── 7. Analyze Results
         • Plot equity
         • Inspect parameter sensitivity
         • Export trades/equity to CSV if needed
```

***

## 📈 Performance & Sensitivity

### Parameter Sensitivities

1. **Delta (Process Noise in Kalman Filter)**  
   - Larger `delta` → more responsive hedge ratio, but noisier β_t.  
   - Smaller `delta` → smoother β_t, but slower adaptation to structural shifts.[1]

2. **Z-Score Threshold**  
   - Lower threshold (e.g. 1.0):
     - More trades.
     - Higher turnover, more potential false signals.
   - Higher threshold (e.g. 2.0):
     - Fewer but stronger signals.
     - Lower turnover but might miss some opportunities.

3. **Half-Life Window**  
   - Directly affects variance estimate of spread; mis-calibration produces over- or under-reactive entries.

### Practical Considerations

- **Transaction Costs**: currently ignored; real-world implementation must subtract per-trade cost and slippage.  
- **Execution Risk**: ETFs trade intraday; using open prices assumes perfect execution.  
- **Regime Shifts**: If macro relationship between Canada and Australia changes (e.g. commodity cycles), cointegration may break, requiring retesting.

***

## 🔮 Extensions & Future Work

### 1. Risk Management Enhancements

- Add **stop-loss** on maximum drawdown per trade.  
- Enforce **maximum holding period** (e.g. N days) to avoid stuck spreads.  
- Incorporate **volatility scaling** to size positions by risk.

### 2. Portfolio-Level Extension

- Extend from single-pair (EWC-EWA) to **portfolio of cointegrated pairs**:
  - Multiple equity pairs.
  - Sector ETFs, commodity ETFs.
  - Dynamic allocation based on Sharpe/Sortino.

