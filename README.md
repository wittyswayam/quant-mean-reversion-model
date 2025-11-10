# 📉 Mean Reversion Strategies in Quantitative Finance  
**Algorithmic Pairs Trading and Statistical Arbitrage with Python**

This project implements and tests **mean reversion trading strategies** using Python.  
It focuses on detecting cointegrated asset pairs, generating entry/exit signals based on z-scores, and backtesting the performance of these strategies using historical market data.

---

## 🚀 Key Features
- **Cointegration Testing:** Implements both Engle–Granger and Johansen methods to identify correlated assets.  
- **Signal Generation:** Uses z-score thresholds to determine long and short trade entries.  
- **Rolling Backtesting:** Evaluates performance with rolling windows to simulate real-world conditions.  
- **Performance Metrics:** Calculates Sharpe Ratio, CAGR, and maximum drawdown.  
- **Automated Visualizations:** Generates charts such as equity curves, spreads, and eigenvalue plots saved in the `images/` folder.

---

## ⚙️ Project Workflow
1. **Data Collection** – Fetches asset price data via `yfinance` or CSV input.  
2. **Testing for Cointegration** – Runs statistical tests to identify stable asset pairs.  
3. **Trading Logic** – Enters/exits trades based on z-score signals and thresholds.  
4. **Backtesting** – Measures performance, returns, and risk.  
5. **Visualization** – Plots spreads, trade signals, and strategy performance metrics.

---

## 🧩 Tech Stack
- **Language:** Python 3.8+  
- **Libraries:** pandas, numpy, matplotlib, seaborn, statsmodels, yfinance, scipy  
- **Tools:** Jupyter Notebook, Git, VS Code  

---

## 📊 Example Outputs
The project automatically saves result visuals in the `images/` folder:
- `Johansen Eigen Value Demo.png` – Eigenvalue plot from Johansen test  
- `pairs_spread.png` – Price spread and z-score signal chart  
- `equity_curve.png` – Backtested portfolio performance  

---
