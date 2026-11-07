# 5-Minute Notebook Walkthrough (Compact) + Q&A Deep-Dive Reference
**Target File to Scroll:** `notebooks/commodity_lstm_trading_v2.ipynb` (or `.html`)  
**Total Time Budget:** **5:00 minutes** (**3:15 Gabriel** [Sections 1–6, `Out [1]–[8]`] | **1:45 Tomàs** [Sections 7–8, `Out [9]–[11]`])

---

## 1. Live 5-Minute Walkthrough (Core Talking Points Only)

### Part 1 — Gabriel (`0:00 – 3:15` | 3 min 15 sec — Sections 1 to 6)

#### Title & Section 1: Data & Stylized Facts (`0:00 – 0:45` | 45 sec) — *Scroll to `Out [1]` & `Out [2]`*
* **Goal:** Test whether a small neural network with memory **[compact LSTM]** beats simple trend rules **[Simple Momentum]** and standard ML **[RidgeCV, LightGBM]** after trading costs **[Net Sharpe Ratio]** across 5 commodity ETFs: broad commodities, oil, gold, silver, and agriculture **[DBC, USO, GLD, SLV, DBA]**.
* **Clean Setup:** Aligned from Jan 2007 (`5,158` business days) **[synchronized inception]**, predicting tomorrow's return from today's data **[zero look-ahead bias, $y_t = r_{t+1}$]**, scaled to 10% yearly risk **[`TARGET_VOL` covariance scaling]**.
* **Why Commodities Need This (`Out [1]` & `Out [2]`):**
  1. Oil (`37.5%` vol) and silver (`33.0%`) swing twice as hard as agriculture (`16.7%`) and gold (`18.1%`) **[volatility dispersion]** $\rightarrow$ requires risk scaling.
  2. Passive oil (`USO`) earns `0.07` Sharpe due to monthly futures rolling losses **[contango / negative roll yield]**, whereas active trend trading can short downturns and ride rallies **[backwardation]**.
  3. Frequent extreme crash days **[fat tails / kurtosis]** motivate a crash-resistant loss **[Huber loss]**, while low gold–oil correlation (`0.20`) gives strong diversification.

#### Section 2 & 3: Features & LSTM Architecture (`0:45 – 1:35` | 50 sec) — *Scroll to `Out [3]` & `Out [4]`*
* **Multi-Speed Trend Inputs:** Instead of just 1-day returns **[`X_raw`]** or 1-year trends **[`X_mom_252`]**, we stack 1-week, 1-month, 3-month, 6-month, and 1-year average returns (`5d–252d`) across the 5 ETFs **[multi-horizon momentum `X_mom`, 25 features]**.
* **Walk-Forward Training:** Retrained monthly (`21d`) on a rolling 3-year window (`756d`), tested out-of-sample on **2020–2026** **[`skfin.Backtester`]**.
* **Anti-Overfitting Design (`Out [4]`):** Because daily returns are mostly noise **[stationary returns]**, we use a tiny 8-unit network **[`hidden_size=8`]**, 20% dropout, strong weight penalty **[`weight_decay=1e-3`]**, crash-resistant **Huber loss**, chronological early stopping on the newest 15% of training data, and a 20-day trailing buffer **[`last_train_X_`]** so test folds have full history.

#### Section 4 & 5: Pre-2020 Calibration vs. 2020–2026 Shocks (`1:35 – 2:25` | 50 sec) — *Scroll to `Out [5]` & `Out [6]`*
* **Fair Calibration (`2012–2019`, `Out [5]`):** A 15-trial random search before the test period favored a longer 32-day memory with weak regularization **[Trial 9: `lookback=32, hs=9, wd=3.4e-4`, IS Sharpe `0.65`]** because 2012–2019 was calm.
* **Out-of-Sample Results (`2020–2026`, `Out [6]`):**
  * **1-Year Trend (`Simple Momentum 252d`)** is our strongest linear baseline (**`0.77` Gross Sharpe**, low turnover `30.8`).
  * **Pre-2020 Winner (`lookback=32`)** drops to **`-0.07` Sharpe** in 2020–2026 because a 32-day memory holds onto pre-crash states too long during sudden V-shaped reversals (COVID oil crash, 2022 Ukraine shock) **[memory lag at turning points]**.
  * **Compact 20-Day LSTM (`lookback=20, hs=8, wd=1e-3`)** resets memory within one trading month $\rightarrow$ jumps to **`1.27` Gross Sharpe** (`17.6%` return).

#### Section 6: Seed Ensemble, Subperiods & $2 \times 2$ Ablation (`2:25 – 3:15` | 50 sec) — *Scroll to `Out [7]` & `Out [8]`*
* **5-Seed Ensemble (`Out [7]`):** Single seeds vary (`-0.33` to `1.27` **[initialization variance]**), so we average positions across 5 random seeds $\rightarrow$ cuts trading turnover in half (`64.8`), shrinks max loss to **`-5.89%`** **[Max Drawdown]**, and yields **`0.81` Gross Sharpe**.
* **Training Windows & $2 \times 2$ Ablation (`Out [8]`):** Across 2013–2026 (`Table 6a`), the **3-year window (`0.68` SR)** beats noisy 2-year (`0.35`) and stale 5-year (`0.05`) windows (`0.88` full-sample for the Regime-Adaptive combination). And `Table 6b` ($2 \times 2$ ablation) proves you need **both** multi-horizon features and the compact 20-day network (`1.27` together vs. `0.17` and `0.20` alone).

---

### Part 2 — Tomàs (`3:15 – 5:00` | 1 min 45 sec — Sections 7 & 8)

#### Section 7.1 & 7.2: Tuned Baselines & Transaction Costs (`3:15 – 4:05` | 50 sec) — *Scroll to `Out [9]` & `Out [10]`*
* **Tuned Baselines (`Out [9]`):** Giving `RidgeCV` and 12-trial `LightGBM` the same 2012–2019 tuning budget still yields weak 2020–2026 performance (`-0.44` and `0.15` Gross Sharpe).
* **Quadratic Trading Costs (`Out [10]`):** Costs scale with 60-day volatility and the square of trade size **[`compute_pnl_components`]** $\rightarrow$ fast-flipping models (`1d Momentum`, `LightGBM`, holding `1.5–3.3` days) lose **`-3.2%` to `-6.1%` per year**, wiping out their net edge (`0.26` and `-0.06` Net Sharpe).
* **Cost-Aware LSTM (`WalkForwardMeanVarianceWithCost`):** Passing yesterday's holdings **[`past_h`]** into the optimizer trades only partway toward the new target $\rightarrow$ cuts cost drag by **57%** (`-0.36%` vs. `-0.84%`) and achieves **`1.30` Gross / `1.26` Net Sharpe** (and **`1.23` Net Sharpe even at `3x` costs**), while the **5-Seed Ensemble** achieves **`0.78` Net Sharpe** (vs. `0.76` for `252d Momentum`).

#### Section 7.3: Statistical Significance & Universe Stability (`4:05 – 4:35` | 30 sec) — *Scroll to `Out [11]`*
* **HAC Significance (`Out [11]` Table):** Using autocorrelation-corrected standard errors **[Lo (2002) Newey-West HAC]**, both the **Cost-Aware LSTM** (**`1.26` Net SR, $p = 0.0014$**) and **5-Seed Ensemble** (**`0.78` Net SR, $p = 0.0327$**) are statistically significant with 95% confidence intervals strictly above zero.
* **Leave-One-Out Test (`Out [11]` Bar Chart):** Dropping any single ETF one at a time keeps Gross Sharpe between **`0.82` and `1.28`** **[off-the-top universe stability]**, proving the strategy doesn't depend on one lucky asset like oil (`USO`).

#### Section 8: Conclusion & Limitations (`4:35 – 5:00` | 25 sec) — *Scroll to Section 8 Markdown*
* **Verdict:** A compact 20-day LSTM on multi-horizon momentum (`5d–252d`) with cost-aware optimization (`1.26` Net SR) or 5-seed ensembling (`0.78` Net SR, `-5.89%` Max DD) beats linear and tree baselines after costs.
* **4 Extensions:** (1) Train directly on portfolio Sharpe **[end-to-end loss]**, (2) re-tune lookback every 1–2 years **[rolling nested HPO]**, (3) add sideways-market filters (`2016–2019`), and (4) add futures curve slope features **[term-structure carry]**.

---
---

## 2. Q&A Deep-Dive Reference (By Section — For Backup Only)

### Q&A — Section 1: Data Pipeline & Stylized Facts
* **Why start in January 2007 (`N = 5,158` business days)?**  
  Commodity ETFs did not exist earlier (`GLD` launched Nov 2004, `DBC/USO/SLV` in 2006, `DBA` in Jan 2007). Starting at `DBA`'s inception avoids missing-data imputation and covers 100% of the modern electronic ETF era.
* **Exact ETF Statistics (`Out [1]`):**  
  * `DBC` (Broad): Mean `5.92%`, Vol `19.36%`, Sharpe `0.31`, Skew `-0.35`, Kurtosis `3.46`, Min `-7.94%`.
  * `USO` (Oil): Mean `2.53%`, Vol `37.49%`, Sharpe `0.07`, Skew `-0.55`, Kurtosis `9.47`, Min `-25.32%` (April 2020 WTI negative oil episode).
  * `GLD` (Gold): Mean `10.87%`, Vol `18.14%`, Sharpe `0.60`, Skew `-0.27`, Kurtosis `7.17`, Min `-10.27%`.
  * `SLV` (Silver): Mean `12.73%`, Vol `33.00%`, Sharpe `0.39`, Skew `-1.05`, Kurtosis `13.41`, Min `-28.54%`.
  * `DBA` (Agriculture): Mean `3.94%`, Vol `16.68%`, Sharpe `0.24`, Skew `-0.13`, Kurtosis `6.41`, Min `-8.61%`.
* **Doesn't `DBC` overlap with `USO`, `GLD`, and `DBA`?**  
  Yes, `DBC` is a broad composite commodity index ETF (`corr = 0.79` with `USO`). However, the Markowitz covariance matrix $V$ in `MeanVariance` explicitly accounts for cross-asset correlations, and our leave-one-out test (`Excl. DBC`) still achieves a Gross Sharpe above `1.0`.
* **How is target volatility scaled?**  
  Daily target risk is `TARGET_VOL = 0.10 / sqrt(260) = 0.006202` (10% annualized). Realized OOS volatility can deviate slightly (`12–14%` for single models, `6.03%` for the ensemble) because covariance is estimated on the trailing window and applied out-of-sample.

### Q&A — Sections 2 & 3: Features, Baselines & PyTorch LSTM Wrapper
* **How do you prevent look-ahead leakage in `Backtester` and `PyTorchLSTMRegressor`?**  
  1. Target at index $t$ is `ret.shift(-1)` ($r_{t+1}$), and PnL is `h.shift(1).mul(ret)`.
  2. `StandardScaler(with_mean=False)` is inside `make_pipeline(...)`, so scaling parameters are fit strictly inside each 756-day training fold.
  3. Validation split (`val_split=0.15`, ~110 days) is sliced chronologically from the end of each training fold *before* mini-batch shuffling.
  4. Sequence boundary buffer (`last_train_X_`): because `test_size=21` is shorter than or comparable to `lookback` (`20–60` days), `fit()` caches the last `lookback` rows of `X_train` and prepends them in `predict(X_test)` so test day 1 uses `[t - L + 1, ..., t]` without peeking ahead or dropping initial test days.
* **Why initialize the forget-gate bias to $b_f = +1.0$?**  
  In PyTorch's `nn.LSTM`, the bias vector has length $4 \times \text{hidden\_size}$ ordered `[input, forget, cell, output]`. Setting the second quarter (`n//4 : n//2`) to `+1.0` ensures $\sigma(b_f) \approx 0.73$ at epoch 1 so gradients flow across the 20-step sequence instead of vanishing immediately.
* **Why Huber loss ($\delta = 0.01$) instead of MSE?**  
  A 1% daily move ($\delta = 0.01$) is roughly 1 standard deviation for a diversified commodity basket. For normal days ($|e| \le 0.01$), Huber is quadratic ($0.5 e^2$); for extreme tail shocks ($|e| > 0.01$, like `-25%` in `USO`), gradients are capped at $\pm \delta$, preventing a single outlier day from dominating AdamW updates.

### Q&A — Sections 4 & 5: Hyperparameter Search & Out-of-Sample Results
* **Exact Random Search Space (`2012–2019`, 15 trials, `random_state=42`):**  
  `hidden_size` $\in \text{randint}(6, 17)$, `lookback` $\in \text{randint}(15, 45)$, `weight_decay` $\in \text{loguniform}(10^{-4}, 5 \times 10^{-3})$, `dropout` $\in \text{uniform}(0.10, 0.35)$, `lr` $\in \text{loguniform}(5 \times 10^{-4}, 2 \times 10^{-3})$.
* **Wasn't choosing `Compact LSTM (lb=20, hs=8, wd=1e-3)` after seeing 2020–2026 a form of hindsight?**  
  We explicitly report **both** the frozen 2012–2019 random search winner (`Trial 9: lb=32, hs=9, wd=3.4e-4`, OOS SR `-0.07`) and the canonical 1-month (`lb=20`) strongly regularized architecture (`wd=1e-3`, OOS SR `1.27`). Showing both side-by-side demonstrates *why* static pre-2020 hyperparameter selection fails across structural breaks (memory lag) and motivates rolling nested HPO in Section 8.

### Q&A — Section 6: Multi-Seed Ensembling & Subperiod Sensitivity
* **Why is the 5-Seed Ensemble Sharpe (`0.81`) higher than the average of the 5 seeds (`0.36`)?**  
  The 5 seeds (`42: 1.27`, `101: 0.35`, `2024: -0.17`, `7: 0.69`, `999: -0.33`) have mean return `4.78%` (the exact average of the 5 seeds' returns), **but** because their forecast errors are weakly correlated across seeds, averaging their holdings reduces portfolio volatility from `~14.1%` down to **`6.03%`**. Since $\text{Sharpe} = \mu / \sigma$, the 57% drop in denominator volatility boosts the ensemble Sharpe to `0.809` (`0.81`) and cuts max net drawdown to `-5.89%`.
* **Exact Subperiod Sharpe Table (`Table 6a`, 2013–2026):**  
  * **2013–2015 (Post-GFC Bear Crash):** `252d Mom = 0.63`, `Pre-2020 LSTM (lb=32) = 0.76`, `Compact LSTM (3y) = 0.22`.
  * **2016–2019 (Sideways Recovery):** `252d Mom = 0.26`, `Pre-2020 LSTM (lb=32) = 0.05`, `Compact LSTM (3y) = -0.41`.
  * **2020–2026 (Pandemic & Inflation):** `252d Mom = 0.77`, `Pre-2020 LSTM (lb=32) = -0.07`, `Compact LSTM (2y) = 0.99`, `Compact LSTM (3y) = 1.27`, `Compact LSTM (5y) = 0.34`.
  * **Full 2013–2026 (13 Years):** `252d Mom = 0.61`, `Compact LSTM (2y) = 0.35`, `Compact LSTM (3y) = 0.68`, `Compact LSTM (5y) = 0.05`, `Regime-Adaptive LSTM = 0.88`.

### Q&A — Section 7: Baselines, Transaction Costs & Statistical Validation
* **How were `RidgeCV` and `MultiLGBMRegressor` tuned?**  
  * `RidgeCV`: Tuned across `alphas = [0.01, 0.1, 1, 10, 100, 1000]` inside every walk-forward fold (`mode alpha = 1000.0`; IS SR `-0.062`, OOS Gross SR `-0.44`, Net SR `-0.46`).
  * `MultiLGBMRegressor`: 12-trial `ParameterSampler` on 2012–2019 (`n_estimators=30`, `min_child_samples=10`, `lr=0.0117`, `max_depth=5`; IS SR `0.142`, OOS Gross SR `0.15`, Net SR `-0.06`).
* **How is the volatility-liquidity cost factor (`vlf`) defined?**  
  $\text{vlf}_{t,i} = \frac{\sigma_{t,i}^{(60\text{d})}}{1/N} \times 2 \times 10^{-3}$ (with $N=5$ ETFs), computed strictly point-in-time using trailing 60-day volatility (expanding volatility for the first 20 days, zero backward filling). Daily impact cost is $\sum_{i=1}^5 \text{vlf}_{t,i} (\Delta h_{t,i})^2$.
* **Why did you subclass `MeanVarianceWithCost` and `BacktesterWithCost`?**  
  1. `skfin`'s default `BacktesterWithCost` assumes `X.shape[1] == y.shape[1]`, whereas `X_mom` has 25 features and `y_mom` has 5 assets (`MultiAssetBacktesterWithCost` uses `y.columns` for holdings).
  2. `WalkForwardMeanVarianceWithCost` averages `vol_liquidity_factor` across the 21-day test block, takes the most recent holding `past_h[-1:]`, computes Gârleanu–Pedersen cost-aware holdings, and rescales ex-ante portfolio volatility $\sqrt{h^\top V h}$ to `TARGET_VOL`.
* **How does the Lo (2002) Newey-West HAC test work (`Out [11]`)?**  
  Standard IID Sharpe standard error is $\sqrt{(1 + 0.5\,\text{SR}_d^2)/T}$. To account for serial correlation and volatility clustering, we compute the influence function of the Sharpe ratio $\phi_t = \frac{r_t - \mu}{\sigma} - \frac{\mu}{2\sigma^3}\big((r_t - \mu)^2 - \sigma^2\big)$ and apply a Bartlett kernel up to lag $q = \lfloor 4(T/100)^{2/9}\rfloor = 7$ days (`T = 1,770` OOS days).
* **Exact Leave-One-ETF-Out OOS Gross Sharpe Ratios (`Out [11]` Bar Chart):**  
  Full 5-ETF universe: `1.27` | Excluding any single ETF stays strictly between **`0.82` and `1.28`**.
