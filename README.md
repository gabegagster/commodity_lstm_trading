# Deep Sequence Models in Commodity Markets

**Authors:** Gabriel Himmelein & Tomàs Planelles Alonso

This repository evaluates whether a compact, regularized **Long Short-Term Memory (LSTM)** network can improve the **net Sharpe ratio (after transaction costs)** of a time-series momentum strategy across five liquid commodity ETFs (`DBC`, `USO`, `GLD`, `SLV`, `DBA`) compared to **Simple Momentum**, **Ridge / RidgeCV**, and **LightGBM** within the `skfin` walk-forward Mean-Variance optimization framework.

---

## Primary Analysis Notebook (Self-Contained)

The core experimental pipeline, empirical results, interactive plots, and statistical validation are packaged into a single, **fully self-contained Jupyter notebook**:

- **[`notebooks/commodity_lstm_trading.ipynb`](file:///Users/gabrielhimmelein/Projects/commodity_lstm_trading/notebooks/commodity_lstm_trading.ipynb)**
- **Static HTML View**: **[`notebooks/commodity_lstm_trading.html`](file:///Users/gabrielhimmelein/Projects/commodity_lstm_trading/notebooks/commodity_lstm_trading.html)** (full pre-rendered outputs, tables, and figures viewable directly in any web browser without running Jupyter)

### Self-Contained Execution
The notebook is entirely standalone and can be executed independently from top to bottom:
- **No local module imports required**: All components—synchronous data loading, business-day alignment (`freq='B'`), multi-horizon momentum feature engineering (`5d–252d`), PyTorch `CommodityLSTM` architecture, scikit-learn / `skfin` estimator wrapper (`PyTorchLSTMRegressor`), walk-forward backtesting, quadratic transaction cost modeling (`BacktesterWithCost`), and Lo (2002) Newey-West HAC Sharpe ratio inference—are defined natively within the notebook cells.
- **Reproducibility**: Random seeds are explicitly pinned across Python, NumPy, and PyTorch (`set_seed(42)`), and data caching guarantees offline execution without network access.

---

## Repository Structure

```text
commodity_lstm_trading/
├── notebooks/
│   ├── commodity_lstm_trading.ipynb   # Unified, self-contained interactive notebook
│   └── commodity_lstm_trading.html    # Pre-compiled HTML report with all figures & tables
├── scripts/
│   ├── create_notebook.py             # Programmatic notebook generator & execution script
│   └── run_random_search.py           # Standalone CLI script for pre-2020 hyperparameter search
├── src/                               # Modular Python library (mirrors notebook functionality)
│   ├── data/                          # Data loading, synchronization & feature preparation
│   ├── models/                        # CommodityLSTM PyTorch module & scikit-learn wrapper
│   └── evaluation/                    # Backtesting runners & metrics
├── report/                            # Presentation deliverables & notes
│   ├── project_presentation.pdf       # Slide deck (PDF)
│   ├── project_presentation.pptx      # Slide deck (PowerPoint)
│   ├── presentation_notes.md          # 5-minute presentation script & Q&A defense notes
│   └── presentation_notes.pdf         # Compiled presentation notes
├── data/                              # Local parquet cache (ignored by git, auto-populated)
│   ├── cache/                         # HPO and subperiod sensitivity parquets
│   └── yf_returns_2007-01-01_latest/  # Synchronized ETF total returns (2007–2026)
└── requirements.txt                   # Project dependencies
```

### Note on `notebooks/` vs. `src/`
The repository provides the pipeline in two equivalent forms:
1. **`notebooks/commodity_lstm_trading.ipynb`**: Recommended for replication, academic evaluation, and interactive exploration. Contains all model classes, wrappers, and backtests inline so it runs independently with zero external file dependencies.
2. **`src/`**: A structured, modular Python package (`src.data`, `src.models`, `src.evaluation`) providing reusable modules for CLI scripts (e.g. `scripts/run_random_search.py`) or integration into external pipelines.

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gabegagster/commodity_lstm_trading.git
   cd commodity_lstm_trading
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux/macOS
   # or: venv\Scripts\activate   # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook:**
   ```bash
   jupyter lab notebooks/commodity_lstm_trading.ipynb
   # or
   jupyter notebook notebooks/commodity_lstm_trading.ipynb
   ```
   Alternatively, open `notebooks/commodity_lstm_trading.html` in your browser for immediate inspection of all outputs.
