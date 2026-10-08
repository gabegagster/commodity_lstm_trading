# Deep Sequence Models in Commodity Markets

This repository implements a quantitative trading pipeline that applies Long Short-Term Memory (LSTM) networks to predict returns for major commodity ETFs (DBC, USO, GLD, SLV, DBA). The project evaluates whether deep sequence models can capture nonlinear market dynamics better than traditional linear algorithms (Ridge, LightGBM, simple momentum). Predictions are fed into a walk-forward Mean-Variance optimizer to analyze whether the LSTM yields a statistically significant improvement in risk-adjusted returns (net Sharpe) once transaction costs and overfitting risks are strictly accounted for.

## Repository Structure
- `data/`: Raw and processed financial data (ignored by git).
- `notebooks/`: Walk-forward training, EDA, and backtesting notebooks.
- `src/`: Reusable Python modules for data loading, model architecture, and evaluation.
- `report/`: LaTeX source files for the final project report.

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
