# Deep Sequence Models in Commodity Markets

This repository contains the code and report for testing whether an LSTM improves the net Sharpe ratio for a commodity ETF momentum strategy compared to linear baselines.

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
