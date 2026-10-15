"""Script to run Random Search hyperparameter optimization for LSTM within Backtester and cache results."""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from skfin.backtesting import Backtester
from skfin.estimators import MeanVariance
from skfin.metrics import sharpe_ratio
from src.data.load_data import load_commodity_data, prepare_backtest_data
from src.models.lstm_wrapper import PyTorchLSTMRegressor

def run_hpo(n_iter=15, force_recompute=False):
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / "hpo_lstm.parquet"

    if fname.exists() and not force_recompute:
        print(f"Loading cached HPO results from {fname}...")
        results = pd.read_parquet(fname)
        print(f"Loaded {len(results)} parameter trials. Best Sharpe: {results['sr'].max():.3f}")
        return results

    print("Loading data...")
    data = load_commodity_data(cache_dir="data")
    ret = data["total_returns"]
    X_mom, y_mom, ret_mom = prepare_backtest_data(ret, feature_type="momentum")

    start_date = "2012-01-01"
    end_date = "2019-12-31"

    param_distributions = {
        "hidden_size": randint(6, 17),
        "lookback": randint(15, 45),
        "weight_decay": loguniform(1e-4, 5e-3),
        "dropout": uniform(0.1, 0.25),
        "lr": loguniform(5e-4, 2e-3),
    }

    sampler = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))

    results_ = {}
    print(f"Running Random Search ({n_iter} iterations) on calibration window {start_date} to {end_date}...")
    for i, prm in enumerate(tqdm(sampler)):
        prm_clean = {
            "hidden_size": int(prm["hidden_size"]),
            "lookback": int(prm["lookback"]),
            "weight_decay": float(prm["weight_decay"]),
            "dropout": float(prm["dropout"]),
            "lr": float(prm["lr"]),
        }

        estimator = make_pipeline(
            StandardScaler(with_mean=False),
            PyTorchLSTMRegressor(
                hidden_size=prm_clean["hidden_size"],
                lookback=prm_clean["lookback"],
                weight_decay=prm_clean["weight_decay"],
                dropout=prm_clean["dropout"],
                lr=prm_clean["lr"],
                epochs=10,
                batch_size=64,
                loss_type="huber",
                val_split=0.15,
                patience=4,
                random_state=42,
                verbose=False,
            ),
            MeanVariance(),
        )

        bt = Backtester(
            estimator=estimator,
            max_train_size=756,
            test_size=21,
            start_date=start_date,
            end_date=end_date,
            name=f"trial_{i}",
        )
        pnl = bt.train(X_mom, y_mom, ret_mom)
        sr = sharpe_ratio(pnl)

        record = dict(prm_clean)
        record["sr"] = float(sr)
        record["ann_ret"] = float(pnl.mean() * 260 * 100)
        record["ann_vol"] = float(pnl.std() * np.sqrt(260) * 100)
        results_[i] = record
        print(f"Trial {i+1}/{n_iter}: hs={prm_clean['hidden_size']}, lb={prm_clean['lookback']}, wd={prm_clean['weight_decay']:.4f}, lr={prm_clean['lr']:.4f} -> Sharpe = {sr:.3f}")

    results = pd.DataFrame.from_dict(results_, orient="index").sort_values("sr")
    results.to_parquet(fname)
    print(f"Saved HPO results to {fname}. Best trial Sharpe: {results['sr'].max():.3f}")
    return results

if __name__ == "__main__":
    run_hpo(n_iter=15, force_recompute=True)
