"""Walk-forward model: predict each permutation's next-day P&L, trade the top-k.

Expanding window; the model is refit every `retrain_every` days using only days up to
t - 1 - embargo.  Baseline: expanding historical mean P&L per permutation (same cut-off).

Usage: python -m src.model
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .common import PROC, load_cfg
from .features import FEATS


def walk_forward(panel, cfg):
    m = cfg["model"]
    feats = [c for c in FEATS if c in panel.columns]
    dates = sorted(panel["date"].unique())
    if len(dates) <= m["min_train_days"] + 5:
        raise SystemExit(f"Only {len(dates)} trading days; need more than min_train_days={m['min_train_days']}")
    preds, model = [], None
    for i, d in enumerate(dates):
        if i < m["min_train_days"]:
            continue
        train_days = dates[: i - m["embargo_days"]]
        train = panel[panel["date"].isin(train_days)]
        if model is None or (i - m["min_train_days"]) % m["retrain_every"] == 0:
            model = HistGradientBoostingRegressor(
                max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0,
                min_samples_leaf=40, random_state=0).fit(train[feats], train["pnl"])
        test = panel[panel["date"] == d].copy()
        test["pred"] = model.predict(test[feats])
        test["base_pred"] = test["perm_id"].map(train.groupby("perm_id")["pnl"].mean())
        preds.append(test)
    return pd.concat(preds, ignore_index=True)


def main():
    cfg = load_cfg()
    k = cfg["model"]["top_k"]
    panel = pd.read_csv(PROC / "panel.csv", parse_dates=["date"])
    pred = walk_forward(panel, cfg)

    def topk_ret(g, col):
        return g.nlargest(k, col)["pnl"].mean()

    groups = list(pred.groupby("date"))
    daily = pd.DataFrame({
        "date": [d for d, _ in groups],
        "model_topk": [topk_ret(g, "pred") for _, g in groups],
        "baseline_topk": [topk_ret(g, "base_pred") for _, g in groups],
        "equal_weight": [g["pnl"].mean() for _, g in groups],
    })
    pred.to_csv(PROC / "predictions.csv", index=False)
    daily.to_csv(PROC / "daily_returns.csv", index=False)
    print(f"walk-forward: {daily['date'].nunique()} out-of-sample days, top-{k} permutations per day")


if __name__ == "__main__":
    main()
