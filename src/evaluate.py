"""Out-of-sample evaluation: Sharpe, Sortino, Max Drawdown + model quality -> reports/evaluation_report.md

Usage: python -m src.evaluate
"""
import json

import numpy as np
import pandas as pd

from .common import PROC, REPORTS, load_cfg


def metrics(r, periods=252):
    r = r.dropna()
    mu, sd = r.mean(), r.std(ddof=1)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    eq = (1 + r).cumprod()
    return {
        "days": int(len(r)),
        "total_return_%": 100 * (eq.iloc[-1] - 1),
        "ann_return_%": 100 * (eq.iloc[-1] ** (periods / len(r)) - 1),
        "ann_vol_%": 100 * sd * np.sqrt(periods),
        "sharpe": mu / sd * np.sqrt(periods) if sd > 0 else np.nan,
        "sortino": mu / downside * np.sqrt(periods) if downside > 0 else np.nan,
        "max_drawdown_%": 100 * (eq / eq.cummax() - 1).min(),
        "hit_rate_%": 100 * (r > 0).mean(),
    }


def sharpe_of(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(252) if x.std(ddof=1) > 0 else np.nan


def main():
    cfg = load_cfg()
    daily = pd.read_csv(PROC / "daily_returns.csv", parse_dates=["date"]).set_index("date")
    pred = pd.read_csv(PROC / "predictions.csv", parse_dates=["date"])
    meta = json.loads((PROC / "strategy_meta.json").read_text())

    table = pd.DataFrame({c: metrics(daily[c]) for c in daily.columns}).T.round(3)

    # model quality
    groups = list(pred.groupby("date"))
    ic = pd.Series([g["pred"].corr(g["pnl"], method="spearman") for _, g in groups])
    ic_base = pd.Series([g["base_pred"].corr(g["pnl"], method="spearman") for _, g in groups])
    rmse_m = float(np.sqrt(((pred["pred"] - pred["pnl"]) ** 2).mean()))
    rmse_b = float(np.sqrt(((pred["base_pred"] - pred["pnl"]) ** 2).mean()))

    # paired bootstrap: is model Sharpe above baseline Sharpe? (iid resampling of days; ignores autocorrelation)
    rng = np.random.default_rng(0)
    a, b = daily["model_topk"].to_numpy(), daily["baseline_topk"].to_numpy()
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(a), len(a))
        diffs.append(sharpe_of(pd.Series(a[i])) - sharpe_of(pd.Series(b[i])))
    diffs = np.array(diffs)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])

    out = {"metrics": table.to_dict("index"), "mean_rank_ic_model": float(ic.mean()),
           "mean_rank_ic_baseline": float(ic_base.mean()), "rmse_model": rmse_m, "rmse_baseline": rmse_b,
           "sharpe_diff_ci95": [float(lo), float(hi)], "price_source": meta["price_source"]}
    (PROC / "metrics.json").write_text(json.dumps(out, indent=2))

    m = cfg["model"]
    warn = ("\n> **WARNING: results below use SYNTHETIC prices (offline fallback). Re-run with internet access "
            "before submitting.**\n") if meta["price_source"] == "SYNTHETIC" else ""
    md = f"""# Evaluation report: macro-aware strategy-permutation selection
{warn}
## Setup
- Data: macro calendar via Apify (`{cfg['apify']['actor_id']}`), news via Tavily, strategy log = {len(pred['perm_id'].unique())} moving-average permutations on {meta['price_source']} (simulated; no trading logs were supplied).
- Target: next-day P&L of each permutation. Each day the model ranks permutations and the portfolio holds the top {m['top_k']} (equal weight).
- Validation: expanding-window walk-forward, first {m['min_train_days']} days used only for training, refit every {m['retrain_every']} days, {m['embargo_days']}-day embargo, every feature uses information up to the previous close.
- Model: gradient boosting (depth 3, strong regularisation). Baseline: expanding historical mean P&L per permutation, top {m['top_k']}. Also shown: equal-weight of all permutations.

## Out-of-sample performance (annualised with 252 days)
{table.to_markdown()}

## Model quality
- Mean daily rank IC (prediction vs realised P&L across permutations): model {ic.mean():+.3f}, baseline {ic_base.mean():+.3f}
- RMSE of predicted P&L: model {rmse_m:.5f}, baseline {rmse_b:.5f}
- Paired bootstrap 95% CI for (model Sharpe - baseline Sharpe): [{lo:+.2f}, {hi:+.2f}]

## Limitations
- Only {len(daily)} out-of-sample days and a few dozen macro releases: Sharpe estimates are very noisy and the CI above is wide. Treat differences as indicative, not proof.
- Permutations are highly correlated (same price series), so the effective number of independent bets is small.
- Sentiment is a lexicon score over search snippets at roughly weekly resolution, not a trained model.
- Costs, slippage and position sizing are ignored.
- The Apify calendar actor can drop events on some runs; see README for known gaps.
"""
    (REPORTS).mkdir(exist_ok=True)
    (REPORTS / "evaluation_report.md").write_text(md)
    print(table.to_string())
    print(f"\nIC model {ic.mean():+.3f} vs baseline {ic_base.mean():+.3f}; Sharpe-diff CI95 [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
