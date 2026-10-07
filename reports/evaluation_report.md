# Evaluation report: macro-aware strategy-permutation selection

## Setup
- Data: macro calendar via Apify (`scrapemint/forexfactory-economic-calendar`), news via Tavily, strategy log = 18 moving-average permutations on yfinance SPY (simulated; no trading logs were supplied).
- Target: next-day P&L of each permutation. Each day the model ranks permutations and the portfolio holds the top 3 (equal weight).
- Validation: expanding-window walk-forward, first 40 days used only for training, refit every 5 days, 1-day embargo, every feature uses information up to the previous close.
- Model: gradient boosting (depth 3, strong regularisation). Baseline: expanding historical mean P&L per permutation, top 3. Also shown: equal-weight of all permutations.

## Out-of-sample performance (annualised with 252 days)
|               |   days |   total_return_% |   ann_return_% |   ann_vol_% |   sharpe |   sortino |   max_drawdown_% |   hit_rate_% |
|:--------------|-------:|-----------------:|---------------:|------------:|---------:|----------:|-----------------:|-------------:|
| model_topk    |     85 |           -6.634 |        -18.414 |      10.05  |   -1.975 |    -2.465 |           -9.276 |       43.529 |
| baseline_topk |     85 |           -5.284 |        -14.866 |       8.513 |   -1.848 |    -2.473 |           -8.093 |       37.647 |
| equal_weight  |     85 |           -2.662 |         -7.687 |       7.377 |   -1.048 |    -1.405 |           -5.399 |       47.059 |

## Model quality
- Mean daily rank IC (prediction vs realised P&L across permutations): model -0.095, baseline -0.103
- RMSE of predicted P&L: model 0.00799, baseline 0.00740
- Paired bootstrap 95% CI for (model Sharpe - baseline Sharpe): [-2.73, +2.61]

## Limitations
- Only 85 out-of-sample days and a few dozen macro releases: Sharpe estimates are very noisy and the CI above is wide. Treat differences as indicative, not proof.
- Permutations are highly correlated (same price series), so the effective number of independent bets is small.
- Sentiment is a lexicon score over search snippets at roughly weekly resolution, not a trained model.
- Costs, slippage and position sizing are ignored.
- The Apify calendar actor can drop events on some runs; see README for known gaps.
