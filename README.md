# CrowdWisdomTrading Macro Prediction Pipeline

Pipeline: macro event ingestion (Apify) -> news (Tavily) -> features -> walk-forward ML -> evaluation and charts.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add APIFY_TOKEN and TAVILY_API_KEY
```

## Step 1: Macro calendar (Apify)
- Actor: `scrapemint/forexfactory-economic-calendar`
- Input: custom range in 30-day chunks, `currencies=["USD"]`, `impactLevels=["high"]`
```bash
python -m src.ingest_apify --test   # 1-week check
python -m src.ingest_apify          # full 180 days
python -m src.clean_events          # -> data/processed/macro_events.csv
```
Run inputs and run IDs are saved in `data/raw/manifest.json` for reproducibility.

## Steps 2-6: news, features, model, report (one command)
```bash
# .env needs TAVILY_API_KEY as well
python -m src.run_all              # clean_events -> Tavily news -> sentiment -> strategy log
                                   # -> features (SQLite join) -> walk-forward model -> evaluation -> charts -> outlook draft
python -m src.run_all --skip-news  # reuse data/raw/news.json
```
Outputs: `data/macro.db` (SQLite: macro_events, news, sentiment_daily, strategy_log, daily_features),
`reports/evaluation_report.md` (Sharpe, Sortino, Max Drawdown), `reports/charts/*.png`, `reports/macro_outlook.md` (draft: edit before submitting).

Trading logs: none were supplied, so permutations of a moving-average strategy are simulated on SPY daily prices (yfinance).
To use real logs, write them to `data/processed/strategy_log.csv` with columns
`date, perm_id, fast, slow, event_filter, position, ret, pnl` and skip `src.strategy_log`.

Example SQL: `SELECT f.date, f.is_cpi_day, AVG(s.pnl) FROM strategy_log s JOIN daily_features f ON s.date=f.date GROUP BY f.date;`

## Known data gaps (Apify ForexFactory actor)
Multi-day pages are only partly captured, so the calendar was pulled in 7-day chunks; some weeks (e.g. FOMC weeks) still came back incomplete. Edit this list with your final gaps.

## Notes
- ForexFactory times come back in a display zone that is NOT labelled. In the test run NFP showed 7:30, i.e. 8:30 ET expressed in Central time, so `calendar.source_tz` is `America/Chicago`. `clean_events` checks that key releases land at 08:30 ET after conversion.
- `actual` is only available on custom/month ranges, so the pipeline uses custom ranges.
