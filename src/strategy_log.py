"""Build the strategy-permutation log (daily P&L per permutation).

The brief mentions historical trading logs but none were supplied, so permutations of a simple
moving-average strategy are simulated on daily prices (yfinance; synthetic prices only as an offline
fallback, flagged in data/processed/strategy_meta.json).  If real logs arrive, write them in this same
schema to data/processed/strategy_log.csv: date, perm_id, fast, slow, event_filter, position, ret, pnl.

Permutation = (fast MA, slow MA, event_filter). event_filter 'flat_on_event_days' goes flat on days
with a scheduled CPI/NFP/PPI/FOMC release (schedule is public, so no look-ahead).

Usage: python -m src.strategy_log
"""
import json
from datetime import timedelta
from itertools import product

import numpy as np
import pandas as pd

from .common import PROC, load_cfg, load_events
from .db import write_table


def load_prices(ticker, start, end):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=(start - timedelta(days=220)).strftime("%Y-%m-%d"),
                         end=(end + timedelta(days=2)).strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)
        close = df["Close"].squeeze().dropna()
        if len(close) > 150:
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            return close.astype(float), f"yfinance {ticker}"
    except Exception as exc:
        print(f"[warn] price download failed: {type(exc).__name__}: {str(exc)[:80]}")
    print("[warn] USING SYNTHETIC PRICES (offline fallback). Do not submit results from synthetic prices.")
    idx = pd.bdate_range(start - timedelta(days=220), end + timedelta(days=1))
    rng = np.random.default_rng(42)
    return pd.Series(400 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))), index=idx), "SYNTHETIC"


def main():
    cfg = load_cfg()
    s = cfg["strategy"]
    ev = load_events()
    start, end = ev["date"].min(), ev["date"].max()
    close, source = load_prices(s["ticker"], start, end)
    ret = close.pct_change()
    key_days = pd.DatetimeIndex(ev.loc[ev["family"].notna(), "date"].unique())

    frames = []
    for fast, slow, filt in product(s["fast"], s["slow"], s["event_filter"]):
        if fast >= slow:
            continue
        signal = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(int) * 2 - 1
        pos = signal.shift(1)                       # decided at previous close
        if filt == "flat_on_event_days":
            pos = pos.where(~close.index.isin(key_days), 0)
        d = pd.DataFrame({"date": close.index, "perm_id": f"f{fast}_s{slow}_{filt}", "fast": fast,
                          "slow": slow, "event_filter": filt, "position": pos.values,
                          "ret": ret.values, "pnl": (pos * ret).values})
        frames.append(d[(d["date"] >= start) & (d["date"] <= end)].dropna(subset=["pnl"]))
    log = pd.concat(frames, ignore_index=True)

    PROC.mkdir(parents=True, exist_ok=True)
    log.to_csv(PROC / "strategy_log.csv", index=False)
    (PROC / "strategy_meta.json").write_text(json.dumps({"price_source": source, "ticker": s["ticker"]}))
    write_table(log, "strategy_log")
    print(f"{log['perm_id'].nunique()} permutations x {log['date'].nunique()} days "
          f"({start.date()} -> {end.date()}), prices: {source}")


if __name__ == "__main__":
    main()
