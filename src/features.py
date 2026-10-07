"""Feature engineering + SQL join of macro, sentiment and strategy log (data/macro.db).

All features for day t use information available at the CLOSE of day t-1 (releases are
scheduled/public, so calendar features for day t are fine; surprises and sentiment are shifted by 1).

Usage: python -m src.features
"""
import numpy as np
import pandas as pd

from .common import PROC, SIGN, load_events
from .db import read_sql, write_table

FEATS = ["fast", "slow", "is_flat_filter", "is_cpi_day", "is_nfp_day", "is_ppi_day", "is_fomc_day",
         "n_key_events_today", "days_to_next_key", "last_cpi_z", "last_nfp_z", "last_unemp_z",
         "last_ppi_z", "surprise_index", "sent_7d", "sent_chg", "dow"]


def surprise_z(ev):
    """Signed, standardized surprise per release; std uses ONLY earlier releases (no look-ahead)."""
    out = {}
    for fam in SIGN:
        e = ev[ev["family"] == fam].sort_values("ts_utc").dropna(subset=["actualNumeric", "forecastNumeric"])
        s = (e["actualNumeric"] - e["forecastNumeric"]).to_numpy(dtype=float)
        z = []
        for i, v in enumerate(s):
            prev = s[:i]
            sd = prev.std(ddof=1) if len(prev) >= 3 else np.nan
            z.append(SIGN[fam] * v / sd if sd and sd > 0 else np.nan)
        out[fam] = pd.Series(z, index=e["date"].to_numpy()).groupby(level=0).last()
    return out


def build_daily_features(ev, trading_days):
    cal = pd.date_range(min(trading_days) - pd.Timedelta(days=60), max(trading_days) + pd.Timedelta(days=30))
    f = pd.DataFrame(index=cal)

    z = surprise_z(ev)
    for fam, name in [("CPI", "last_cpi_z"), ("NFP", "last_nfp_z"), ("UNEMP", "last_unemp_z"), ("PPI", "last_ppi_z")]:
        f[name] = z[fam].reindex(cal).ffill().shift(1)
    f["surprise_index"] = f[["last_cpi_z", "last_nfp_z", "last_unemp_z", "last_ppi_z"]].mean(axis=1)

    key = ev[ev["family"].notna()]
    for fam, name in [("CPI", "is_cpi_day"), ("NFP", "is_nfp_day"), ("PPI", "is_ppi_day"), ("FOMC", "is_fomc_day")]:
        f[name] = cal.isin(key.loc[key["family"] == fam, "date"].unique()).astype(int)
    f["n_key_events_today"] = key.groupby("date")["family"].nunique().reindex(cal).fillna(0).astype(int)
    key_dates = np.sort(key["date"].unique())
    pos = np.searchsorted(key_dates, cal.to_numpy())
    nxt = np.where(pos < len(key_dates), key_dates[np.minimum(pos, len(key_dates) - 1)], np.datetime64("NaT"))
    f["days_to_next_key"] = (pd.to_datetime(nxt) - cal).days.to_numpy()
    f["days_to_next_key"] = f["days_to_next_key"].fillna(30).clip(upper=30)

    sent_path = PROC / "sentiment_daily.csv"
    if sent_path.exists():
        s = pd.read_csv(sent_path, parse_dates=["date"]).set_index("date")["sentiment"].reindex(cal)
        f["sent_7d"] = s.rolling(7, min_periods=1).mean().shift(1)
        f["sent_chg"] = f["sent_7d"] - s.rolling(28, min_periods=1).mean().shift(1)
    else:
        f["sent_7d"] = np.nan
        f["sent_chg"] = np.nan

    f["dow"] = cal.dayofweek
    f = f.loc[pd.DatetimeIndex(sorted(trading_days))]
    f.index.name = "date"
    return f.reset_index()


def main():
    ev = load_events()
    log = pd.read_csv(PROC / "strategy_log.csv", parse_dates=["date"])
    daily = build_daily_features(ev, log["date"].unique())

    write_table(ev.drop(columns=["ts_et"]), "macro_events")
    write_table(daily, "daily_features")
    cols = ", ".join(f"f.{c}" for c in FEATS if c in daily.columns)
    panel = read_sql(f"""
        SELECT s.date, s.perm_id, s.fast, s.slow, s.event_filter, s.position, s.pnl, {cols}
        FROM strategy_log s JOIN daily_features f ON s.date = f.date
        ORDER BY s.date, s.perm_id""")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["is_flat_filter"] = (panel["event_filter"] == "flat_on_event_days").astype(int)
    panel.to_csv(PROC / "panel.csv", index=False)
    print(f"panel: {len(panel)} rows, {panel['date'].nunique()} days, {panel['perm_id'].nunique()} permutations")


if __name__ == "__main__":
    main()
