"""Clean raw ForexFactory rows into one macro-events table (UTC + ET timestamps).

ForexFactory returns times in an unlabelled "display zone" that can differ from run to run
(it depends on the IP the actor runs from). So the zone is inferred PER RAW FILE: we pick the
US timezone that makes known release times land where they should in New York time
(CPI/PPI/jobs 08:30 ET, FOMC statement 14:00 ET, press conference 14:30 ET).

Usage:
    python -m src.clean_events
"""
from pathlib import Path
import json
import re

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "macro_events.csv"

CANDIDATE_TZ = ["America/New_York", "America/Chicago", "America/Denver",
                "America/Los_Angeles", "UTC"]

# (title regex, scheduled release time in New York time)
EXPECTED_ET = [
    (r"CPI|PPI|Non-Farm|Unemployment Rate|Average Hourly|Retail Sales|GDP|PCE", "08:30"),
    (r"FOMC Statement|Federal Funds Rate|FOMC Meeting Minutes|FOMC Economic Projections", "14:00"),
    (r"FOMC Press Conference", "14:30"),
]


def expected_et(title):
    for pat, hhmm in EXPECTED_ET:
        if isinstance(title, str) and re.search(pat, title, flags=re.I):
            return hhmm
    return None


def infer_tz(df_file, default_tz):
    """Return (tz, match_share, n_anchor_rows) for one raw file."""
    ts = pd.to_datetime(df_file["timestamp"], errors="coerce")
    exp = df_file["title"].map(expected_et)
    mask = exp.notna() & ts.notna()
    if mask.sum() == 0:
        return default_tz, None, 0
    best_tz, best_share = default_tz, -1.0
    for tz in CANDIDATE_TZ:
        loc = ts[mask].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        et = loc.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
        share = float((et == exp[mask]).mean())
        if share > best_share + 1e-9:
            best_tz, best_share = tz, share
    if best_share < 0.6:  # no zone fits well: do not trust inference
        return default_tz, best_share, int(mask.sum())
    return best_tz, best_share, int(mask.sum())


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    default_tz = cfg["calendar"]["source_tz"]

    frames, report = [], []
    for f in sorted(RAW.glob("ff_*.json")):
        rows = json.loads(f.read_text())
        if not rows:
            continue
        d = pd.DataFrame(rows)
        if "blocked" in d.columns:
            d = d[~d["blocked"].fillna(False).astype(bool)]
        if d.empty:
            continue
        tz, share, n = infer_tz(d, default_tz)
        ts = pd.to_datetime(d["timestamp"], errors="coerce")
        d["source_file"] = f.name
        d["inferred_tz"] = tz
        # convert to UTC per file BEFORE concatenating (mixed tz columns turn into object dtype)
        d["ts_utc"] = ts.dt.tz_localize(
            tz, ambiguous="NaT", nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
        frames.append(d)
        report.append((f.name, len(d), tz, "n/a" if share is None else f"{share:.0%}", n))

    if not frames:
        raise SystemExit("No raw files found. Run: python -m src.ingest_apify")

    print("Timezone inferred per raw file:")
    print(pd.DataFrame(report, columns=["file", "rows", "tz", "anchor_match", "anchor_rows"]).to_string(index=False), "\n")

    df = pd.concat(frames, ignore_index=True)
    df["ts_et"] = df["ts_utc"].dt.tz_convert("America/New_York")

    # surprise features
    for c in ["actualNumeric", "forecastNumeric", "previousNumeric", "revisedNumeric"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["surprise"] = df["actualNumeric"] - df["forecastNumeric"]
    df["surprise_pct"] = df["surprise"] / df["forecastNumeric"].abs()

    # keep target event families
    keep = df["title"].str.contains(cfg["events"]["keep_pattern"], case=False, regex=True, na=False)
    df = df[keep].sort_values("ts_utc").drop_duplicates(subset="eventId")

    # final timezone check across all kept rows
    exp = df["title"].map(expected_et)
    m = exp.notna()
    share = (df.loc[m, "ts_et"].dt.strftime("%H:%M") == exp[m]).mean() if m.any() else float("nan")
    print(f"Scheduled-time check after per-file correction: {share:.0%} of {int(m.sum())} anchor rows")
    if share < 0.9:
        print("WARNING: some files still look off. Look at the per-file table above.")

    # coverage by month (spot gaps)
    core = ["CPI m/m", "Core CPI m/m", "PPI m/m", "Non-Farm Employment Change",
            "Unemployment Rate", "FOMC Statement", "Federal Funds Rate"]
    cov = pd.crosstab(df["ts_et"].dt.strftime("%Y-%m"), df["title"])
    cov = cov[[c for c in core if c in cov.columns]]
    print("\nCoverage (events per month):")
    print(cov.to_string(), "\n")

    # which rows lack actual/forecast
    miss = df[df["actualNumeric"].isna() | df["forecastNumeric"].isna()]
    print(f"Rows kept: {len(df)}")
    print(f"Missing actual: {df['actualNumeric'].isna().sum()}  | Missing forecast: {df['forecastNumeric'].isna().sum()}")
    if len(miss):
        print("Titles with missing values:")
        print(miss["title"].value_counts().to_string())
    print(f"Date range (UTC): {df['ts_utc'].min()} -> {df['ts_utc'].max()}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
