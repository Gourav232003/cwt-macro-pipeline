"""Shared paths and helpers."""
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CHARTS = REPORTS / "charts"

# event families used for features. SIGN: +1 means "actual above forecast" is hawkish/stronger-economy.
FAMILIES = [
    ("CPI", r"^CPI m/m"),
    ("NFP", r"Non-Farm"),
    ("UNEMP", r"Unemployment Rate"),
    ("PPI", r"^PPI m/m"),
    ("FOMC", r"FOMC Statement|Federal Funds Rate"),
]
SIGN = {"CPI": 1, "NFP": 1, "UNEMP": -1, "PPI": 1}


def load_cfg():
    return yaml.safe_load(open(ROOT / "config.yaml"))


def family_of(title):
    if isinstance(title, str):
        for name, pat in FAMILIES:
            if re.search(pat, title, flags=re.I):
                return name
    return None


def load_events():
    """macro_events.csv with ts_utc, ts_et and a naive New York calendar `date`."""
    df = pd.read_csv(PROC / "macro_events.csv")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["ts_et"] = df["ts_utc"].dt.tz_convert("America/New_York")
    df["date"] = df["ts_et"].dt.tz_localize(None).dt.normalize()
    df["family"] = df["title"].map(family_of)
    return df
