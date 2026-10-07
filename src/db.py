"""Tiny SQLite helper: all datasets are joined in data/macro.db."""
import sqlite3

import pandas as pd

from .common import ROOT

DB_PATH = ROOT / "data" / "macro.db"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            fmt = "%Y-%m-%d" if col == "date" else "%Y-%m-%d %H:%M:%S"
            if getattr(df[col].dt, "tz", None) is not None:
                df[col] = df[col].dt.tz_convert("UTC")
            df[col] = df[col].dt.strftime(fmt)
    return df


def write_table(df: pd.DataFrame, name: str):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        _prep(df).to_sql(name, con, if_exists="replace", index=False)


def read_sql(query: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(query, con)
