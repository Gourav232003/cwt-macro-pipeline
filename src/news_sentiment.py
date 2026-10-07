"""Score news with a transparent finance lexicon -> daily macro sentiment (-1 bearish .. +1 bullish).

Dates are conservative to avoid look-ahead: an article counts on its published date if it falls in
its search window, otherwise on the window END date (the day the whole window is known).

Usage: python -m src.news_sentiment
"""
import json
import re

import pandas as pd

from .common import RAW, PROC
from .db import write_table

POS = ["growth", "strong", "stronger", "beat", "beats", "exceeded", "rally", "rallies", "gain", "gains",
       "surge", "surges", "record high", "improve", "improved", "improving", "resilient", "optimism",
       "optimistic", "easing", "eases", "cooling", "cooled", "rate cut", "rate cuts", "soft landing",
       "upgrade", "upbeat", "rebound", "recovery", "expansion", "expands", "cut rates", "disinflation"]
NEG = ["recession", "slowdown", "slows", "slowing", "weak", "weaker", "miss", "misses", "missed", "fall",
       "falls", "drop", "drops", "plunge", "selloff", "sell-off", "slump", "decline", "declines", "fear",
       "fears", "worries", "worry", "rate hike", "rate hikes", "hawkish", "layoffs", "job losses",
       "tariff", "tariffs", "shutdown", "downgrade", "volatile", "volatility", "uncertainty", "crisis",
       "contraction", "stagflation", "higher for longer", "sticky inflation", "hotter than expected"]

_pos = re.compile(r"\b(" + "|".join(map(re.escape, POS)) + r")\b")
_neg = re.compile(r"\b(" + "|".join(map(re.escape, NEG)) + r")\b")


def score_text(text: str) -> float:
    t = (text or "").lower()
    p, n = len(_pos.findall(t)), len(_neg.findall(t))
    return (p - n) / max(1, p + n)


def main():
    path = RAW / "news.json"
    if not path.exists():
        raise SystemExit("No data/raw/news.json. Run: python -m src.news_tavily")
    df = pd.DataFrame(json.loads(path.read_text()))
    df["text"] = df["title"].fillna("") + ". " + df["content"].fillna("")
    df["sentiment"] = df["text"].map(score_text)

    pub = pd.to_datetime(df["published_date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    ws = pd.to_datetime(df["window_start"])
    we = pd.to_datetime(df["window_end"])
    ok = pub.notna() & (pub >= ws) & (pub <= we)
    df["date"] = we.where(~ok, pub)

    daily = (df.groupby("date")["sentiment"].agg(["mean", "count"])
             .rename(columns={"mean": "sentiment", "count": "n_articles"}).reset_index())
    PROC.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["text"]).to_csv(PROC / "news_scored.csv", index=False)
    daily.to_csv(PROC / "sentiment_daily.csv", index=False)
    write_table(df.drop(columns=["text"]), "news")
    write_table(daily, "sentiment_daily")
    print(f"{len(df)} articles scored; {len(daily)} days with sentiment. Mean sentiment {df['sentiment'].mean():+.3f}")


if __name__ == "__main__":
    main()
