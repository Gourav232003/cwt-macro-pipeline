"""Draft the 1-page macro outlook (reports/macro_outlook.md) from the data.

This is a DRAFT built from the latest releases, sentiment trend and Tavily headlines.
Read it, check the facts, and edit the outlook paragraph in your own words before submitting.

Usage: python -m src.macro_summary
"""
import json
from datetime import date
from urllib.parse import urlparse

import pandas as pd

from .common import PROC, RAW, REPORTS, load_events
from .features import surprise_z
from .news_sentiment import score_text

NAMES = {"CPI": "CPI m/m", "NFP": "Non-Farm Employment", "UNEMP": "Unemployment Rate", "PPI": "PPI m/m"}


def direction(a, f):
    if pd.isna(a) or pd.isna(f):
        return "n/a"
    return "above forecast" if a > f else "below forecast" if a < f else "in line"


def main():
    ev = load_events()
    lines = [f"# US macro outlook (draft, {date.today():%d %b %Y})", ""]

    lines += ["## Latest releases", "", "| Indicator | Date | Actual | Forecast | Previous | Result |", "|---|---|---|---|---|---|"]
    for fam, nm in NAMES.items():
        e = ev[(ev["family"] == fam) & ev["actualNumeric"].notna()].sort_values("ts_utc")
        if e.empty:
            continue
        r = e.iloc[-1]
        lines.append(f"| {nm} | {r['date']:%d %b} | {r['actual']} | {r['forecast']} | {r['previous']} | "
                     f"{direction(r['actualNumeric'], r['forecastNumeric'])} |")
    fomc = ev[(ev["family"] == "FOMC") & ev["title"].str.contains("Federal Funds Rate", na=False)].sort_values("ts_utc")
    if len(fomc):
        r = fomc.iloc[-1]
        lines.append(f"| Fed funds rate (last decision in data) | {r['date']:%d %b} | {r['actual']} | {r['forecast']} | {r['previous']} | {direction(r['actualNumeric'], r['forecastNumeric'])} |")

    z = {k: v.dropna() for k, v in surprise_z(ev).items()}
    hawk = [(k, v.iloc[-1]) for k, v in z.items() if len(v)]
    lines += ["", "## Surprise read", ""]
    if hawk:
        idx = sum(v for _, v in hawk) / len(hawk)
        tone = "hawkish / stronger-than-expected" if idx > 0.5 else "dovish / weaker-than-expected" if idx < -0.5 else "mixed"
        lines.append(f"Latest standardised surprises: " + ", ".join(f"{NAMES[k]} {v:+.1f}" for k, v in hawk) +
                     f". Overall tone: **{tone}** (positive = hawkish). Few releases per indicator, so treat z-scores as rough.")
    else:
        lines.append("Not enough releases yet to standardise surprises.")

    sent_path = PROC / "sentiment_daily.csv"
    if sent_path.exists():
        s = pd.read_csv(sent_path, parse_dates=["date"]).set_index("date")["sentiment"].sort_index()
        recent, prior = s.iloc[-28:].mean(), s.iloc[:-28].mean() if len(s) > 28 else float("nan")
        lines += ["", "## News sentiment", "",
                  f"Average news sentiment over the latest observations is {recent:+.2f}"
                  + (f" versus {prior:+.2f} earlier in the window, i.e. {'improving' if recent > prior else 'weakening'}." if pd.notna(prior) else ".")]

    lp = RAW / "news_latest.json"
    if lp.exists():
        news = pd.DataFrame(json.loads(lp.read_text())).drop_duplicates("url")
        if len(news):
            news = news.sort_values("score", ascending=False).head(6)
            lines += ["", "## Recent headlines (Tavily)", ""]
            for _, r in news.iterrows():
                lines.append(f"- {r['title']} ({urlparse(r['url']).netloc}), tone {score_text(r['title'] + '. ' + (r['content'] or '')):+.1f}")

    lines += ["", "## Outlook (edit in your own words)", "",
              "- Inflation: <summarise CPI/PPI trend vs forecasts>",
              "- Labour market: <summarise NFP, unemployment, wages>",
              "- Fed / rates: <summarise latest FOMC decision and market pricing>",
              "- Risks and what to watch next: <next CPI / NFP / FOMC dates>"]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "macro_outlook.md").write_text("\n".join(lines) + "\n")
    print("Wrote reports/macro_outlook.md")


if __name__ == "__main__":
    main()
