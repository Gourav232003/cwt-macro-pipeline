"""Charts: macro sentiment over time, surprise heatmap, equity curves + drawdown.

Usage: python -m src.viz
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import CHARTS, PROC, load_events
from .features import surprise_z

COLORS = {"CPI": "tab:red", "NFP": "tab:green", "FOMC": "tab:purple", "PPI": "tab:orange", "UNEMP": "tab:blue"}


def sentiment_chart(ev):
    s = pd.read_csv(PROC / "sentiment_daily.csv", parse_dates=["date"]).set_index("date")["sentiment"].sort_index()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(s.index, s.values, s=10, alpha=0.35, color="gray", label="daily score")
    ax.plot(s.rolling(4, min_periods=1).mean(), color="black", lw=2, label="4-obs rolling mean")
    ax.axhline(0, color="gray", lw=0.8)
    drawn = set()
    for fam in ["CPI", "NFP", "FOMC"]:
        for d in ev.loc[ev["family"] == fam, "date"].unique():
            ax.axvline(d, color=COLORS[fam], alpha=0.35, lw=1, label=fam if fam not in drawn else None)
            drawn.add(fam)
    ax.set_title("Macro news sentiment over time (vertical lines = CPI / NFP / FOMC releases)")
    ax.set_ylabel("sentiment (-1 bearish, +1 bullish)")
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "sentiment_timeline.png", dpi=150)
    plt.close(fig)


def surprise_heatmap(ev):
    z = surprise_z(ev)
    rows = {fam: s.dropna() for fam, s in z.items()}
    dates = sorted({d for s in rows.values() for d in s.index})
    if not dates:
        return
    mat = np.full((len(rows), len(dates)), np.nan)
    for i, (fam, s) in enumerate(rows.items()):
        for d, v in s.items():
            mat[i, dates.index(d)] = v
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows.keys())
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([pd.Timestamp(d).strftime("%b %d") for d in dates], rotation=60, fontsize=7)
    ax.set_title("Standardised macro surprise (red = hawkish / stronger than forecast, blue = dovish / weaker)")
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(CHARTS / "surprise_heatmap.png", dpi=150)
    plt.close(fig)


def equity_chart():
    r = pd.read_csv(PROC / "daily_returns.csv", parse_dates=["date"]).set_index("date")
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for c in eq.columns:
        a1.plot(eq[c], label=c, lw=2 if c == "model_topk" else 1.3)
        a2.plot(dd[c], lw=1.2)
    a1.set_title("Out-of-sample equity curves (walk-forward)")
    a1.set_ylabel("growth of 1")
    a1.legend()
    a2.set_ylabel("drawdown")
    fig.tight_layout()
    fig.savefig(CHARTS / "equity_curves.png", dpi=150)
    plt.close(fig)


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    ev = load_events()
    if (PROC / "sentiment_daily.csv").exists():
        sentiment_chart(ev)
    surprise_heatmap(ev)
    if (PROC / "daily_returns.csv").exists():
        equity_chart()
    print(f"Charts saved in {CHARTS}")


if __name__ == "__main__":
    main()
