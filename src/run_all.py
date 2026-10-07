"""Run the whole pipeline after the Apify pull.

Usage:
    python -m src.run_all              # everything incl. Tavily news
    python -m src.run_all --skip-news  # reuse data/raw/news.json
"""
import subprocess
import sys

STEPS = ["src.clean_events", "src.news_tavily", "src.news_sentiment", "src.strategy_log",
         "src.features", "src.model", "src.evaluate", "src.viz", "src.macro_summary"]


def main():
    skip_news = "--skip-news" in sys.argv
    for step in STEPS:
        if skip_news and step == "src.news_tavily":
            continue
        print(f"\n=== {step} ===", flush=True)
        if subprocess.run([sys.executable, "-m", step]).returncode != 0:
            raise SystemExit(f"Step failed: {step}")
    print("\nAll done. See reports/ and data/macro.db")


if __name__ == "__main__":
    main()
