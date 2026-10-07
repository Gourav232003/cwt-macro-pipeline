"""Fetch macro news with Tavily: weekly history windows + a latest-news pull for the outlook.

Usage: python -m src.news_tavily
Needs TAVILY_API_KEY in .env. Re-runs skip (window, query) pairs already stored.
"""
import json
import os
from datetime import date, timedelta

from dotenv import load_dotenv
from tavily import TavilyClient

from .common import ROOT, RAW, load_cfg


def main():
    load_dotenv(ROOT / ".env")
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise SystemExit("TAVILY_API_KEY missing in .env")
    t = load_cfg()["tavily"]
    client = TavilyClient(api_key=key)
    RAW.mkdir(parents=True, exist_ok=True)

    path = RAW / "news.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    seen_urls = {r["url"] for r in rows}
    done_path = RAW / "news_done.json"
    done = {tuple(x) for x in json.loads(done_path.read_text())} if done_path.exists() else set()

    end = date.today()
    cur = end - timedelta(days=t["window_days"])
    while cur < end:
        w_end = min(cur + timedelta(days=t["step_days"] - 1), end)
        for q in t["queries"]:
            if (cur.isoformat(), q) in done:
                continue
            try:
                res = client.search(query=q, topic=t["topic"], search_depth=t["search_depth"],
                                    max_results=t["max_results"],
                                    start_date=cur.isoformat(), end_date=w_end.isoformat())
            except Exception as exc:
                print(f"[skip] {cur}..{w_end} '{q[:25]}': {type(exc).__name__}: {str(exc)[:100]}")
                continue
            done.add((cur.isoformat(), q))
            for r in res.get("results", []):
                if r["url"] in seen_urls:
                    continue
                seen_urls.add(r["url"])
                rows.append({"query": q, "window_start": cur.isoformat(), "window_end": w_end.isoformat(),
                             "title": r.get("title"), "url": r["url"], "content": r.get("content"),
                             "published_date": r.get("published_date"), "score": r.get("score")})
        print(f"window {cur}..{w_end}: {len(rows)} articles so far")
        cur = w_end + timedelta(days=1)
        path.write_text(json.dumps(rows, indent=1))
        (RAW / "news_done.json").write_text(json.dumps(sorted(done)))

    # latest news (last 14 days) for the 1-page outlook
    latest = []
    for q in t["queries"]:
        try:
            res = client.search(query=q, topic="news", days=14, max_results=8, search_depth=t["search_depth"])
        except Exception as exc:
            print(f"[latest skip] {type(exc).__name__}: {str(exc)[:100]}")
            continue
        for r in res.get("results", []):
            latest.append({"query": q, "title": r.get("title"), "url": r["url"], "content": r.get("content"),
                           "published_date": r.get("published_date"), "score": r.get("score")})
    (RAW / "news_latest.json").write_text(json.dumps(latest, indent=1))
    print(f"Saved {len(rows)} historical and {len(latest)} latest articles")


if __name__ == "__main__":
    main()
