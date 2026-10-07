"""Pull the ForexFactory economic calendar through Apify, several chunks in parallel.

Usage:
    python -m src.ingest_apify --test        # one-week sanity check
    python -m src.ingest_apify               # full window from config.yaml
    python -m src.ingest_apify --workers 8   # more parallel runs
    python -m src.ingest_apify --force       # ignore cached chunks / empty markers
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml
from apify_client import ApifyClient
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def make_chunks(start: date, end: date, n_days: int):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=n_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def _field(obj, snake: str, camel: str):
    """apify-client 2.x returns a Run object; older versions return a dict."""
    if isinstance(obj, dict):
        return obj[camel]
    return getattr(obj, snake)


def run_chunk(client, cfg, s: date, e: date, attempt: int, max_attempts: int):
    a = cfg["apify"]
    run_input = {
        "range": "custom",
        "startDate": s.isoformat(),
        "endDate": e.isoformat(),
        "currencies": a["currencies"],
        "impactLevels": a["impact_levels"],
        "maxEvents": a["max_events"],
    }
    # last attempt: residential proxy (costs more, helps with ForexFactory blocking)
    if attempt == max_attempts:
        run_input["proxyConfiguration"] = {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],
        }
    actor = client.actor(a["actor_id"])
    try:
        run = actor.call(run_input=run_input, logger=None)  # no log streaming from threads
    except TypeError:  # older client without the logger argument
        run = actor.call(run_input=run_input)
    dataset_id = _field(run, "default_dataset_id", "defaultDatasetId")
    run_id = _field(run, "id", "id")
    items = list(client.dataset(dataset_id).iterate_items())
    return run_input, run_id, items


def is_blocked(items):
    return any(i.get("blocked") for i in items) or len(items) == 0


def process_chunk(client, cfg, s: date, e: date, force: bool):
    """Returns (status, manifest_entry_or_None). status: cached | saved | empty | failed"""
    out = RAW / f"ff_{s}_{e}.json"
    empty_marker = RAW / f"ff_{s}_{e}.empty"
    if not force and (out.exists() or empty_marker.exists()):
        return "cached", None

    max_attempts = cfg["apify"].get("max_retries", 2)
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            run_input, run_id, items = run_chunk(client, cfg, s, e, attempt, max_attempts)
        except Exception as exc:  # network blips etc.
            last_err = exc
            log(f"[retry ] {s} -> {e} attempt {attempt} error: {type(exc).__name__}")
            continue
        if not is_blocked(items):
            out.write_text(json.dumps(items, indent=2))
            if empty_marker.exists():
                empty_marker.unlink()
            entry = {"start": s.isoformat(), "end": e.isoformat(), "run_id": run_id,
                     "rows": len(items), "run_input": run_input}
            return "saved", entry
        log(f"[retry ] {s} -> {e} attempt {attempt}: 0 rows or blocked")

    if last_err is None:
        # never blocked-with-error, just 0 rows every time: treat as a genuinely quiet period
        empty_marker.write_text("no events returned after retries")
        return "empty", None
    return "failed", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="pull only the last 7 days")
    parser.add_argument("--force", action="store_true", help="re-pull cached chunks")
    parser.add_argument("--workers", type=int, default=6, help="parallel Apify runs")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        raise SystemExit("APIFY_TOKEN missing. Copy .env.example to .env and fill it in.")

    cfg = load_config()
    a = cfg["apify"]
    client = ApifyClient(token)

    end = date.fromisoformat(a["end_date"]) if a["end_date"] else date.today()
    days = 7 if args.test else a["window_days"]
    start = end - timedelta(days=days)
    chunks = list(make_chunks(start, end, a["chunk_days"]))
    log(f"{len(chunks)} chunks, {args.workers} in parallel ({start} -> {end})")

    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"chunks": []}
    manifest.update({"actor_id": a["actor_id"], "last_pull_utc": datetime.now(timezone.utc).isoformat(),
                     "window": [start.isoformat(), end.isoformat()]})

    counts = {"cached": 0, "saved": 0, "empty": 0, "failed": 0}
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_chunk, client, cfg, s, e, args.force): (s, e) for s, e in chunks}
        for fut in as_completed(futures):
            s, e = futures[fut]
            try:
                status, entry = fut.result()
            except Exception as exc:
                status, entry = "failed", None
                log(f"[error ] {s} -> {e}: {exc}")
            counts[status] += 1
            if status == "failed":
                failed.append(f"{s}..{e}")
            if entry:
                manifest["chunks"].append(entry)
                log(f"[saved ] {s} -> {e}: {entry['rows']} rows (run {entry['run_id']})")
            elif status in ("empty", "cached"):
                log(f"[{status:6}] {s} -> {e}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"\nDone: {counts}")
    if failed:
        log(f"Failed chunks (rerun the same command to retry only these): {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
