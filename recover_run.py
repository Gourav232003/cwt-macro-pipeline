"""One-off: save the dataset of an Apify run that already finished (no re-run, no extra cost).

Usage: python recover_run.py RUN_ID 2026-04-10 2026-05-09
"""
import json, os, sys
from pathlib import Path
from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()
run_id, start, end = sys.argv[1:4]
client = ApifyClient(os.environ["APIFY_TOKEN"])
run = client.run(run_id).get()
ds_id = run["defaultDatasetId"] if isinstance(run, dict) else run.default_dataset_id
items = list(client.dataset(ds_id).iterate_items())
out = Path("data/raw") / f"ff_{start}_{end}.json"
out.write_text(json.dumps(items, indent=2))
print(f"saved {len(items)} rows -> {out}")
