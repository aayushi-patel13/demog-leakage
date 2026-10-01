"""Collect Reddit comments for the US vs Nigerian English dataset.

Uses the Arctic Shift archive API (https://arctic-shift.photon-reddit.com), a
public Reddit archive that needs no key. The official Reddit API was the plan in
the proposal, but self-service keys were withdrawn in late 2025 and research
access now goes through manual review, so it could not be used in time.

For each subreddit in config/reddit_groups.json the script pages through
comments in time order (100 per request), politely (default 1 request/second,
backing off on HTTP 429), and appends them to data/reddit/raw/<group>_<sub>.jsonl.
It is resumable: re-running continues from the last comment already saved.

Usage:
  python src/collect_reddit.py                      # everything in the config
  python src/collect_reddit.py --max-per-sub 2000   # quick trial
  python src/collect_reddit.py --only Nigeria lagos  # just these subreddits
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
FIELDS = "id,author,body,created_utc,subreddit,link_id,parent_id,score,author_flair_text"
UA = "COMP8240-student-research/1.0 (Macquarie University; academic replication project)"


def to_epoch(s):
    return int(dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def fetch_page(subreddit, after, before, limit=100, retries=6):
    params = {"subreddit": subreddit, "after": after, "before": before, "limit": limit,
              "sort": "asc", "fields": FIELDS}
    url = API + "?" + urllib.parse.urlencode(params)
    delay = 5
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.loads(r.read().decode("utf-8"))
            if isinstance(payload, list):
                return payload
            return payload.get("data", []) or []
        except urllib.error.HTTPError as e:
            if e.code == 429:
                reset = e.headers.get("X-RateLimit-Reset")
                wait = int(float(reset)) + 1 if reset and reset.replace(".", "").isdigit() else delay
                print(f"    rate limited; waiting {wait}s", flush=True)
                time.sleep(min(wait, 300))
            elif 500 <= e.code < 600:
                time.sleep(delay)
            else:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"    network error ({e}); retry {attempt + 1}/{retries}", flush=True)
            time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"giving up on {url}")


def last_saved(path):
    """Resume point: created_utc of the last line, and the ids already saved."""
    if not os.path.exists(path):
        return None, set()
    last, ids = None, set()
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            try:
                c = json.loads(ln)
            except json.JSONDecodeError:
                continue
            ids.add(c["id"])
            last = c["created_utc"]
    return last, ids


def collect_sub(group, sub, after, before, cap, outdir, pause, fetch=fetch_page):
    path = os.path.join(outdir, f"{group}_{sub}.jsonl")
    resume, seen = last_saved(path)
    start = max(after, int(resume)) if resume else after
    n = len(seen)
    print(f"[{group}] r/{sub}: {n:,} already saved; collecting from "
          f"{dt.datetime.fromtimestamp(start, dt.timezone.utc):%Y-%m-%d}", flush=True)
    with open(path, "a", encoding="utf-8") as out:
        cursor = start
        while n < cap:
            page = fetch(sub, cursor, before)
            new = [c for c in page if c.get("id") not in seen]
            if not page:
                break
            for c in new:
                c["group"] = group
                out.write(json.dumps(c, ensure_ascii=False) + "\n")
                seen.add(c["id"])
                n += 1
                if n >= cap:
                    break
            newest = max(int(c["created_utc"]) for c in page)
            # Re-request from one second before the newest comment, so comments
            # sharing that second are not skipped whether `after` is inclusive or
            # exclusive; already-saved ids are filtered out above.
            cursor = max(cursor, newest - 1) if new else max(cursor + 1, newest)
            if n % 2000 < len(new):
                print(f"    r/{sub}: {n:,} comments (up to "
                      f"{dt.datetime.fromtimestamp(cursor, dt.timezone.utc):%Y-%m-%d})", flush=True)
            time.sleep(pause)
    print(f"[{group}] r/{sub}: done, {n:,} comments", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/reddit_groups.json")
    ap.add_argument("--out", default="data/reddit/raw")
    ap.add_argument("--max-per-sub", type=int, default=None)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--only", nargs="*", help="collect only these subreddits")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    after, before = to_epoch(cfg["window"]["after"]), to_epoch(cfg["window"]["before"])
    caps = cfg.get("max_comments_per_subreddit_by_group", {})
    os.makedirs(args.out, exist_ok=True)
    totals = {}
    for group, subs in cfg["groups"].items():
        for sub in subs:
            if args.only and sub not in args.only:
                continue
            cap = args.max_per_sub or caps.get(group, cfg["max_comments_per_subreddit"])
            totals[f"{group}/{sub}"] = collect_sub(group, sub, after, before, cap, args.out, args.pause)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    sys.exit(main())
