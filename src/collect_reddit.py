"""Collect Reddit comments for the US vs Nigerian English dataset.

Uses the Arctic Shift archive API (https://arctic-shift.photon-reddit.com), a
public Reddit archive that needs no key. The official Reddit API was the plan in
the proposal, but self-service keys were withdrawn in late 2025 and research
access now goes through manual review, so it could not be used in time.

For each subreddit in config/reddit_groups.json the script pages through
comments in time order (100 per request), politely (default 1 request/second,
backing off on HTTP 429), and appends them to data/reddit/raw/<group>_<sub>.jsonl.
It is resumable: re-running continues from the last comment already saved.

If the archive refuses a query (it answers HTTP 422 when a query is too
expensive, which happens on large subreddits over a long time range), the
script narrows the time window and carries on, widening it again once windows
come back empty. A window that still fails at one hour is skipped and logged
as a gap rather than stopping the run, and a problem with one subreddit does
not stop the others.

Usage:
  python src/collect_reddit.py                      # everything in the config
  python src/collect_reddit.py --max-per-sub 2000   # quick trial
  python src/collect_reddit.py --only Nigeria lagos  # just these subreddits
"""
import argparse
import datetime as dt
import http.client
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
FIELDS = "id,author,body,created_utc,subreddit,link_id,parent_id,score,author_flair_text"
UA = "COMP8240-student-research/1.0 (Macquarie University; academic replication project)"
MIN_SPAN = 3600          # narrowest time window tried before a window is skipped
MAX_CONSECUTIVE_SKIPS = 5


class QueryFailed(Exception):
    """The archive kept refusing one query (HTTP 422, 5xx or network errors)."""


def to_epoch(s):
    return int(dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())


def day(t):
    return f"{dt.datetime.fromtimestamp(t, dt.timezone.utc):%Y-%m-%d %H:%M}"


def _error_text(e):
    try:
        return e.read().decode("utf-8", "replace").strip()[:200]
    except Exception:
        return ""


def fetch_page(subreddit, after, before, limit=100, retries=6, retries_422=2):
    params = {"subreddit": subreddit, "after": after, "before": before, "limit": limit,
              "sort": "asc", "fields": FIELDS}
    url = API + "?" + urllib.parse.urlencode(params)
    delay, last, n422 = 5, "", 0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.loads(r.read().decode("utf-8"))
            if isinstance(payload, list):
                return payload
            return payload.get("data", []) or []
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} {_error_text(e)}".strip()
            if e.code == 429:
                reset = e.headers.get("X-RateLimit-Reset")
                wait = int(float(reset)) + 1 if reset and reset.replace(".", "").isdigit() else delay
                print(f"    rate limited; waiting {wait}s", flush=True)
                time.sleep(min(wait, 300))
            elif e.code == 422 or 500 <= e.code < 600:
                # 422 is how the archive reports a query it could not finish
                # (for example "Query timed out"); a narrower window usually works.
                print(f"    r/{subreddit}: {last}; retry {attempt + 1}/{retries}", flush=True)
                n422 += e.code == 422
                if n422 >= retries_422:
                    break
                time.sleep(delay)
            else:
                raise
        except (OSError, http.client.HTTPException, ValueError) as e:
            # OSError covers URLError, timeouts and connection resets;
            # HTTPException covers truncated responses; ValueError bad JSON.
            last = f"{type(e).__name__}: {e}"
            print(f"    network error ({last}); retry {attempt + 1}/{retries}", flush=True)
            time.sleep(delay)
        delay = min(delay * 2, 120)
    raise QueryFailed(f"{day(after)} to {day(before)}: {last}")


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


def collect_sub(group, sub, after, before, cap, outdir, pause, fetch=fetch_page, gaps=None):
    """Page through r/sub in time order until `cap` comments are saved.

    Queries cover [cursor, cursor + span]. The span starts as the whole
    remaining range; when the archive refuses a query it is cut to an eighth
    (down to MIN_SPAN), and each empty window doubles it again, up to half the
    narrowest span that was refused. A window that
    fails even at MIN_SPAN is skipped and recorded in `gaps`.
    """
    path = os.path.join(outdir, f"{group}_{sub}.jsonl")
    resume, seen = last_saved(path)
    # one second back, so comments sharing the last saved second are not lost
    start = max(after, int(resume) - 1) if resume else after
    n = len(seen)
    gaps = [] if gaps is None else gaps
    print(f"[{group}] r/{sub}: {n:,} already saved; collecting from {day(start)}", flush=True)
    full = before - start
    span, ceiling, skips = full, full, 0
    with open(path, "a", encoding="utf-8") as out:
        cursor = start
        while n < cap and cursor < before:
            end = min(before, cursor + span)
            try:
                page = fetch(sub, cursor, end)
            except QueryFailed as e:
                if span > MIN_SPAN:
                    ceiling = max(MIN_SPAN, min(ceiling, span // 2))  # never widen back to this
                    span = max(MIN_SPAN, span // 8)
                    print(f"    r/{sub}: query refused; narrowing the window to "
                          f"{span / 3600:,.0f} h", flush=True)
                    continue
                skips += 1
                gaps.append({"from": day(cursor), "to": day(end), "error": str(e)})
                print(f"    r/{sub}: skipping {day(cursor)} to {day(end)} ({e})", flush=True)
                if skips >= MAX_CONSECUTIVE_SKIPS:
                    raise RuntimeError(f"r/{sub}: {skips} windows in a row failed; "
                                       f"stopping this subreddit (rerun later to resume)")
                cursor = end - 1
                continue
            skips = 0
            if not page:
                if end >= before:
                    break
                # nothing in this window: move on (one second of overlap, as
                # above) and try a wider window next time
                cursor = end - 1
                span = min(ceiling, span * 2)
                continue
            new = [c for c in page if c.get("id") not in seen]
            for c in new:
                c["group"] = group
                out.write(json.dumps(c, ensure_ascii=False) + "\n")
                seen.add(c["id"])
                n += 1
                if n >= cap:
                    break
            out.flush()
            newest = max(int(c["created_utc"]) for c in page)
            # Re-request from one second before the newest comment, so comments
            # sharing that second are not skipped whether `after` is inclusive or
            # exclusive; already-saved ids are filtered out above.
            cursor = max(cursor, newest - 1) if new else max(cursor + 1, newest)
            if n % 2000 < len(new):
                print(f"    r/{sub}: {n:,} comments (up to {day(cursor)[:10]})", flush=True)
            time.sleep(pause)
    print(f"[{group}] r/{sub}: done, {n:,} comments"
          + (f", {len(gaps)} skipped window(s)" if gaps else ""), flush=True)
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
    totals, gaps, failed = {}, {}, []
    for group, subs in cfg["groups"].items():
        for sub in subs:
            if args.only and sub not in args.only:
                continue
            cap = args.max_per_sub or caps.get(group, cfg["max_comments_per_subreddit"])
            g = gaps.setdefault(f"{group}/{sub}", [])
            try:
                totals[f"{group}/{sub}"] = collect_sub(group, sub, after, before, cap, args.out,
                                                       args.pause, gaps=g)
            except Exception:
                # keep going with the other subreddits; this one resumes on rerun
                traceback.print_exc()
                print(f"[{group}] r/{sub}: stopped early, see the error above", flush=True)
                failed.append(f"{group}/{sub}")
    # skipped windows are appended to a log so the data description can report them
    gaps = {k: v for k, v in gaps.items() if v}
    if gaps:
        with open(os.path.join(args.out, "skipped_windows.jsonl"), "a", encoding="utf-8") as fh:
            for k, v in gaps.items():
                for w in v:
                    fh.write(json.dumps({"subreddit": k, **w}) + "\n")
    print(json.dumps({"comments": totals, "skipped_windows": {k: len(v) for k, v in gaps.items()},
                      "stopped_early": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
