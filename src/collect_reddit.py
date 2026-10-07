"""Collect Reddit comments for the US vs Nigerian English dataset.

Uses the Arctic Shift archive API (https://arctic-shift.photon-reddit.com), a
public Reddit archive that needs no key. The official Reddit API was the plan in
the proposal, but self-service keys were withdrawn in late 2025 and research
access now goes through manual review, so it could not be used in time.

Sampling (config/reddit_groups.json): a group with a daily quota (the US
subreddits, some of which post thousands of comments a day) gets up to that
many comments per subreddit per day, taken in time order from a random start
time in each day, so the sample covers all 18 months and all hours of the day
rather than the first few days of the window. A group without a quota (the
Nigerian subreddits, which are far smaller) is collected in full, in time
order, up to its cap. Requests are polite (100 comments each, default 1
request/second, backing off on HTTP 429) and comments are appended to
data/reddit/raw/<group>_<sub>.jsonl. It is resumable: re-running continues
from the last day (or comment) already saved.

If the archive refuses a query (it answers HTTP 422 when a query is too
expensive, which happens on large subreddits over a long time range), the
script narrows the time window and carries on, widening it again once windows
come back empty. A window that still fails at one hour is skipped and logged
as a gap rather than stopping the run, and a problem with one subreddit does
not stop the others.

Usage:
  python src/collect_reddit.py                      # everything in the config
  python src/collect_reddit.py --max-per-sub 2000 --no-quota   # quick trial
  python src/collect_reddit.py --only Nigeria lagos  # just these subreddits
"""
import argparse
import datetime as dt
from collections import Counter
import http.client
import json
import os
import random
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
FIELDS = "id,author,body,created_utc,subreddit,link_id,parent_id,score,author_flair_text"
UA = "COMP8240-student-research/1.0 (Macquarie University; academic replication project)"
DAY = 86400
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
    """Ids already saved, the time of the last saved line, and comments per day."""
    ids, per_day, last_t = set(), Counter(), None
    if not os.path.exists(path):
        return ids, last_t, per_day
    with open(path, "rb+") as fh:  # a run killed mid-write leaves half a line: end it
        fh.seek(0, 2)
        if fh.tell():
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            try:
                c = json.loads(ln)
            except json.JSONDecodeError:
                continue
            ids.add(c["id"])
            last_t = int(c["created_utc"])
            per_day[last_t // DAY] += 1
    return ids, last_t, per_day


def page_through(sub, lo, hi, want, seen, out, group, fetch, pause, state, gaps, progress=None):
    """Save up to `want` unseen comments created in [lo, hi), in time order.

    Queries cover [cursor, cursor + span]. When the archive refuses a query
    the span is cut to an eighth (down to MIN_SPAN) and the widest span ever
    allowed again becomes half the refused one; each empty window doubles the
    span. A window that fails even at MIN_SPAN is skipped and added to `gaps`.
    """
    # start one second early: whether the API's `after` is inclusive or
    # exclusive, nothing at second `lo` is lost (repeats are filtered by id)
    cursor, got = lo - 1, 0
    span = min(state["ceiling"], hi - cursor)
    while got < want and cursor < hi:
        end = min(hi, cursor + span)
        try:
            page = fetch(sub, cursor, end)
        except QueryFailed as e:
            if span > MIN_SPAN:
                state["ceiling"] = max(MIN_SPAN, min(state["ceiling"], span // 2))
                span = max(MIN_SPAN, span // 8)
                print(f"    r/{sub}: query refused; narrowing the window to "
                      f"{span / 3600:,.1f} h", flush=True)
                continue
            state["skips"] += 1
            gaps.append({"from": day(cursor), "to": day(end), "error": str(e)})
            print(f"    r/{sub}: skipping {day(cursor)} to {day(end)} ({e})", flush=True)
            if state["skips"] >= MAX_CONSECUTIVE_SKIPS:
                raise RuntimeError(f"r/{sub}: {state['skips']} windows in a row failed; "
                                   f"stopping this subreddit (rerun later to resume)")
            cursor = end - 1
            continue
        state["skips"] = 0
        time.sleep(pause)
        if not page:
            if end >= hi:
                break
            cursor = end - 1          # nothing here: move on, try a wider window
            span = min(state["ceiling"], span * 2)
            continue
        new = [c for c in page if c.get("id") not in seen]
        for c in new[:want - got]:
            c["group"] = group
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
            seen.add(c["id"])
            got += 1
        out.flush()
        newest = max(int(c["created_utc"]) for c in page)
        if progress:
            progress(got, newest)
        # Re-request from one second before the newest comment, so comments
        # sharing that second are not skipped; saved ids are filtered above.
        cursor = max(cursor, newest - 1) if new else max(cursor + 1, newest)
    return got


def collect_sub(group, sub, after, before, cap, outdir, pause, fetch=fetch_page, gaps=None,
                quota=None, seed=8240):
    """Collect up to `cap` comments from r/sub created in [after, before).

    With a daily `quota`, every day contributes up to `quota` comments, taken
    in time order from a start time drawn at random for that day (seeded, so a
    resumed run draws the same ones) and continuing from the start of the day
    if the quota is not yet met. Busy subreddits then contribute the same
    number of comments on every day and at every time of day; quiet days
    contribute all they have. Without a quota, the first `cap` comments are
    taken in time order (all of them, for a small subreddit).

    Resumable: with a quota, days before the last saved one count as done and
    the last day is topped up; without, collection continues from the last
    saved comment.
    """
    spread = quota is not None
    path = os.path.join(outdir, f"{group}_{sub}.jsonl")
    seen, last_t, saved = last_saved(path)
    last_day = None if last_t is None else last_t // DAY
    n = len(seen)
    gaps = [] if gaps is None else gaps
    days = list(range(after // DAY, (before - 1) // DAY + 1))
    rng = random.Random(f"{seed}/{sub}")
    offsets = [rng.randrange(DAY) for _ in days]   # drawn up front: same on resume
    if spread:
        slices = [(max(after, d * DAY), min(before, (d + 1) * DAY), off, d)
                  for d, off in zip(days, offsets)]
    else:
        quota = cap
        start = after if last_t is None else max(after, last_t)
        slices = [(start, before, 0, None)]
    print(f"[{group}] r/{sub}: {n:,} already saved; "
          + (f"up to {quota:,} per day" if spread else "all comments in time order")
          + (f"; resuming at {day(last_day * DAY)[:10]}" if last_day is not None else ""), flush=True)
    state = {"ceiling": min(before - after, DAY) if spread else before - after, "skips": 0}
    month = None
    with open(path, "a", encoding="utf-8") as out:
        for d0, d1, off, d in slices:
            if n >= cap:
                break
            if spread and last_day is not None and d < last_day:
                continue
            if spread and day(d0)[:7] != month:
                if month is not None:
                    print(f"    r/{sub}: {n:,} comments (through {month})", flush=True)
                month = day(d0)[:7]
            want = min(quota - (saved[d] if spread else 0), cap - n)
            mid = min(d0 + off, d1 - 1)
            report = None
            if not spread:   # one long pass: report every 2,000 comments
                def report(got, t, n0=n, mark=[n // 2000]):
                    if (n0 + got) // 2000 > mark[0]:
                        mark[0] = (n0 + got) // 2000
                        print(f"    r/{sub}: {n0 + got:,} comments (up to {day(t)[:10]})", flush=True)
            for lo, hi in ((mid, d1), (d0, mid)):
                if want > 0 and hi > lo:
                    got = page_through(sub, lo, hi, want, seen, out, group, fetch, pause, state, gaps,
                                       progress=report)
                    want -= got
                    n += got
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
    ap.add_argument("--no-quota", action="store_true",
                    help="ignore daily quotas: first comments in time order (quick trials)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    after, before = to_epoch(cfg["window"]["after"]), to_epoch(cfg["window"]["before"])
    caps = cfg.get("max_comments_per_subreddit_by_group", {})
    quotas = {} if args.no_quota else cfg.get("comments_per_day_by_group", {})
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
                                                       args.pause, gaps=g, quota=quotas.get(group))
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
