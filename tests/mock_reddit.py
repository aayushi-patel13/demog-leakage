"""Offline test of the Reddit pipeline: fake Arctic Shift pages -> collector ->
preprocess -> label. Writes mock raw files to data/mock/reddit_raw.

The fake API serves synthetic comments in time order, 100 per page, with the
same fields as Arctic Shift, including deleted comments, bots, quotes, links,
emojis, repeated authors and an author active in both groups. Nothing here is
real data; it only checks the code paths.

Checked for the collector: the daily quota (busy subreddits give exactly the
quota every day, quiet ones give everything), random start times spread over
the day, no duplicates, resuming after an interruption gives the same sample
as an uninterrupted run, refused wide queries are narrowed, and a stretch that
always fails is skipped and logged without losing anything else.
"""
import json
import os
import random
import shutil
import sys
import zlib
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import collect_reddit  # noqa: E402

NG = ["abeg", "wetin", "dey", "sef", "wahala", "oga", "na", "e", "don", "una", "sha", "ehn", "o",
      "light", "fuel", "traffic", "Lagos", "naira", "jollof", "danfo", "Tinubu"]
US = ["gonna", "y'all", "awesome", "super", "totally", "dude", "honestly", "like", "downtown",
      "freeway", "Chicago", "dollars", "Trump", "Target", "grocery", "tipping", "football"]
COMMON = ["the", "i", "to", "and", "it", "is", "this", "that", "you", "in", "my", "for", "people",
          "really", "just", "think", "time", "good", "bad", "day", "work", "going", "know"]
POSW, NEGW = ["love", "great", "happy", "amazing", "best"], ["hate", "terrible", "sad", "awful", "worst"]
HAPPY, SAD = ["\U0001F602", "\U0001F60D", "\U0001F60A", ":)"], ["\U0001F62D", "\U0001F612", "\U0001F622", ":("]

DAY = 86400
T0 = 1704067200            # 2024-01-01
DAYS = 20
T1 = T0 + DAYS * DAY
CAP = 5000
QUOTA = 250                # per day: CAP / DAYS


def fake_comment(rng, group, sub, t, i):
    words = [rng.choice(NG if (group == "ng") == (rng.random() < 0.75) else US) if rng.random() < 0.25
             else rng.choice(COMMON) for _ in range(rng.randint(3, 30))]
    r = rng.random()
    if r < 0.25:
        words.insert(rng.randint(0, len(words)), rng.choice(HAPPY) + " " + rng.choice(POSW))
    elif r < 0.5:
        words.insert(rng.randint(0, len(words)), rng.choice(SAD) + " " + rng.choice(NEGW))
    elif r < 0.7:
        words.append(rng.choice(POSW + NEGW) + " " + rng.choice(POSW + NEGW))
    body = " ".join(words)
    x = rng.random()
    if x < 0.03:
        body = "[deleted]"
    elif x < 0.06:
        body = "> quoted text here\n\n" + body
    elif x < 0.09:
        body += " see [this](https://example.com/x) and r/" + sub + " u/someone"
    author = "AutoModerator" if rng.random() < 0.02 else f"{group}_user{rng.randint(1, 400)}"
    if rng.random() < 0.01:
        author = "both_groups_person"
    return {"id": f"{group}{sub}{i}", "author": author, "body": body, "created_utc": t,
            "subreddit": sub, "link_id": "t3_x", "parent_id": "t3_x", "score": 1,
            "author_flair_text": None}


def make_fake_api(group, sub, per_day, seed, quiet_days=()):
    """Comments over DAYS days at about `per_day` a day, some sharing a second."""
    rng = random.Random(seed)
    comments, t, i = [], T0, 0
    gap = DAY / per_day
    while True:
        t += int(rng.expovariate(1 / gap)) if rng.random() > 0.05 else 0
        if t >= T1:
            break
        if (t - T0) // DAY in quiet_days:
            continue
        comments.append(fake_comment(rng, group, sub, t, i))
        i += 1

    def fetch(subreddit, after, before, limit=100):
        page = [c for c in comments if after < c["created_utc"] < before]
        return page[:limit]
    fetch.comments = comments
    return fetch


def refuse_wide(fetch, max_span):
    """Like the archive's HTTP 422: refuse queries over a long time range."""
    def f(subreddit, after, before, limit=100):
        if before - after > max_span:
            raise collect_reddit.QueryFailed("simulated HTTP 422 Query timed out")
        return fetch(subreddit, after, before, limit)
    f.comments = fetch.comments
    return f


def refuse_stretch(fetch, bad_from, bad_to):
    """Refuse every query touching one stretch of time, however narrow."""
    def f(subreddit, after, before, limit=100):
        if after < bad_to and before > bad_from:
            raise collect_reddit.QueryFailed("simulated persistent failure")
        return fetch(subreddit, after, before, limit)
    f.comments = fetch.comments
    return f


class Interrupted(Exception):
    pass


def interrupt_after(fetch, calls):
    count = [0]

    def f(subreddit, after, before, limit=100):
        count[0] += 1
        if count[0] > calls:
            raise Interrupted()
        return fetch(subreddit, after, before, limit)
    f.comments = fetch.comments
    return f


def saved(out, g, s):
    rows = []
    for ln in open(os.path.join(out, f"{g}_{s}.jsonl"), encoding="utf-8"):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:   # the deliberately broken line in test 3
            pass
    return rows


def check_quota(rows, fetch, quota, exempt=lambda t: False):
    ids = [c["id"] for c in rows]
    assert len(ids) == len(set(ids)), "duplicate comments"
    got = Counter(c["created_utc"] // DAY for c in rows)
    avail = Counter(c["created_utc"] // DAY for c in fetch.comments)
    for d, a in avail.items():
        if exempt(d * DAY) or exempt(d * DAY + DAY - 1):
            continue
        assert got[d] == min(quota, a), (d, got[d], a)
    starts = Counter()
    for d in got:
        first = min(c["created_utc"] for c in rows if c["created_utc"] // DAY == d)
        starts[(first % DAY) // 3600] += 1
    return len(starts)


def main():
    out, ref = "data/mock/reddit_raw", "data/mock/reddit_ref"
    for d in (out, ref):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    quota = QUOTA
    busy, small = 400, 120
    seed = lambda g, s: zlib.crc32((g + s).encode()) % 1000

    # 1. busy subreddit: exactly the quota every day, start times spread over the day
    f = make_fake_api("ng", "Nigeria", busy, seed("ng", "Nigeria"))
    n = collect_reddit.collect_sub("ng", "Nigeria", T0, T1, CAP, out, 0, fetch=f, quota=quota)
    assert n == CAP, n
    hours = check_quota(saved(out, "ng", "Nigeria"), f, quota)
    assert hours >= 6, f"daily start times not spread over the day ({hours} distinct hours)"

    # 2. small subreddit with a stretch that always fails: everything else collected
    f = make_fake_api("ng", "lagos", small, seed("ng", "lagos"))
    t_bad = T0 + 7 * DAY + 5 * 3600
    gaps = []
    collect_reddit.collect_sub("ng", "lagos", T0, T1, CAP, out, 0,
                               fetch=refuse_stretch(f, t_bad, t_bad + 600), gaps=gaps,
                               quota=quota)
    rows = saved(out, "ng", "lagos")
    assert gaps, "the failing stretch was not skipped"
    check_quota(rows, f, quota, exempt=lambda t: abs(t - t_bad) < DAY)
    have = {c["id"] for c in rows}
    lost = [c for c in f.comments if c["id"] not in have]
    assert lost and all(abs(c["created_utc"] - t_bad) <= 3600 + 600 for c in lost), lost[:2]

    # 3. interrupted run, then resumed: same sample as an uninterrupted run
    f = make_fake_api("us", "AskAnAmerican", busy, seed("us", "AskAnAmerican"))
    collect_reddit.collect_sub("us", "AskAnAmerican", T0, T1, CAP, ref, 0, fetch=f, quota=quota)
    try:
        collect_reddit.collect_sub("us", "AskAnAmerican", T0, T1, CAP, out, 0,
                                   fetch=interrupt_after(f, 23), quota=quota)
        raise AssertionError("interruption did not happen")
    except Interrupted:
        pass
    with open(os.path.join(out, "us_AskAnAmerican.jsonl"), "a") as fh:
        fh.write('{"id": "half a li')       # as if killed in the middle of a write
    n = collect_reddit.collect_sub("us", "AskAnAmerican", T0, T1, CAP, out, 0, fetch=f, quota=quota)
    a = {c["id"] for c in saved(out, "us", "AskAnAmerican")}
    b = {c["id"] for c in saved(ref, "us", "AskAnAmerican")}
    assert n == CAP and a == b, (n, len(a ^ b))
    check_quota(saved(out, "us", "AskAnAmerican"), f, quota)
    # and a rerun after completion adds nothing
    assert collect_reddit.collect_sub("us", "AskAnAmerican", T0, T1, CAP, out, 0, fetch=f, quota=quota) == CAP

    # 4. archive refuses wide queries, and ten quiet days
    f = make_fake_api("us", "chicago", busy, seed("us", "chicago"), quiet_days=range(5, 15))
    collect_reddit.collect_sub("us", "chicago", T0, T1, CAP, out, 0, fetch=refuse_wide(f, 6 * 3600), quota=quota)
    check_quota(saved(out, "us", "chicago"), f, quota)

    # 5. no quota: all comments in time order up to the cap, resumable
    f = make_fake_api("us", "houston", small, seed("us", "houston"))
    try:
        collect_reddit.collect_sub("us", "houston", T0, T1, 700, ref, 0, fetch=interrupt_after(f, 4))
    except Interrupted:
        pass
    n = collect_reddit.collect_sub("us", "houston", T0, T1, 700, ref, 0, fetch=f)
    assert n == 700
    assert [c["id"] for c in saved(ref, "us", "houston")] == [c["id"] for c in f.comments[:700]]
    n = collect_reddit.collect_sub("us", "houston", T0, T1, 10 ** 6, ref, 0, fetch=refuse_wide(f, 3 * DAY))
    assert [c["id"] for c in saved(ref, "us", "houston")] == [c["id"] for c in f.comments]
    shutil.rmtree(ref)
    print("collector OK (daily quota, spread start times, resume, refused queries, skipped windows)")


if __name__ == "__main__":
    main()
