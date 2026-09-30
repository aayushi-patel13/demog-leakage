"""Offline test of the Reddit pipeline: fake Arctic Shift pages -> collector ->
preprocess -> label. Writes mock raw files to data/mock/reddit_raw.

The fake API serves synthetic comments in time order, 100 per page, with the
same fields as Arctic Shift, including deleted comments, bots, quotes, links,
emojis, repeated authors and an author active in both groups. Nothing here is
real data; it only checks the code paths.
"""
import os
import random
import zlib
import sys

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


def make_fake_api(group, sub, n_total, seed):
    rng = random.Random(seed)
    t0 = 1704067200
    comments = []
    t = t0
    for i in range(n_total):
        t += rng.choice([0, 1, 5, 30, 120])  # includes same-second comments
        comments.append(fake_comment(rng, group, sub, t, i))

    def fetch(subreddit, after, before, limit=100):
        page = [c for c in comments if c["created_utc"] > after and c["created_utc"] < before]
        return page[:limit]
    fetch.comments = comments
    return fetch


def main():
    out = "data/mock/reddit_raw"
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    subs = {"ng": ["Nigeria", "lagos"], "us": ["AskAnAmerican", "chicago"]}
    for g, ss in subs.items():
        for k, s in enumerate(ss):
            fetch = make_fake_api(g, s, 6000, seed=zlib.crc32((g + s).encode()) % 1000)
            n = collect_reddit.collect_sub(g, s, 1704067200, 1751328000, cap=5000, outdir=out,
                                           pause=0, fetch=fetch)
            assert n == 5000, n
            # resume: a second call must not duplicate anything
            n2 = collect_reddit.collect_sub(g, s, 1704067200, 1751328000, cap=5000, outdir=out,
                                            pause=0, fetch=fetch)
            assert n2 == 5000, n2
            import json
            got = [json.loads(l)["id"] for l in open(os.path.join(out, f"{g}_{s}.jsonl"))]
            want = [c["id"] for c in fetch.comments[:5000]]
            assert got == want, "comments lost or reordered at page boundaries"
    print("collector OK (pagination, caps and resume)")


if __name__ == "__main__":
    main()
