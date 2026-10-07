"""Clean the collected Reddit comments.

Input : data/reddit/raw/*.jsonl  (from collect_reddit.py)
Output: data/reddit/clean/comments.jsonl and data/reddit/clean/clean_report.json

With --until YYYY-MM-DD, comments from that date on are left out, so that both
groups cover the same period if one group's collection stopped early.

Steps (each counted in the report):
  1. drop deleted/removed comments and bot accounts (AutoModerator, *bot)
  2. strip quoted lines ("> ..."), Markdown, links; map u/user and r/sub to
     placeholders (subreddit names would give the group away)
  3. tokenise exactly as the tweets (twokenize, emojis removed, @/u mentions mapped)
  4. keep 3-40 tokens, so comments are closer in length to tweets
  5. drop authors who posted in both groups (their variety is ambiguous)
  6. cap each author at 25 comments, so no single person dominates a group
  7. remove texts that occur more than once (all copies, as in the original)
Authors are replaced by a salted hash; usernames are never written out.
Two extra fields support the analysis: a Nigerian Pidgin marker count, and a
topic-masked token list in which place names, politicians, currencies and
similar giveaways are replaced by _TOPIC_ (for a "dialect or topic?" ablation).
"""
import argparse
import glob
import hashlib
import html
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textutils import HAPPY, SAD, emoji_hits, normalize_text  # noqa: E402

MIN_TOK, MAX_TOK, AUTHOR_CAP, SEED = 3, 40, 25, 16

PIDGIN_MARKERS = {"dey", "wetin", "abeg", "una", "wahala", "sef", "abi", "oya", "sabi", "pikin",
                  "comot", "wey", "shey", "oga", "ehn", "jare", "gbam", "wahala", "sha", "dem",
                  "shebi", "biko", "ehen", "chai", "haba", "kuku", "kpatakpata", "ogbeni"}

TOPIC_TERMS = {
    # Nigeria
    "nigeria", "nigerian", "nigerians", "naija", "lagos", "abuja", "ibadan", "kano", "kaduna", "enugu",
    "owerri", "benin", "calabar", "jos", "ilorin", "abeokuta", "harcourt", "ph", "lekki", "ikeja", "yaba",
    "surulere", "ajah", "vi", "naira", "tinubu", "buhari", "atiku", "obi", "jonathan", "wike", "sanwo-olu",
    "yoruba", "igbo", "hausa", "fulani", "biafra", "apc", "pdp", "inec", "efcc", "nepa", "phcn", "danfo",
    "okada", "keke", "jollof", "eko", "lasgidi", "japa", "cbn", "nysc", "jamb", "waec", "unilag",
    # United States
    "america", "american", "americans", "usa", "u.s.", "u.s", "us-based", "chicago", "chicagoland",
    "houston", "atlanta", "atl", "philadelphia", "philly", "texas", "illinois", "georgia", "pennsylvania",
    "california", "florida", "york", "nyc", "jersey", "ohio", "michigan", "virginia", "carolina",
    "dallas", "austin", "boston", "seattle", "denver", "detroit", "trump", "biden", "harris", "obama",
    "republican", "republicans", "democrat", "democrats", "gop", "maga", "cta", "septa", "marta", "hpd",
    "cpd", "irs", "dmv", "ssn", "fema", "midwest",
    # money
    "dollar", "dollars", "usd", "ngn", "kobo",
}
MONEY_RE = re.compile(r"^[$₦£€]\d|^\d+[kKmM]?[$₦]$|^N\d{2,}")

QUOTE_RE = re.compile(r"^\s*(&gt;|>).*$", re.MULTILINE)
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:[^)]+)\)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]+")
SUB_RE = re.compile(r"(?<!\w)/?r/[A-Za-z0-9_]+")
MD_CHARS_RE = re.compile(r"[*_~^`#]+")


def is_bot(author):
    a = (author or "").lower()
    return a in {"automoderator", "[deleted]", ""} or a.endswith("bot") or a.endswith("-bot")


def clean_body(body):
    t = html.unescape(body)
    t = QUOTE_RE.sub(" ", t)
    t = MD_LINK_RE.sub(r"\1", t)
    # sentinels first, so Markdown stripping cannot damage the placeholders
    t = URL_RE.sub(" QQURLQQ ", t)
    t = USER_RE.sub(" @user ", t)  # becomes the mention token after tokenising
    t = SUB_RE.sub(" QQSUBQQ ", t)
    t = MD_CHARS_RE.sub(" ", t)
    t = t.replace("QQURLQQ", "_URL_").replace("QQSUBQQ", "_SUBREDDIT_")
    return re.sub(r"\s+", " ", t).strip()


def mask_topic(tokens):
    out = []
    for t in tokens:
        lo = t.lower().strip(".,!?;:'\"()")
        out.append("_TOPIC_" if lo in TOPIC_TERMS or MONEY_RE.match(t) else t)
    return out


def author_hash(author, salt):
    return hashlib.sha256((salt + author).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/reddit/raw")
    ap.add_argument("--out", default="data/reddit/clean")
    ap.add_argument("--salt", default="comp8240", help="salt for author hashing")
    ap.add_argument("--until", default=None,
                    help="keep comments created before this date (YYYY-MM-DD), so both groups cover "
                         "the same period when one collection stopped early")
    args = ap.parse_args()
    until = None
    if args.until:
        until = int(datetime.strptime(args.until, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    os.makedirs(args.out, exist_ok=True)
    rep = {"until": args.until, "raw_per_subreddit": Counter(), "dropped": Counter()}
    span = {}   # first and last comment per subreddit, as collected
    rows = []
    for fn in sorted(glob.glob(os.path.join(args.raw, "*.jsonl"))):
        if os.path.basename(fn) == "skipped_windows.jsonl":   # collector's gap log, not comments
            continue
        with open(fn, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    c = json.loads(ln)
                except json.JSONDecodeError:
                    rep["dropped"]["bad_json"] += 1
                    continue
                if "group" not in c or "id" not in c:
                    rep["dropped"]["not_a_comment"] += 1
                    continue
                key = f"{c['group']}/{c['subreddit']}"
                rep["raw_per_subreddit"][key] += 1
                t = int(c["created_utc"])
                lo, hi = span.get(key, (t, t))
                span[key] = (min(lo, t), max(hi, t))
                if until and t >= until:
                    rep["dropped"]["after_until_date"] += 1
                    continue
                body = c.get("body") or ""
                if body.strip() in ("[deleted]", "[removed]", ""):
                    rep["dropped"]["deleted_or_removed"] += 1
                    continue
                if is_bot(c.get("author")):
                    rep["dropped"]["bot_or_deleted_author"] += 1
                    continue
                text = clean_body(body)
                toks = normalize_text(text)
                if not (MIN_TOK <= len(toks) <= MAX_TOK):
                    rep["dropped"]["length_outside_3_40"] += 1
                    continue
                rows.append({"id": c["id"], "group": c["group"], "subreddit": c["subreddit"],
                             "author": c["author"], "created_utc": int(c["created_utc"]),
                             "text": text, "tokens": toks})
    # authors active in both groups
    groups_of = defaultdict(set)
    for r in rows:
        groups_of[r["author"]].add(r["group"])
    both = {a for a, g in groups_of.items() if len(g) > 1}
    before = len(rows)
    rows = [r for r in rows if r["author"] not in both]
    rep["dropped"]["author_in_both_groups"] = before - len(rows)
    rep["authors_in_both_groups"] = len(both)
    # per-author cap
    rng = random.Random(SEED)
    by_author = defaultdict(list)
    for r in rows:
        by_author[r["author"]].append(r)
    capped = []
    for a, rs in by_author.items():
        if len(rs) > AUTHOR_CAP:
            rep["dropped"]["over_author_cap"] += len(rs) - AUTHOR_CAP
            rs = rng.sample(rs, AUTHOR_CAP)
        capped += rs
    rows = capped
    # exact duplicates (keep none)
    cnt = Counter(" ".join(r["tokens"]).lower() for r in rows)
    before = len(rows)
    rows = [r for r in rows if cnt[" ".join(r["tokens"]).lower()] == 1]
    rep["dropped"]["duplicate_text"] = before - len(rows)
    # enrich and write
    stats = defaultdict(Counter)
    ntok = defaultdict(list)
    authors = defaultdict(set)
    per_month = defaultdict(Counter)   # comments kept, by group and month
    per_sub = Counter()
    with open(os.path.join(args.out, "comments.jsonl"), "w", encoding="utf-8") as out:
        for r in sorted(rows, key=lambda r: r["created_utc"]):
            g = r["group"]
            low = [t.lower() for t in r["tokens"]]
            pid = sum(1 for t in low if t in PIDGIN_MARKERS)
            rec = {"id": r["id"], "group": g, "subreddit": r["subreddit"],
                   "author_hash": author_hash(r["author"], args.salt), "created_utc": r["created_utc"],
                   "text": r["text"], "tokens": r["tokens"], "tokens_masked": mask_topic(r["tokens"]),
                   "n_tokens": len(r["tokens"]), "happy": emoji_hits(r["text"], HAPPY),
                   "sad": emoji_hits(r["text"], SAD), "pidgin_markers": pid}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[g]["comments"] += 1
            stats[g]["with_sentiment_emoji"] += bool(rec["happy"] or rec["sad"])
            stats[g]["pidgin_2plus_markers"] += pid >= 2
            stats[g]["topic_masked_tokens"] += sum(t == "_TOPIC_" for t in rec["tokens_masked"])
            ntok[g].append(len(r["tokens"]))
            authors[g].add(rec["author_hash"])
            per_month[g][f"{datetime.fromtimestamp(r['created_utc'], timezone.utc):%Y-%m}"] += 1
            per_sub[f"{g}/{r['subreddit']}"] += 1
    rep["final_per_group"] = {g: dict(s, authors=len(authors[g]),
                                      median_tokens=sorted(ntok[g])[len(ntok[g]) // 2] if ntok[g] else 0)
                              for g, s in stats.items()}
    rep["collected_span"] = {k: [f"{datetime.fromtimestamp(a, timezone.utc):%Y-%m-%d}",
                                 f"{datetime.fromtimestamp(b, timezone.utc):%Y-%m-%d}"]
                             for k, (a, b) in sorted(span.items())}
    rep["final_per_subreddit"] = dict(per_sub)
    rep["final_per_month"] = {g: dict(sorted(c.items())) for g, c in per_month.items()}
    rep["raw_per_subreddit"] = dict(rep["raw_per_subreddit"])
    rep["dropped"] = dict(rep["dropped"])
    with open(os.path.join(args.out, "clean_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
