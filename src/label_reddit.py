"""Distant-supervision sentiment labels and balanced quadrants for the Reddit data.

Input : data/reddit/clean/comments.jsonl  (from preprocess_reddit.py)
Output: data/reddit/processed/<variant>/{pos_ng,pos_us,neg_ng,neg_us}.txt,
        ids per quadrant, split.json, label_report.json

Sentiment (main task y), in the spirit of the original emoji-based labels:
  * emoji: comment contains a happy or a sad emoji from the paper's lists
    (comments with both are dropped); emojis are removed from the text, so the
    label cannot be read straight off the input, as in the original.
  * vader (only with --labels emoji+vader): no emoji, but the VADER lexicon
    score is strongly positive (>= 0.6) or negative (<= -0.6). Used because
    emojis are much rarer on Reddit than on Twitter.
Protected attribute z: ng (Nigerian English subreddits) = 1, us = 0.

Design guards
  * the four quadrants are balanced, and matched on comment length and on
    label source (per length bin and source, every quadrant gets the same
    count), so neither length nor the way a comment was labelled can stand in
    for dialect. (Emojis are about four times more common in the Nigerian
    comments, so without the source match the Nigerian quadrants would hold
    more emoji-labelled and fewer VADER-labelled comments, which differ in
    wording);
  * train and test are author-disjoint (10% of authors held out), so an
    attacker cannot succeed by recognising individual people.
Variants: tokens (as written) and masked (topic words replaced by _TOPIC_).
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

LEN_BINS = [(3, 5), (6, 10), (11, 20), (21, 40)]
SEED = 16


def vader_labeller():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        raise SystemExit("pip install vaderSentiment  (needed for --labels emoji+vader)")
    sia = SentimentIntensityAnalyzer()

    def lab(tokens, thr=0.6):
        s = sia.polarity_scores(" ".join(tokens))["compound"]
        return "pos" if s >= thr else "neg" if s <= -thr else None
    return lab


def length_bin(n):
    for i, (a, b) in enumerate(LEN_BINS):
        if a <= n <= b:
            return i
    return None


def is_test_author(h, frac=0.1):
    return int(hashlib.md5(h.encode()).hexdigest(), 16) % 1000 < frac * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/reddit/clean/comments.jsonl")
    ap.add_argument("--out", default="data/reddit/processed")
    ap.add_argument("--labels", choices=["emoji", "emoji+vader"], default="emoji+vader")
    ap.add_argument("--vader-threshold", type=float, default=0.6)
    args = ap.parse_args()

    vader = vader_labeller() if args.labels == "emoji+vader" else None
    rep = {"labels": args.labels, "label_source": Counter(), "unlabelled": 0, "conflicting_emojis": 0}
    pools = defaultdict(list)  # (sent, group, bin, split, source) -> rows
    for ln in open(args.clean, encoding="utf-8"):
        c = json.loads(ln)
        if c["happy"] and c["sad"]:
            rep["conflicting_emojis"] += 1
            continue
        if c["happy"] or c["sad"]:
            sent, src = ("pos" if c["happy"] else "neg"), "emoji"
        elif vader:
            sent, src = vader(c["tokens"], args.vader_threshold), "vader"
            if sent is None:
                rep["unlabelled"] += 1
                continue
        else:
            rep["unlabelled"] += 1
            continue
        b = length_bin(c["n_tokens"])
        if b is None:
            continue
        split = "test" if is_test_author(c["author_hash"]) else "train"
        c["label_source"] = src
        pools[(sent, c["group"], b, split, src)].append(c)
        rep["label_source"][f"{c['group']}/{sent}/{src}"] += 1

    rng = random.Random(SEED)
    quads = {q: {"train": [], "test": []} for q in ("pos_ng", "pos_us", "neg_ng", "neg_us")}
    rep["per_length_bin_and_source"] = {}
    sources = ("emoji", "vader") if vader else ("emoji",)
    for split in ("train", "test"):
        for b, rng_bin in enumerate(LEN_BINS):
            for src in sources:
                keys = [(s, g, b, split, src) for s in ("pos", "neg") for g in ("ng", "us")]
                k = min(len(pools[key]) for key in keys)
                rep["per_length_bin_and_source"][f"{split}_{rng_bin[0]}-{rng_bin[1]}_tokens_{src}"] = k
                for s, g, _, _, _ in keys:
                    rows = pools[(s, g, b, split, src)]
                    rng.shuffle(rows)
                    quads[f"{s}_{g}"][split] += rows[:k]
    split_counts = {}
    for variant, field in (("tokens", "tokens"), ("masked", "tokens_masked")):
        d = os.path.join(args.out, variant)
        os.makedirs(d, exist_ok=True)
        for q, parts in quads.items():
            tr, te = parts["train"], parts["test"]
            rng.shuffle(tr)
            with open(os.path.join(d, q + ".txt"), "w", encoding="utf-8") as fh:
                for c in tr + te:
                    fh.write(" ".join(c[field]) + "\n")
            with open(os.path.join(d, q + ".ids.jsonl"), "w", encoding="utf-8") as fh:
                for split, part in (("train", tr), ("test", te)):
                    for c in part:
                        fh.write(json.dumps({"id": c["id"], "split": split,
                                             "label_source": c["label_source"],
                                             "subreddit": c["subreddit"]}) + "\n")
            split_counts[q] = {"train": len(tr), "test": len(te)}
        json.dump(split_counts, open(os.path.join(d, "split.json"), "w"), indent=2)
    rep["label_source"] = dict(rep["label_source"])
    rep["quadrants"] = split_counts
    rep["quadrant_label_sources"] = {
        q: dict(Counter(c["label_source"] for part in parts.values() for c in part)) for q, parts in quads.items()}
    rep["total"] = sum(v["train"] + v["test"] for v in split_counts.values())
    json.dump(rep, open(os.path.join(args.out, "label_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
