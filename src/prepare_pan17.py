"""Build a sentiment x language-variety dataset from PAN 2017 author profiling.

PAN17 (Rangel et al., 2017) labels Twitter authors with gender and English
language variety (Australia, Canada, Great Britain, Ireland, New Zealand,
United States). Variety is the closest existing analogue to the dialect
attribute in the original paper, so we use it as the protected attribute:
by default z = 1 for Great Britain and z = 0 for the United States. The main
task is the same emoji-based sentiment as for TwitterAAE.

Expected layout after unzipping the training set:
  <dir>/en/<author_id>.xml   one file per author, tweets in <document> elements
  <dir>/en/truth.txt         lines "author_id:::gender:::variety"
If the XML files turn out to hold tweet ids instead of text, the script says so:
that would mean PAN17 has the same decay problem as PAN16.

Train and test are author-disjoint (each author contributes ~100 tweets, so a
shared author would let an attacker recognise people rather than varieties);
20% of authors are held out for test. Retweets are left out (not the author's
own writing), and so is every text that is identical to another after
normalisation, as in the clean TwitterAAE data.

Usage:
  python src/prepare_pan17.py data/raw/pan17/en data/processed/pan17_gb_us
  python src/prepare_pan17.py data/raw/pan17/en data/processed/pan17_gender --attribute gender
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textutils import HAPPY, SAD, emoji_hits, normalize_text  # noqa: E402

SEED = 16


def read_truth(folder):
    truth = {}
    for line in open(os.path.join(folder, "truth.txt"), encoding="utf-8"):
        parts = line.strip().split(":::")
        if len(parts) >= 3:
            truth[parts[0]] = {"gender": parts[1].lower(), "variety": parts[2].lower()}
    return truth


def read_author(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    return [(d.text or "").strip() for d in root.iter("document")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder with <author>.xml files and truth.txt (the 'en' folder)")
    ap.add_argument("output")
    ap.add_argument("--attribute", choices=["variety", "gender"], default="variety")
    ap.add_argument("--z1", default="great britain", help="value mapped to z = 1")
    ap.add_argument("--z0", default="united states", help="value mapped to z = 0")
    args = ap.parse_args()
    if args.attribute == "gender" and args.z1 == "great britain":
        args.z1, args.z0 = "female", "male"

    truth = read_truth(args.input)
    rep = {"authors_in_truth": len(truth), "by_value": Counter(), "tweets_read": 0,
           "looks_like_ids_only": 0, "emoji_tweets": Counter(), "conflicting_emojis": 0}
    pools = defaultdict(list)  # (sent, z, split)
    for path in sorted(glob.glob(os.path.join(args.input, "*.xml"))):
        aid = os.path.splitext(os.path.basename(path))[0]
        if aid not in truth:
            continue
        val = truth[aid][args.attribute]
        rep["by_value"][val] += 1
        if val not in (args.z1, args.z0):
            continue
        z = 1 if val == args.z1 else 0
        split = "test" if int(hashlib.md5(aid.encode()).hexdigest(), 16) % 10 < 2 else "train"
        for tw in read_author(path):
            rep["tweets_read"] += 1
            if re.fullmatch(r"\d{15,20}", tw):
                rep["looks_like_ids_only"] += 1
                continue
            if tw.startswith("RT @"):
                rep["retweets_skipped"] = rep.get("retweets_skipped", 0) + 1
                continue
            h, s = emoji_hits(tw, HAPPY), emoji_hits(tw, SAD)
            if not h and not s:
                continue
            if h and s:
                rep["conflicting_emojis"] += 1
                continue
            toks = normalize_text(tw)
            if len(toks) < 3:
                continue
            sent = "pos" if h else "neg"
            pools[(sent, z, split)].append(toks)
            rep["emoji_tweets"][f"{sent}_z{z}"] += 1
    if rep["tweets_read"] and rep["looks_like_ids_only"] > 0.5 * rep["tweets_read"]:
        print("WARNING: most documents are tweet ids, not text. PAN17 would then need the Twitter API,"
              " like PAN16, and cannot be used without it.")
    # texts identical after normalisation, anywhere in the pools: drop every copy
    seen = Counter(" ".join(t) for rows in pools.values() for t in rows)
    for key in pools:
        before = len(pools[key])
        pools[key] = [t for t in pools[key] if seen[" ".join(t)] == 1]
        rep["dropped_same_after_normalising"] = rep.get("dropped_same_after_normalising", 0) + before - len(pools[key])
    rng = random.Random(SEED)
    os.makedirs(args.output, exist_ok=True)
    split_counts = {}
    for split in ("train", "test"):
        k = min(len(pools[(s, z, split)]) for s in ("pos", "neg") for z in (0, 1))
        for s in ("pos", "neg"):
            for z in (0, 1):
                rng.shuffle(pools[(s, z, split)])
                pools[(s, z, split)] = pools[(s, z, split)][:k]
    for s in ("pos", "neg"):
        for z in (0, 1):
            name = f"{s}_z{z}"
            tr, te = pools[(s, z, "train")], pools[(s, z, "test")]
            with open(os.path.join(args.output, name + ".txt"), "w", encoding="utf-8") as fh:
                for toks in tr + te:
                    fh.write(" ".join(toks) + "\n")
            split_counts[name] = {"train": len(tr), "test": len(te)}
    json.dump(split_counts, open(os.path.join(args.output, "split.json"), "w"), indent=2)
    rep.update({"attribute": args.attribute, "z1": args.z1, "z0": args.z0, "quadrants": split_counts,
                "by_value": dict(rep["by_value"]), "emoji_tweets": dict(rep["emoji_tweets"])})
    json.dump(rep, open(os.path.join(args.output, "data_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
