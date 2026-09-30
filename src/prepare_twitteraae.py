"""Build the balanced sentiment/race dataset from the TwitterAAE corpus.

Python 3 re-implementation of make_data.py + data_utils.py from Elazar & Goldberg
(2018). It streams the corpus instead of loading it into pandas, so it runs in
8 GB of RAM (GitHub Codespaces / Colab).

Recipe (as in the original code):
  * dialect group: AA-aligned if P(AA) > 0.8, White-aligned if P(White) > 0.8
  * sentiment: tweet contains one of the paper's happy / sad emojis
  * emojis are then removed, text tokenised with twokenize, @mentions mapped
  * tweets with < 3 tokens, or only mentions, are dropped
  * exact duplicate texts are removed entirely (pandas keep=False)
  * 4 quadrants (pos/neg sentiment x AA/White); per quadrant the first 41,500
    go to training and the next 2,500 to test -> 166k train / 10k test.
    We also write 2,500 more per quadrant as a separate validation set, so
    models can be selected without looking at the test set (the original code
    selects the best epoch on the test set).

Two selection modes:
  clean (default): a tweet with both happy and sad emojis is discarded, every
      tweet is used at most once, quadrants are shuffled with a fixed seed.
  faithful: mimics the original code exactly as written, including two quirks
      found on inspection: tweets are collected emoji-by-emoji (so a tweet with
      two different happy emojis is added twice, and one with a happy and a sad
      emoji lands in both classes, because the conflict check compares tokens
      against regex strings and never fires), and quadrants are ordered by
      emoji rather than shuffled, so train and test come from different emojis.

Input: a TwitterAAE file (tab separated: tweet_id, timestamp, user_id,
lat/lon, census_blockgroup, text, P(AA), P(Hispanic), P(Other), P(White)).
Either twitteraae_all, or the twitteraae_all_aa / twitteraae_all_white subsets.

Usage:
  python src/prepare_twitteraae.py data/raw/TwitterAAE-full-v1 data/processed/sent_race
  python src/prepare_twitteraae.py <in> <out> --mode faithful
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textutils import HAPPY, SAD, MENTION, decode_escapes, emoji_hits, normalize_text  # noqa: E402

CONF_LEVEL = 0.8
MIN_SENTENCE_LEN = 3
SEED = 16
TRAIN_PER_QUAD = 41500
TEST_PER_QUAD = 2500
VAL_PER_QUAD = 2500  # extra held-out set (not in the paper) for honest model selection
QUADS = ["pos_aa", "pos_wh", "neg_aa", "neg_wh"]


def find_inputs(path):
    """Return the list of TwitterAAE files to read."""
    if os.path.isfile(path):
        return [path]
    names = os.listdir(path)
    subsets = [n for n in names if n in ("twitteraae_all_aa", "twitteraae_all_white")]
    if len(subsets) == 2:
        return [os.path.join(path, n) for n in sorted(subsets)]
    if "twitteraae_all" in names:
        return [os.path.join(path, "twitteraae_all")]
    cands = [n for n in names if n.startswith("twitteraae")]
    if cands:
        return [os.path.join(path, n) for n in sorted(cands)]
    raise SystemExit(f"No TwitterAAE file found in {path}. Contents: {names[:20]}")


def parse_line(line):
    f = line.rstrip("\n").split("\t")
    if len(f) < 10:
        return None
    try:
        aa = float(f[-4])
        wh = float(f[-1])
    except ValueError:
        return None
    text = "\t".join(f[5:-4])
    return text, aa, wh


def text_key(text):
    return hashlib.md5(text.encode("utf-8", "ignore")).digest()


def scan(files, max_lines=None):
    """Pass 1: keep high-confidence tweets that contain a sentiment emoji."""
    stats = Counter()
    cands = []  # dicts: key (md5 of raw text), text, group, happy, sad
    t0 = time.time()
    for fn in files:
        with open(fn, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stats["lines_read"] += 1
                if max_lines and stats["lines_read"] > max_lines:
                    break
                row = parse_line(line)
                if row is None:
                    stats["unparseable"] += 1
                    continue
                raw, aa, wh = row
                if aa > CONF_LEVEL:
                    group = "aa"
                elif wh > CONF_LEVEL:
                    group = "wh"
                else:
                    continue
                stats[f"highconf_{group}"] += 1
                text = decode_escapes(raw)
                h = emoji_hits(text, HAPPY)
                s = emoji_hits(text, SAD)
                if not h and not s:
                    continue
                cands.append({"key": text_key(raw), "text": text, "group": group, "happy": h, "sad": s})
                if stats["lines_read"] % 5_000_000 == 0:
                    print(f"  {stats['lines_read']:,} lines, {len(cands):,} candidates, "
                          f"{time.time() - t0:.0f}s", flush=True)
    stats["emoji_candidates"] = len(cands)
    return cands, stats


def duplicate_texts(files, cands, max_lines=None):
    """Pass 2: texts occurring more than once in the input (pandas keep=False)."""
    wanted = {c["key"] for c in cands}
    counts = Counter()
    n = 0
    for fn in files:
        with open(fn, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                n += 1
                if max_lines and n > max_lines:
                    break
                row = parse_line(line)
                if row is None:
                    continue
                k = text_key(row[0])
                if k in wanted:
                    counts[k] += 1
    return {k for k, v in counts.items() if v > 1}


def valid_tokens(toks):
    if len(toks) < MIN_SENTENCE_LEN:
        return False
    if len(set(toks)) == 1 and toks[0] == MENTION:
        return False
    return True


def select_clean(cands, stats):
    quads = defaultdict(list)
    for c in cands:
        if c["happy"] and c["sad"]:
            stats["dropped_conflicting_emojis"] += 1
            continue
        toks = normalize_text(c["text"])
        if not valid_tokens(toks):
            stats["dropped_short"] += 1
            continue
        sent = "pos" if c["happy"] else "neg"
        quads[f"{sent}_{c['group']}"].append(toks)
    rng = random.Random(SEED)
    for q in QUADS:
        rng.shuffle(quads[q])
    return quads


def select_faithful(cands, stats):
    """Reproduce get_attr_sentiments() as written in the original code."""
    rng = random.Random(SEED)
    cands = cands[:]
    rng.shuffle(cands)  # original: sklearn.utils.shuffle(df, random_state=16)
    tok_cache = {}
    quads = defaultdict(list)
    for group in ("aa", "wh"):
        rows = [c for c in cands if c["group"] == group]
        for sent, emos in (("pos", HAPPY), ("neg", SAD)):
            key = f"{sent}_{group}"
            for i, _ in enumerate(emos):
                field = "happy" if sent == "pos" else "sad"
                for c in rows:
                    if i not in c[field]:
                        continue
                    ident = id(c)
                    if ident not in tok_cache:
                        tok_cache[ident] = normalize_text(c["text"])
                    toks = tok_cache[ident]
                    if not valid_tokens(toks):
                        continue
                    quads[key].append(toks)
    # quirk accounting
    both = sum(1 for c in cands if c["happy"] and c["sad"])
    multi = sum(1 for c in cands if len(c["happy"]) > 1 or len(c["sad"]) > 1)
    stats["faithful_tweets_in_both_classes"] = both
    stats["faithful_tweets_counted_multiple_times"] = multi
    return quads


def overlap_between_splits(quads):
    """How many test sentences also appear in training (duplicates leak)."""
    train, test = set(), []
    for q in QUADS:
        for toks in quads[q][:TRAIN_PER_QUAD]:
            train.add(" ".join(toks))
        test += [" ".join(t) for t in quads[q][TRAIN_PER_QUAD:TRAIN_PER_QUAD + TEST_PER_QUAD]]
    return sum(1 for t in test if t in train)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="TwitterAAE file or folder (unzipped TwitterAAE-full-v1)")
    ap.add_argument("output", help="output folder, e.g. data/processed/sent_race")
    ap.add_argument("--mode", choices=["clean", "faithful"], default="clean")
    ap.add_argument("--no-global-dedup", action="store_true",
                    help="skip the second pass that removes duplicated texts")
    ap.add_argument("--max-lines", type=int, default=None, help="read only the first N lines (testing)")
    args = ap.parse_args()

    files = find_inputs(args.input)
    print("Reading:", ", ".join(files))
    t0 = time.time()
    cands, stats = scan(files, args.max_lines)
    print(f"Pass 1 done: {len(cands):,} emoji tweets from {stats['lines_read']:,} lines "
          f"({time.time() - t0:.0f}s)")

    if not args.no_global_dedup:
        dups = duplicate_texts(files, cands, args.max_lines)
        before = len(cands)
        cands = [c for c in cands if c["key"] not in dups]
        stats["dropped_duplicate_texts"] = before - len(cands)
        print(f"Pass 2 done: removed {before - len(cands):,} duplicated tweets")

    quads = select_clean(cands, stats) if args.mode == "clean" else select_faithful(cands, stats)

    need = TRAIN_PER_QUAD + TEST_PER_QUAD + VAL_PER_QUAD
    os.makedirs(args.output, exist_ok=True)
    sizes = {}
    for q in QUADS:
        avail = len(quads[q])
        sizes[q] = {"available": avail, "written": min(avail, need)}
        if avail < TRAIN_PER_QUAD + TEST_PER_QUAD:
            print(f"WARNING: {q} has only {avail:,} tweets (paper split needs "
                  f"{TRAIN_PER_QUAD + TEST_PER_QUAD:,}); the split will be smaller than the paper's.")
        elif avail < need:
            print(f"NOTE: {q} has {avail:,} tweets; the extra validation set will be carved "
                  f"from training instead.")
        with open(os.path.join(args.output, q + ".txt"), "w", encoding="utf-8") as fh:
            for toks in quads[q][:need]:
                fh.write(" ".join(toks) + "\n")

    report = {
        "mode": args.mode,
        "input_files": files,
        "stats": dict(stats),
        "quadrants": sizes,
        "test_sentences_also_in_train": overlap_between_splits(quads),
        "mean_tokens": {q: round(sum(len(t) for t in quads[q][:need]) / max(1, min(len(quads[q]), need)), 2)
                        for q in QUADS},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.output, "data_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
