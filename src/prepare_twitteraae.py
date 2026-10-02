"""Build the balanced sentiment/race dataset from the TwitterAAE corpus.

Python 3 re-implementation of make_data.py + data_utils.py from Elazar & Goldberg
(2018). It streams the corpus in a single pass (straight out of the downloaded
zip if given the .zip), so it runs in a few GB of RAM and never unpacks the
12 GB file.

File format (TwitterAAE-full-v1/twitteraae_all, tab separated):
  tweet_id, "timestamp", "user_id", [lon, lat], "census_blockgroup",
  "text" (a JSON string: emojis as \\uXXXX surrogate pairs, \\n for newlines),
  P(AA), P(Hispanic), P(Asian/other), P(White)

Recipe (as in the original code):
  * dialect group: AA-aligned if P(AA) > 0.8, White-aligned if P(White) > 0.8
  * sentiment: tweet contains one of the paper's happy / sad emojis
  * emojis are then removed, text tokenised with twokenize, @mentions mapped
  * tweets with < 3 tokens, or only mentions, are dropped
  * texts that occur more than once in the corpus are removed entirely
    (pandas drop_duplicates(keep=False) in the original)
  * 4 quadrants (pos/neg x AA/White); per quadrant the first 41,500 go to
    training and the next 2,500 to test -> 166k train / 10k test. 2,500 more
    per quadrant form a separate validation set, so models can be selected
    without looking at the test set (the original selects on the test set).

Each quadrant is drawn as a uniform random sample (reservoir sampling), so
only a few hundred thousand tweets are ever held in memory.

Modes
  clean (default): tweets with both happy and sad emojis are dropped, each
      tweet is used once, quadrants are in random order, and tweets whose
      text is identical to another's after normalisation (same words, a
      different mention, link or emoji) are removed, so no text appears under
      both labels or in both training and test.
  faithful: reproduces how the original code behaves as written:
      - tweets are collected emoji by emoji, so a tweet with two different
        happy emojis is added twice;
      - the check meant to drop tweets with both happy and sad emojis compares
        tokens with regex strings and never fires, so they enter both classes;
      - the pattern for the sob emoji has a missing backslash
        ('\\\\\\ud83d\\ude2d'), so under Python 2 it never matches and sob-only
        tweets are never collected as sad;
      - quadrants are ordered by emoji, not shuffled, so the training and test
        portions can come from different emojis.
      The original's emoji order came from Python 2 set iteration and cannot be
      reproduced exactly; the list order of textutils.py is used instead.
  Both modes report how many tweets each quirk affects (data_report.json).

Usage:
  python src/prepare_twitteraae.py data/raw/TwitterAAE-full-v1.zip data/processed/sent_race
  python src/prepare_twitteraae.py data/raw/TwitterAAE-full-v1.zip data/processed/sent_race_faithful --mode faithful
  python src/prepare_twitteraae.py <zip> <out> --max-lines 2000000      # quick trial
"""
import argparse
import array
import contextlib
import io
import json
import os
import random
import sys
import time
import zipfile
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textutils import HAPPY, SAD, MENTION, decode_escapes, emoji_hits, normalize_text  # noqa: E402

CONF_LEVEL = 0.8
MIN_SENTENCE_LEN = 3
SEED = 16
TRAIN_PER_QUAD = 41500
TEST_PER_QUAD = 2500
VAL_PER_QUAD = 2500
NEED = TRAIN_PER_QUAD + TEST_PER_QUAD + VAL_PER_QUAD
RESERVOIR = 100000        # per quadrant (the original also kept 100,000 per quadrant)
FAITHFUL_RESERVOIR = 60000  # per (quadrant, emoji) bucket
QUADS = ["pos_aa", "pos_wh", "neg_aa", "neg_wh"]
SOB = SAD.index("\U0001F62D")


# ---------------------------------------------------------------- input files
def _pick(names):
    """Choose which TwitterAAE files to read from a list of (base)names."""
    base = {os.path.basename(n): n for n in names if not n.endswith("/")}
    if "twitteraae_all_aa" in base and "twitteraae_all_white" in base:
        return [base["twitteraae_all_aa"], base["twitteraae_all_white"]]
    if "twitteraae_all" in base:
        return [base["twitteraae_all"]]
    return sorted(v for k, v in base.items()
                  if k.startswith("twitteraae") and not k.endswith(".zip") and "limited" not in k)


def find_inputs(path):
    """Files to read: a file, a folder, or the downloaded .zip itself."""
    if path.endswith(".zip") and os.path.isfile(path):
        with zipfile.ZipFile(path) as zf:
            chosen = _pick(zf.namelist())
        if not chosen:
            raise SystemExit(f"No TwitterAAE file inside {path}")
        return [(path, m) for m in chosen]
    if os.path.isfile(path):
        return [path]
    names = os.listdir(path)
    chosen = _pick(names)
    if chosen:
        return [os.path.join(path, n) for n in chosen]
    raise SystemExit(f"No TwitterAAE file found in {path}. Contents: {names[:20]}")


@contextlib.contextmanager
def open_source(src):
    """Open a plain file, or a (zip, member) pair, as a text stream."""
    if isinstance(src, tuple):
        with zipfile.ZipFile(src[0]) as zf, zf.open(src[1]) as raw:
            yield io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    else:
        with open(src, encoding="utf-8", errors="replace") as fh:
            yield fh


def describe(src):
    return f"{src[0]}::{src[1]}" if isinstance(src, tuple) else src


# ---------------------------------------------------------------- text field
def maybe_emoji(field):
    """Cheap test on the raw (escaped) text: could it contain a sentiment emoji?"""
    if "\\ud83" in field or "\\u263" in field:  # escaped emoji, relaxed, frowning
        return True
    if not field.isascii():                      # emojis stored as raw characters
        return True
    if ("(" in field or ")" in field) and (":" in field or "=" in field):
        return True
    return ":D" in field


def decode_text(field, stats):
    """The text column is a JSON string literal; decode it to real characters."""
    if len(field) >= 2 and field[0] == '"' and field[-1] == '"':
        try:
            text = json.loads(field)
        except ValueError:
            stats["text_not_valid_json"] += 1
            text = decode_escapes(field[1:-1])
    else:
        text = decode_escapes(field)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates in truncated tweets
        text = text.encode("utf-8", "replace").decode("utf-8")
    return text


def valid_tokens(toks):
    if len(toks) < MIN_SENTENCE_LEN:
        return False
    if len(set(toks)) == 1 and toks[0] == MENTION:
        return False
    return True


# ---------------------------------------------------------------- the pass
def reservoir_add(bucket, seen, key, item, cap, rng):
    seen[key] += 1
    if len(bucket) < cap:
        bucket.append(item)
    else:
        j = rng.randrange(seen[key])
        if j < cap:
            bucket[j] = item


def scan(files, mode, max_lines, rng):
    stats = Counter()
    seen = Counter()   # items offered to each reservoir bucket
    pop = Counter()    # population count per quadrant (each tweet once)
    res = defaultdict(list)
    hashes = array.array("q")  # hash of every emoji tweet, any confidence (for dedup)
    emoji_counts = {g: Counter() for g in ("aa", "wh")}
    t0 = time.time()
    n = 0
    for src in files:
        with open_source(src) as fh:
            for line in fh:
                n += 1
                if max_lines and n > max_lines:
                    n -= 1
                    break
                if n % 5_000_000 == 0:
                    rate = n / max(1e-9, time.time() - t0)
                    print(f"  {n / 1e6:.0f}M lines | emoji tweets {stats['emoji_tweets_any_confidence']:,} | "
                          ", ".join(f"{q} {pop.get(q, 0):,}" for q in QUADS) +
                          f" | {time.time() - t0:.0f}s ({rate / 1000:.0f}k lines/s)", flush=True)
                f = line.rstrip("\r\n").split("\t")
                if len(f) < 10:
                    stats["unparseable_lines"] += 1
                    continue
                field = f[5] if len(f) == 10 else "\t".join(f[5:-4])
                if not maybe_emoji(field):
                    continue
                text = decode_text(field, stats)
                h = emoji_hits(text, HAPPY)
                s = emoji_hits(text, SAD)
                if not h and not s:
                    continue
                key = hash(field)
                hashes.append(key)
                stats["emoji_tweets_any_confidence"] += 1
                try:
                    aa, wh = float(f[-4]), float(f[-1])
                except ValueError:
                    stats["unparseable_lines"] += 1
                    continue
                if aa > CONF_LEVEL:
                    g = "aa"
                elif wh > CONF_LEVEL:
                    g = "wh"
                else:
                    continue
                # population statistics, including the original code's quirks
                stats[f"highconf_emoji_{g}"] += 1
                for i in h:
                    emoji_counts[g][HAPPY[i]] += 1
                for i in s:
                    emoji_counts[g][SAD[i]] += 1
                if h and s:
                    stats[f"quirk_both_happy_and_sad_{g}"] += 1
                if len(h) > 1 or len(s) > 1:
                    stats[f"quirk_two_or_more_emojis_same_class_{g}"] += 1
                if s and not h and s == [SOB]:
                    stats[f"quirk_sad_only_via_sob_{g}"] += 1
                if s and not h:
                    stats[f"sad_only_{g}"] += 1
                if h and not s:
                    stats[f"happy_only_{g}"] += 1
                item = (key, text)
                if mode == "clean":
                    if h and s:
                        continue
                    q = ("pos_" if h else "neg_") + g
                    pop[q] += 1
                    reservoir_add(res[q], seen, q, item, RESERVOIR, rng)
                else:
                    if h:
                        pop["pos_" + g] += 1
                    if [i for i in s if i != SOB]:
                        pop["neg_" + g] += 1
                    for i in h:
                        b = ("pos_" + g, i)
                        reservoir_add(res[b], seen, b, item, FAITHFUL_RESERVOIR, rng)
                    for i in s:
                        if i != SOB:
                            b = ("neg_" + g, i)
                            reservoir_add(res[b], seen, b, item, FAITHFUL_RESERVOIR, rng)
    stats["lines_read"] = n
    stats["seconds_scanning"] = round(time.time() - t0)
    return res, pop, hashes, emoji_counts, stats


def duplicated_keys(hashes):
    if not hashes:
        return set()
    arr = np.array(hashes, dtype=np.int64)
    arr.sort()
    same = arr[1:] == arr[:-1]
    return set(arr[1:][same].tolist())


def select(res, mode, dups, rng, stats):
    if mode == "clean":
        return select_clean(res, dups, rng, stats)
    quads, origin = {}, {}
    for q in QUADS:
        out, src = [], []
        if mode == "clean":
            buckets = [(None, res.get(q, []))]
        else:
            emos = HAPPY if q.startswith("pos") else SAD
            buckets = [(emos[i], res.get((q, i), [])) for i in range(len(emos)) if (q, i) in res]
        for emo, bucket in buckets:
            rng.shuffle(bucket)
            for key, text in bucket:
                if key in dups:
                    stats[f"sample_dropped_duplicate_{q}"] += 1
                    continue
                toks = normalize_text(text)
                if not valid_tokens(toks):
                    stats[f"sample_dropped_short_{q}"] += 1
                    continue
                out.append(toks)
                src.append(emo)
                if len(out) == NEED:
                    break
            if len(out) == NEED:
                break
        quads[q], origin[q] = out, src
    return quads, origin


def select_clean(res, dups, rng, stats):
    """Clean mode: besides exact duplicates of the raw text, drop every tweet
    whose text is identical to another candidate's after normalisation (all
    copies, as for raw duplicates). Tweets that differ only in the mention,
    link or emoji would otherwise appear twice, possibly under both labels or
    in both training and test."""
    pools, seen_norm = {}, Counter()
    for q in QUADS:
        bucket = res.get(q, [])
        rng.shuffle(bucket)
        pool = []
        for key, text in bucket:
            if key in dups:
                stats[f"sample_dropped_duplicate_{q}"] += 1
                continue
            toks = normalize_text(text)
            if not valid_tokens(toks):
                stats[f"sample_dropped_short_{q}"] += 1
                continue
            line = " ".join(toks)
            seen_norm[line] += 1
            pool.append(line)
        pools[q] = pool
    quads, origin = {}, {}
    for q in QUADS:
        out = []
        for line in pools[q]:
            if seen_norm[line] > 1:
                stats[f"sample_dropped_same_after_normalising_{q}"] += 1
                continue
            out.append(line.split(" "))
            if len(out) == NEED:
                break
        quads[q], origin[q] = out, [None] * len(out)
    return quads, origin


def overlap_between_splits(quads):
    """Test sentences that also occur in training (possible in faithful mode)."""
    train, test = set(), []
    for q in QUADS:
        train.update(" ".join(t) for t in quads[q][:TRAIN_PER_QUAD])
        test += [" ".join(t) for t in quads[q][TRAIN_PER_QUAD:TRAIN_PER_QUAD + TEST_PER_QUAD]]
    return sum(1 for t in test if t in train)


def emoji_composition(origin):
    """Faithful mode: which emojis the train and test portions come from."""
    comp = {}
    for q, src in origin.items():
        tr = Counter(src[:TRAIN_PER_QUAD])
        te = Counter(src[TRAIN_PER_QUAD:TRAIN_PER_QUAD + TEST_PER_QUAD])
        comp[q] = {"train": dict(tr.most_common()), "test": dict(te.most_common())}
    return comp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="TwitterAAE-full-v1.zip (read directly), or an unzipped file/folder")
    ap.add_argument("output", help="output folder, e.g. data/processed/sent_race")
    ap.add_argument("--mode", choices=["clean", "faithful"], default="clean")
    ap.add_argument("--max-lines", type=int, default=None, help="read only the first N lines (trial runs)")
    args = ap.parse_args()

    rng = random.Random(SEED)
    files = find_inputs(args.input)
    print("Reading:", ", ".join(describe(f) for f in files), flush=True)
    t0 = time.time()
    res, pop, hashes, emoji_counts, stats = scan(files, args.mode, args.max_lines, rng)
    print(f"Scan done: {stats['lines_read']:,} lines in {stats['seconds_scanning']}s", flush=True)
    dups = duplicated_keys(hashes)
    stats["distinct_texts_occurring_more_than_once"] = len(dups)
    del hashes
    quads, origin = select(res, args.mode, dups, rng, stats)

    os.makedirs(args.output, exist_ok=True)
    sizes = {}
    for q in QUADS:
        got = len(quads[q])
        sizes[q] = {"emoji_tweets_in_population": pop.get(q, 0), "written": got}
        if got < TRAIN_PER_QUAD + TEST_PER_QUAD:
            print(f"WARNING: {q} has only {got:,} usable tweets (paper split needs "
                  f"{TRAIN_PER_QUAD + TEST_PER_QUAD:,}).")
        with open(os.path.join(args.output, q + ".txt"), "w", encoding="utf-8") as fh:
            for toks in quads[q]:
                fh.write(" ".join(toks) + "\n")

    report = {
        "mode": args.mode,
        "input_files": [describe(f) for f in files],
        "max_lines": args.max_lines,
        "quadrants": sizes,
        "stats": dict(sorted(stats.items())),
        "test_sentences_also_in_train": overlap_between_splits(quads),
        "mean_tokens": {q: round(sum(map(len, quads[q])) / max(1, len(quads[q])), 2) for q in QUADS},
        "emoji_counts_highconf": {g: dict(c.most_common()) for g, c in emoji_counts.items()},
        "seconds": round(time.time() - t0, 1),
    }
    if args.mode == "faithful":
        report["emoji_composition_train_vs_test"] = emoji_composition(origin)
    with open(os.path.join(args.output, "data_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output, "examples.txt"), "w", encoding="utf-8") as fh:
        for q in QUADS:
            fh.write(f"== {q}\n" + "".join(" ".join(t) + "\n" for t in quads[q][:15]))
    short = {k: report[k] for k in ("quadrants", "stats", "test_sentences_also_in_train", "mean_tokens", "seconds")}
    print(json.dumps(short, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output}/ (quadrant files, data_report.json, examples.txt)")


if __name__ == "__main__":
    main()
