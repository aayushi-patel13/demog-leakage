"""Summarise the prepared TwitterAAE datasets for the write-up and slides.

Reads data/processed/sent_race (clean) and sent_race_faithful (the original
preprocessing as written), and reports, per quadrant: tweets available in the
corpus, tweets written, distinct tweets among them, tweets that also appear
with the opposite sentiment label, and the train/validation/test sizes the
experiments will use. For the faithful data it also shows which emojis the
training and test portions come from. Writes results/twitteraae_data.md.

Usage: python src/report_twitteraae.py [--clean DIR] [--faithful DIR] [--out FILE]
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from data import QUADS, make_splits, read_lines  # noqa: E402

PAPER_PER_QUAD = 41500 + 2500
OPPOSITE = {"pos_aa": "neg_aa", "neg_aa": "pos_aa", "pos_wh": "neg_wh", "neg_wh": "pos_wh"}


def lines(folder, q):
    return [" ".join(ln.split()) for ln in read_lines(os.path.join(folder, q + ".txt"))]


def describe(folder, title):
    rep = json.load(open(os.path.join(folder, "data_report.json"), encoding="utf-8"))
    rows = {q: lines(folder, q) for q, _, _ in QUADS}
    splits = make_splits(folder)
    size = {k: Counter((y, z) for _, y, z in v) for k, v in splits.items()}
    yz = {q: (y, z) for q, y, z in QUADS}
    out = [f"### {title} (`{folder}`)", "",
           "| Quadrant | Available in corpus | Written | Distinct | Also under the opposite label "
           "| Train / val / test |",
           "|---|---:|---:|---:|---:|---|"]
    for q, _, _ in QUADS:
        avail = rep["quadrants"][q]["emoji_tweets_in_population"]
        written = len(rows[q])
        distinct = len(set(rows[q]))
        both = len(set(rows[q]) & set(rows[OPPOSITE[q]]))
        tvt = " / ".join(f"{size[k][yz[q]]:,}" for k in ("train", "val", "test"))
        flag = " (short of 44,000)" if written < PAPER_PER_QUAD else ""
        out.append(f"| {q} | {avail:,} | {written:,}{flag} | {distinct:,} | {both:,} | {tvt} |")
    out += ["", f"Test sentences also in training (paper split): {rep['test_sentences_also_in_train']:,}. "
            f"Mean tokens per tweet: " + ", ".join(f"{q} {v}" for q, v in rep["mean_tokens"].items()) + ".", ""]
    comp = rep.get("emoji_composition_train_vs_test")
    if comp:
        out += ["As in the original code, a tweet is written once per emoji it contains, so "
                "*Written* can exceed *Available* (which counts each tweet once), and tweets with "
                "both happy and sad emojis are written under both labels.", ""]
        out += ["Where the training and test portions come from (top emojis, paper split):", "",
                "| Quadrant | Training (first 41,500) | Test (next 2,500) |", "|---|---|---|"]
        for q, c in comp.items():
            top = lambda d: ", ".join(f"{e} {n:,}" for e, n in list(d.items())[:3]) or "none"
            out.append(f"| {q} | {top(c['train'])} | {top(c['test'])} |")
        out.append("")
    return out, rep


def quirks(rep):
    st = rep["stats"]
    out = ["### Quirks of the original preprocessing, counted on the full corpus", "",
           "High-confidence tweets (posterior > 0.8) with at least one happy or sad emoji.", "",
           "| | AAE group | White-aligned group |", "|---|---:|---:|"]

    def row(label, key):
        a, w = st.get(f"{key}_aa", 0), st.get(f"{key}_wh", 0)
        pa = 100 * a / max(1, st.get("highconf_emoji_aa", 0))
        pw = 100 * w / max(1, st.get("highconf_emoji_wh", 0))
        out.append(f"| {label} | {a:,} ({pa:.1f}%) | {w:,} ({pw:.1f}%) |")
    out.append(f"| emoji tweets | {st.get('highconf_emoji_aa', 0):,} | {st.get('highconf_emoji_wh', 0):,} |")
    row("happy emojis only", "happy_only")
    row("sad emojis only", "sad_only")
    row("both happy and sad (enter both classes in the original)", "quirk_both_happy_and_sad")
    row("two or more emojis of one class (counted twice in the original)",
        "quirk_two_or_more_emojis_same_class")
    row("sad only through the sob emoji (never collected in the original)", "quirk_sad_only_via_sob")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/processed/sent_race")
    ap.add_argument("--faithful", default="data/processed/sent_race_faithful")
    ap.add_argument("--out", default="results/twitteraae_data.md")
    args = ap.parse_args()
    md = ["## TwitterAAE: the replication data", "",
          f"The paper uses {PAPER_PER_QUAD:,} tweets per quadrant (41,500 train + 2,500 test).", ""]
    rep = None
    for folder, title in ((args.clean, "Clean preprocessing (main replication)"),
                          (args.faithful, "Original preprocessing as written (ablation)")):
        if os.path.exists(os.path.join(folder, "data_report.json")):
            part, rep = describe(folder, title)
            md += part
    if rep:
        md += quirks(rep)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"(written to {args.out})")


if __name__ == "__main__":
    main()
