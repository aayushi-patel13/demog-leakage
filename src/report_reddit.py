"""Summarise the Reddit dataset for the write-up: collection, cleaning, labels,
balanced quadrants and (once scored) the human annotation.

Reads clean_report.json, label_report.json, the collection logs and, if present,
annotation/sample_annotated_scores.json. Writes results/reddit_data.md.

Usage: python src/report_reddit.py [--clean DIR] [--processed DIR] [--annotation DIR]
"""
import argparse
import glob
import json
import os
import re
from collections import Counter


def load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/reddit/clean")
    ap.add_argument("--processed", default="data/reddit/processed")
    ap.add_argument("--annotation", default="annotation")
    ap.add_argument("--logs", default="data/reddit")
    ap.add_argument("--out", default="results/reddit_data.md")
    args = ap.parse_args()
    c = load(os.path.join(args.clean, "clean_report.json"))
    lab = load(os.path.join(args.processed, "label_report.json"))
    md = ["## Reddit: US vs Nigerian English", ""]
    if c:
        md += [f"Comments created before {c.get('until') or 'the end of the window'} "
               "(2024-01-01 onwards); both groups cover the same period.", "",
               "| Subreddit | Collected | Covers | Kept after cleaning |", "|---|---:|---|---:|"]
        span = c.get("collected_span", {})
        for k, n in sorted(c["raw_per_subreddit"].items()):
            a, b = span.get(k, ["", ""])
            md.append(f"| r/{k.split('/')[1]} ({k.split('/')[0].upper()}) | {n:,} | {a} to {b} | "
                      f"{c.get('final_per_subreddit', {}).get(k, 0):,} |")
        md += ["", "Removed during cleaning:", ""]
        md += [f"* {k.replace('_', ' ')}: {v:,}" for k, v in sorted(c["dropped"].items(), key=lambda x: -x[1])]
        md += ["", "| Group | Comments | Authors | Median tokens | With a sentiment emoji | 2+ Pidgin markers |",
               "|---|---:|---:|---:|---:|---:|"]
        for g, v in sorted(c["final_per_group"].items()):
            md.append(f"| {g.upper()} | {v['comments']:,} | {v['authors']:,} | {v['median_tokens']} | "
                      f"{v['with_sentiment_emoji']:,} ({100 * v['with_sentiment_emoji'] / v['comments']:.1f}%) | "
                      f"{v['pidgin_2plus_markers']:,} ({100 * v['pidgin_2plus_markers'] / v['comments']:.1f}%) |")
        md.append("")
    skips = Counter()
    for f in glob.glob(os.path.join(args.logs, "collect_*.log")):
        for ln in open(f, encoding="utf-8", errors="replace"):
            m = re.match(r"\s+r/(\w+): skipping ", ln)
            if m:
                skips[m.group(1)] += 1
    if c:   # only subreddits that are part of the dataset
        used = {k.split("/")[1] for k in c["raw_per_subreddit"]}
        skips = Counter({s: n for s, n in skips.items() if s in used})
    if skips:
        md += ["One-hour windows the archive refused even after retries (skipped): " +
               ", ".join(f"r/{s} {n}" for s, n in sorted(skips.items())) + ".", ""]
    if lab:
        src = Counter()
        for k, v in lab["label_source"].items():
            g, s, how = k.split("/")
            src[(g, how)] += v
        md += ["Sentiment labels (before balancing): " + ", ".join(
            f"{g.upper()} {how} {n:,}" for (g, how), n in sorted(src.items())) +
            f"; comments with both happy and sad emojis dropped: {lab['conflicting_emojis']:,}.", "",
            "| Quadrant | Train | Test (held-out authors) |", "|---|---:|---:|"]
        for q, v in lab["quadrants"].items():
            md.append(f"| {q} | {v['train']:,} | {v['test']:,} |")
        qs = lab.get("quadrant_label_sources")
        src_note = ""
        if qs:
            one = next(iter(qs.values()))
            src_note = (" and on label source (each quadrant: " +
                        ", ".join(f"{n:,} {k}" for k, n in sorted(one.items())) + ")")
        md += ["", f"Total {lab['total']:,} comments, balanced across the four quadrants and matched on "
               f"length (3-5, 6-10, 11-20, 21-40 tokens){src_note}.", ""]
    for fname, title in (("sample_annotated_scores.json", "Human annotation (blind sample)"),
                         ("sample_llm_judge_scores.json",
                          "LLM judge (Claude, zero-shot, text only, blind to labels; see annotation/LLM_JUDGE.md)")):
        sc = load(os.path.join(args.annotation, fname))
        if not sc:
            continue
        who, Who = ("LLM", "LLM") if "llm" in fname else ("human", "Human")
        md += [f"### {title}", ""]
        if "sentiment_agreement_pct" in sc:
            by = sc.get("agreement_by_label_source", {})
            md += [f"* Distant sentiment label matches the {who} label in {sc['sentiment_agreement_pct']}% of "
                   f"{sc['annotated_sentiment']} comments (Cohen's kappa {sc['sentiment_kappa']}); the {who} "
                   f"called {sc['human_said_neutral_pct']}% neutral. By source: " +
                   ", ".join(f"{s} {v['agreement_pct']}% (n={v['n']})" for s, v in by.items()) + "."]
        if "variety_accuracy_unsure_as_chance_pct" in sc:
            md += [f"* {Who} guess of NG vs US from the text alone: "
                   f"{sc.get('variety_accuracy_when_decided_pct', '-')}% correct when decided, "
                   f"{sc['variety_unsure_pct']}% unsure; {sc['variety_accuracy_unsure_as_chance_pct']}% "
                   "counting unsure as a coin flip (comparable to an attacker's accuracy, chance = 50%)."]
        md.append("")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"(written to {args.out})")


if __name__ == "__main__":
    main()
