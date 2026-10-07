"""Draw a blind annotation sample from the Reddit dataset, and score it.

  python src/annotation.py sample --n 50          # 50 per quadrant -> 200 comments
  python src/annotation.py score  annotation/sample_for_annotation.csv

`sample` writes two files:
  annotation/sample_for_annotation.csv  what the annotator sees: the comment
      text (with emojis) and empty columns human_sentiment (pos/neg/neutral)
      and human_variety (NG/US/unsure). It shows no labels, so the annotation
      is blind to the automatic labels and to the subreddit.
  annotation/sample_key.csv  the hidden answers: distant sentiment label and
      its source (emoji / vader), group (ng/us), subreddit.

`score` joins the filled sheet with the key and reports
  * how often the distant sentiment label matches the human one (overall, and
    separately for emoji- and VADER-based labels), with Cohen's kappa;
  * how often a human reader can tell Nigerian from US English from the text
    alone: a human, text-level reference point for the leakage numbers.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

QUADS = ["pos_ng", "pos_us", "neg_ng", "neg_us"]


def load_clean(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        out[c["id"]] = c
    return out


def sample(args):
    comments = load_clean(args.clean)
    rng = random.Random(args.seed)
    items = []
    for q in QUADS:
        ids = [json.loads(l) for l in open(os.path.join(args.processed, q + ".ids.jsonl"))]
        pick = rng.sample(ids, min(args.n, len(ids)))
        sent, group = q.split("_")
        for r in pick:
            c = comments[r["id"]]
            items.append({"id": r["id"], "text": c["text"], "distant_sentiment": sent,
                          "label_source": r["label_source"], "group": group, "subreddit": c["subreddit"]})
    rng.shuffle(items)
    os.makedirs(args.out, exist_ok=True)
    # utf-8-sig: Excel then shows emojis and accents correctly
    with open(os.path.join(args.out, "sample_for_annotation.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "id", "text", "human_sentiment", "human_variety", "notes"])
        for i, it in enumerate(items, 1):
            w.writerow([i, it["id"], it["text"], "", "", ""])
    with open(os.path.join(args.out, "sample_key.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "distant_sentiment", "label_source", "group", "subreddit"])
        w.writeheader()
        for it in items:
            w.writerow({k: it[k] for k in w.fieldnames})
    print(f"wrote {len(items)} items to {args.out}/sample_for_annotation.csv (key kept separately)")


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(args):
    key = {r["id"]: r for r in csv.DictReader(open(args.key, encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(args.sheet, encoding="utf-8-sig"))]
    norm_s = {"pos": "pos", "positive": "pos", "p": "pos", "neg": "neg", "negative": "neg", "n": "neg",
              "neutral": "neutral", "neu": "neutral", "mixed": "neutral", "": None}
    norm_v = {"ng": "ng", "nigeria": "ng", "nigerian": "ng", "us": "us", "usa": "us", "american": "us",
              "unsure": "unsure", "?": "unsure", "": None}
    sent_pairs, by_src, var_rows = [], defaultdict(list), []
    for r in rows:
        k = key.get(r["id"])
        if not k:
            continue
        hs = norm_s.get(r["human_sentiment"].strip().lower())
        hv = norm_v.get(r["human_variety"].strip().lower())
        if hs:
            sent_pairs.append((k["distant_sentiment"], hs))
            by_src[k["label_source"]].append((k["distant_sentiment"], hs))
        if hv:
            var_rows.append((k["group"], hv))
    res = {"annotated_sentiment": len(sent_pairs), "annotated_variety": len(var_rows)}
    if sent_pairs:
        res["sentiment_agreement_pct"] = round(100 * sum(a == b for a, b in sent_pairs) / len(sent_pairs), 1)
        res["sentiment_kappa"] = round(kappa([a for a, _ in sent_pairs], [b for _, b in sent_pairs]), 3)
        res["human_said_neutral_pct"] = round(100 * sum(b == "neutral" for _, b in sent_pairs) / len(sent_pairs), 1)
        res["agreement_by_label_source"] = {
            s: {"n": len(p), "agreement_pct": round(100 * sum(a == b for a, b in p) / len(p), 1)}
            for s, p in by_src.items()}
    if var_rows:
        decided = [(g, h) for g, h in var_rows if h != "unsure"]
        res["variety_unsure_pct"] = round(100 * sum(h == "unsure" for _, h in var_rows) / len(var_rows), 1)
        if decided:
            res["variety_accuracy_when_decided_pct"] = round(100 * sum(g == h for g, h in decided) / len(decided), 1)
        # counting "unsure" as a coin flip gives a number comparable to attacker accuracy
        res["variety_accuracy_unsure_as_chance_pct"] = round(
            100 * (sum(g == h for g, h in var_rows) + 0.5 * sum(h == "unsure" for _, h in var_rows)) / len(var_rows), 1)
    out = os.path.splitext(args.sheet)[0] + "_scores.json"
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--clean", default="data/reddit/clean/comments.jsonl")
    s.add_argument("--processed", default="data/reddit/processed/tokens")
    s.add_argument("--out", default="annotation")
    s.add_argument("--n", type=int, default=50, help="comments per quadrant")
    s.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("score")
    c.add_argument("sheet")
    c.add_argument("--key", default="annotation/sample_key.csv")
    args = ap.parse_args()
    sample(args) if args.cmd == "sample" else score(args)


if __name__ == "__main__":
    main()
