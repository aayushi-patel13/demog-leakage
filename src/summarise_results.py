"""One table of every result, with 95% confidence intervals, for the write-up.

Reads results/<tag>/results.json for the runs below and writes results/summary.md.
An accuracy p measured on n test examples is given as p ± 1.96·sqrt(p(1-p)/n);
n is taken from the data folder each run used. All numbers use held-out
selection (epochs chosen on validation, scored on test).

Usage: python src/summarise_results.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import make_splits  # noqa: E402

RUNS = [("twitteraae", "Tweets, race (clean data)"),
        ("twitteraae_faithful", "Tweets, race (original pipeline as written)"),
        ("reddit_tokens", "Reddit, NG vs US"),
        ("reddit_masked", "Reddit, NG vs US, topic words masked"),
        ("pan17_gb_us", "Tweets, GB vs US English (PAN 2017)")]
PAPER = {"y_alone": 67.4, "z_alone": 83.9, "leak": 64.5, "adv_y": 64.7, "adv_adv": None, "adv_leak": 56.0}


def n_test(data, cache={}):
    if data not in cache:
        try:
            cache[data] = len(make_splits(data)["test"])
        except (FileNotFoundError, OSError):
            cache[data] = None
    return cache[data]


def e4_epoch(tag):
    """Epoch at which the adversarially trained encoder was chosen (best validation
    accuracy, first one if tied), read from the run's log."""
    path = os.path.join("results", tag, "log.txt")
    if not os.path.exists(path):
        return None
    lines = open(path, encoding="utf-8").read().splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("== E4")]
    if not starts:
        return None
    best, ep = -1, None
    for ln in lines[starts[-1]:]:
        ln = ln.strip()
        if ln.startswith("{") and '"val_acc"' in ln:
            r = json.loads(ln)
            if r["val_acc"] > best:
                best, ep = r["val_acc"], r["epoch"]
    return ep


def ci(p, n):
    if p is None:
        return "-"
    if not n:
        return f"{p:.1f}"
    h = 196 * math.sqrt((p / 100) * (1 - p / 100) / n)
    return f"{p:.1f} ± {h:.1f}"


def main():
    md = ["# Results summary", "",
          "Held-out selection throughout; ± is a 95% interval for each accuracy (chance = 50.0).", ""]
    cols, rows = [], {}
    labels = [("y_alone", "Main task, encoder trained alone"),
              ("z_alone", "Protected attribute, encoder trained for it"),
              ("leak", "Leakage: attacker on the main-task encoder"),
              ("adv_y", "Adversarial training: main task"),
              ("adv_adv", "Adversarial training: online adversary"),
              ("adv_leak", "Adversarial training: post-hoc attacker (leakage)")]
    ns = []
    for tag, name in RUNS:
        p = os.path.join("results", tag, "results.json")
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        n = n_test(R["data"])
        e4 = R["E4_adv_lambda1"]["heldout"]
        vals = {"y_alone": R["E1_sentiment_only"]["val"]["test_acc"],
                "z_alone": R["E2_race_only"]["val"]["test_acc"],
                "leak": R["E3_leakage_no_adversary"]["heldout"],
                "adv_y": e4["task_acc"], "adv_adv": e4["adversary_acc"], "adv_leak": e4["leakage"]}
        cols.append(name)
        ns.append(f"{name}: {n:,} test examples" if n else f"{name}: test size unknown")
        for k, _ in labels:
            rows.setdefault(k, []).append(ci(vals[k], n))
        ep = e4_epoch(tag)
        rows.setdefault("epoch", []).append(f"{ep} of {R['epochs']}" if ep else "-")
    if cols:
        md += ["| | Paper | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for k, lab in labels:
            pv = "-" if PAPER[k] is None else f"{PAPER[k]:.1f}"
            md.append(f"| {lab} | {pv} | " + " | ".join(rows[k]) + " |")
        md.append("| Adversarial training: epoch chosen on validation | - | " + " | ".join(rows["epoch"]) + " |")
        md += ["", "Test sizes: " + "; ".join(ns) + ".", ""]
    p = os.path.join("results", "twitteraae_advcheck", "results.json")
    if os.path.exists(p):
        A = json.load(open(p))
        n = n_test(A["data"])
        md += ["## Adversarial training, repeated (tweets, clean data, lambda = 1)", "",
               "| Run | Epochs | Main task | Online adversary | Post-hoc attacker (leakage) |",
               "|---|---|---|---|---|"]
        main_ep = e4_epoch("twitteraae")
        for r in A["rows"]:
            if r["run"].startswith("seed 16 (main run)") and main_ep:
                r["epochs"] = f"20 (best on val: {main_ep})"
            md.append(f"| {r['run']} | {r['epochs']} | {ci(r['task_acc'], n)} | {ci(r['adversary_acc'], n)} | "
                      f"{ci(r['leakage'], n)} |")
        s = A.get("seed_summary") or {}
        if s:
            md += ["", f"Across {s['leakage']['n']} seeds: leakage {s['leakage']['mean']:.1f} "
                   f"(sd {s['leakage']['sd']:.1f}), online adversary {s['adversary_acc']['mean']:.1f} "
                   f"(sd {s['adversary_acc']['sd']:.1f}), main task {s['task_acc']['mean']:.1f} "
                   f"(sd {s['task_acc']['sd']:.1f})."]
        md.append("")
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("(written to results/summary.md)")


if __name__ == "__main__":
    main()
