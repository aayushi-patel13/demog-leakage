"""Summarise the original-code runs (scripts/run_original.sh) beside the PyTorch port.

Reads the logs in results/original_code/ and, when present, the port's results on the same
faithful data (results/twitteraae_faithful/results.json, paper protocol: best epoch on test, as
the original code selects). Writes results/original_code/summary.md and summary.json.

Usage: python scripts/summarise_original.py [results/original_code]
"""
import json
import os
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/original_code"
PORT = os.path.join(os.path.dirname(OUT.rstrip("/")), "twitteraae_faithful", "results.json")
PAPER = {"sentiment": 67.4, "leakage": 64.5, "adv_task": 64.7, "adversary": 51.0, "adv_leakage": 56.0}


def read(name):
    path = os.path.join(OUT, name + ".txt")
    return open(path, encoding="utf-8", errors="replace").read() if os.path.exists(path) else ""


def best(text):
    """best_score / best_epoch lines written by the original scripts."""
    s = re.findall(r"best_score:\s*([0-9.]+)", text)
    e = re.findall(r"best_epoch:\s*(\d+)", text)
    return (100 * float(s[-1]), int(e[-1])) if s and e else (None, None)


def pct(x):
    return "-" if x is None else f"{x:.1f}"


base = read("baseline_sentiment")
adv = read("adversarial_lambda1")
base_curve = [(int(e), 100 * float(a)) for e, a in re.findall(r"dev-task epoch: (\d+), acc: ([0-9.]+)", base)]
adv_curve = [(int(e), 100 * float(t), 100 * float(a))
             for e, t, a in re.findall(r"^dev, (\d+), ([0-9.]+), ([0-9.]+)", adv, re.M)]

res = {}
res["sentiment"], res["sentiment_epoch"] = best(base)
res["adv_task"], res["adv_epoch"] = best(adv)
# an interrupted run never writes best_score; its best_model is the epoch with the best
# test accuracy so far (saved on a strict improvement), so read that from the curve
if res["sentiment"] is None and base_curve:
    e, a = max(base_curve, key=lambda r: r[1])
    res["sentiment"], res["sentiment_epoch"] = a, e
if res["adv_task"] is None and adv_curve:
    e, t_, a_ = max(adv_curve, key=lambda r: r[1])
    res["adv_task"], res["adv_epoch"] = t_, e
res["epochs_completed"] = {"baseline": len(base_curve), "adversarial": len(adv_curve)}
res["adversary"] = next((a for e, t, a in adv_curve if e == res["adv_epoch"]), None)
res["leakage"], res["leakage_attacker_epoch"] = best(read("attacker_baseline"))


def att_curve(text):
    """per-epoch test accuracy of an original attacker (dev, N, adv acc: X)"""
    return [(int(e), 100 * float(a)) for e, a in re.findall(r"^dev, (\d+), adv acc: ([0-9.]+)", text, re.M)]


base_att = att_curve(read("attacker_baseline"))
# the attacker on the adversarial model may have been cut short (a sleeping Codespace stops it);
# then use the longer of its logs and report the best epoch so far, with the number of epochs run
full = read("attacker_adversarial_best")
adv_att_text = full
if best(full)[0] is None and len(att_curve(read("attacker_adversarial_best_interrupted"))) > len(att_curve(full)):
    adv_att_text = read("attacker_adversarial_best_interrupted")
adv_att = att_curve(adv_att_text)
res["adv_leakage"], res["adv_leakage_attacker_epoch"] = best(adv_att_text)
if res["adv_leakage"] is None and adv_att:
    e, a = max(adv_att, key=lambda r: r[1])
    res["adv_leakage"], res["adv_leakage_attacker_epoch"] = a, e
res["attacker_epochs"] = {"baseline": len(base_att), "adversarial": len(adv_att)}
res["base_att_curve"], res["adv_att_curve"] = base_att, adv_att
res["adv_leakage_epoch10"], _ = best(read("attacker_adversarial_epoch10"))
res["adversary_epoch10"] = next((a for e, t, a in adv_curve if e == 10), None)
res["base_curve"], res["adv_curve"] = base_curve, adv_curve

port = {}
if os.path.exists(PORT):
    r = json.load(open(PORT))
    port = {"sentiment": r["E1_sentiment_only"]["test"]["test_acc"],
            "leakage": r["E3_leakage_no_adversary"]["paper_protocol"],
            "adv_task": r["E4_adv_lambda1"]["paper_protocol"]["task_acc"],
            "adversary": r["E4_adv_lambda1"]["paper_protocol"]["adversary_acc"],
            "adv_leakage": r["E4_adv_lambda1"]["paper_protocol"]["leakage"]}

rows = [("Sentiment only (main task)", "sentiment"),
        ("Leakage: attacker on the sentiment encoder", "leakage"),
        ("Adversarial training: main task", "adv_task"),
        ("Adversarial training: online adversary", "adversary"),
        ("Adversarial training: attacker on best model", "adv_leakage")]
lines = ["# Authors' original code (Python 2.7, DyNet) on the faithful TwitterAAE data", "",
         "Best epoch chosen on the test set, as the original code does; 10,000 test tweets; chance = 50.", "",
         "| | Paper | Original code | PyTorch port, same data |", "|---|---|---|---|"]
for label, k in rows:
    lines.append(f"| {label} | {pct(PAPER.get(k))} | {pct(res.get(k))} | {pct(port.get(k))} |")
if res["adv_leakage_epoch10"] is not None:
    lines.append(f"| Adversarial training: attacker on epoch 10 | - | {pct(res['adv_leakage_epoch10'])} | - |")
lines += ["", f"Epochs completed: baseline {len(base_curve)}, adversarial {len(adv_curve)}; "
          f"attacker epochs: on the sentiment encoder {len(base_att)}, on the adversarial encoder {len(adv_att)}. "
          f"Epochs chosen: sentiment {res['sentiment_epoch']}, adversarial {res['adv_epoch']}. "
          "The paper's adversary figure is its attacker score minus its reported gap of 5.0.", "",
          "Adversarial run, per epoch (test): " + ", ".join(f"epoch {e}: task {t:.1f}, adversary {a:.1f}"
                                                         for e, t, a in adv_curve)]
# epoch-matched port runs (scripts/port_epoch_match.sh): the port trained for exactly as many
# epochs as the original-code run completed, on the same data, attacked the same way
EM = os.environ.get("EPOCH_MATCH", "models/epoch_match")


def port_run(name):
    d = os.path.join(EM, name)
    try:
        ts = json.load(open(os.path.join(d, "train_summary.json")))
        at = json.load(open(os.path.join(d, "attack_z_enc-test.json")))
    except (OSError, ValueError):
        return None
    sel = ts["selected_by_test"]
    return {"epochs": ts["epochs"], "epoch": sel["epoch"], "task": sel["test_acc"],
            "adversary": sel["test_adv_acc"], "leakage": at["paper_protocol_max_test"],
            "attacker_curve": [(h["epoch"], h["test_acc"]) for h in at.get("history", [])]}


pb, pa = port_run("sentiment"), port_run("adversarial")
if pb or pa:
    nb, na = len(base_curve), len(adv_curve)
    g = lambda d, k: None if d is None else d.get(k)
    lines += ["", "## Epoch-matched comparison: same data, same number of epochs", "",
              "| | Original code | PyTorch port |", "|---|---|---|",
              f"| Sentiment only, best of first {nb} epoch(s) | {pct(res['sentiment'])} | {pct(g(pb, 'task'))} |",
              f"| Leakage from that encoder | {pct(res['leakage'])} | {pct(g(pb, 'leakage'))} |",
              f"| Adversarial, best of first {na} epoch(s): sentiment | {pct(res['adv_task'])} | {pct(g(pa, 'task'))} |",
              f"| Adversarial: online adversary | {pct(res['adversary'])} | {pct(g(pa, 'adversary'))} |",
              f"| Adversarial: attacker | {pct(res['adv_leakage'])} | {pct(g(pa, 'leakage'))} |",
              "", "Both select the best test epoch and report the attacker's best test epoch, as the original "
              "code does. The port trains on 156,000 tweets because it holds out a validation set; the "
              "original trains on 166,000. The 10,000 test tweets are the same."]
    # if the attacker on the adversarial model stopped early, compare every attacker over the same
    # number of attacker epochs, so a longer-trained attacker does not look like more leakage
    K = len(adv_att)
    if 0 < K < len(base_att):
        cap = lambda c: None if not c else max(a for e, a in c if e <= K)
        lines += ["", f"## Same attacker budget: best of the first {K} attacker epochs", "",
                  f"The attacker on the original adversarial model ran for {K} epochs, because the Codespace "
                  "stopped longer runs, so every attacker here is scored over its first "
                  f"{K} epochs.", "",
                  "| | Original code | PyTorch port |", "|---|---|---|",
                  f"| Leakage from the sentiment-only encoder | {pct(cap(base_att))} | {pct(cap(g(pb, 'attacker_curve')))} |",
                  f"| Attacker on the adversarial encoder | {pct(cap(adv_att))} | {pct(cap(g(pa, 'attacker_curve')))} |"]
        res["same_attacker_budget"] = {"epochs": K, "orig_leakage": cap(base_att), "orig_adv_leakage": cap(adv_att),
                                       "port_leakage": cap(g(pb, "attacker_curve")),
                                       "port_adv_leakage": cap(g(pa, "attacker_curve"))}
open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")
json.dump({"original": res, "port_same_data": port, "port_epoch_matched": {"sentiment": pb, "adversarial": pa}, "paper": PAPER}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("\n".join(lines))
