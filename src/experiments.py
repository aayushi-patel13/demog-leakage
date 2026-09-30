"""Train, attack and tabulate: the replication experiments of Elazar & Goldberg (2018).

Subcommands
  train    train an encoder, either on one task (lambda = 0) or adversarially
  attack   freeze a trained encoder and train a fresh post-hoc attacker on h
  run-all  run the paper's four balanced experiments and write a results table

Optimisation follows the original DyNet code: momentum SGD (lr 0.01, momentum
0.9), losses summed over mini-batches of 32, gradient-norm clipping at 5 (DyNet's
default), dropout 0.2. The original trained for 100 epochs and kept the epoch
with the best *test* accuracy; we log every epoch and report both that
"paper protocol" number and an honest one where the epoch is chosen on a
separate validation set.

Examples
  python src/experiments.py run-all --data data/processed/sent_race --epochs 20 --tag twitteraae
  python src/experiments.py train --data data/processed/sent_race --target y --lambd 1.0 --out models/adv_l1
  python src/experiments.py attack --data data/processed/sent_race --model models/adv_l1
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Vocab, batches, make_splits  # noqa: E402
from models import MLP, AdvModel  # noqa: E402
from optim import DyNetStyleSGD  # noqa: E402

PAPER = {  # Elazar & Goldberg (2018), balanced DIAL sentiment/race, Tables 1-3
    "sentiment_only_acc": 67.4,
    "race_only_acc": 83.9,
    "leakage_no_adversary": 64.5,
    "adv_task_acc": 64.7,
    "adv_leakage": 56.0,
    "adv_delta": 5.0,
}


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(folder, splits_cache={}):
    if folder not in splits_cache:
        s = make_splits(folder)
        vocab = Vocab([t for part in s.values() for t, _, _ in part])
        splits_cache[folder] = (s, vocab)
    return splits_cache[folder]


def sgd(params, lr):
    return torch.optim.SGD(params, lr=lr, momentum=0.9)


@torch.no_grad()
def evaluate(model, rows, vocab, dev, target, lambd, bs=512):
    model.eval()
    ok_t = ok_a = n = 0
    for x, lens, y, z in batches(rows, vocab, bs, shuffle=False):
        x, lens, y, z = x.to(dev), lens.to(dev), y.to(dev), z.to(dev)
        h, logits, _ = model(x, lens, 0.0)
        gold = y if target == "y" else z
        ok_t += (logits.argmax(1) == gold).sum().item()
        if lambd > 0:
            adv_logits = model.advs[0](h)
            ok_a += (adv_logits.argmax(1) == z).sum().item()
        n += len(y)
    return 100 * ok_t / n, (100 * ok_a / n if lambd > 0 else None)


def train(data, target, lambd, epochs, out, n_adv=1, batch_size=32, lr=0.01, seed=16, max_train=None, log=print):
    set_seed(seed)
    dev = device()
    splits, vocab = load_data(data)
    train_rows = splits["train"][:max_train] if max_train else splits["train"]
    model = AdvModel(len(vocab), n_adv=n_adv if lambd > 0 else 0).to(dev)
    opt = DyNetStyleSGD(model, lr=lr, momentum=0.9, clip=5.0)
    ce = nn.CrossEntropyLoss(reduction="sum")
    os.makedirs(out, exist_ok=True)
    hist, best = [], {"test": (-1, None), "val": (-1, None)}
    log(f"[train] target={target} lambda={lambd} epochs={epochs} train={len(train_rows):,} "
        f"val={len(splits['val']):,} test={len(splits['test']):,} vocab={len(vocab):,} device={dev}")
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, lens, y, z in batches(train_rows, vocab, batch_size, shuffle=True, seed=seed + ep):
            x, lens, y, z = x.to(dev), lens.to(dev), y.to(dev), z.to(dev)
            _, logits, adv_logits = model(x, lens, lambd)
            loss = ce(logits, y if target == "y" else z)
            for al in adv_logits:
                loss = loss + ce(al, z)
            opt.zero_grad()
            loss.backward()
            opt.step()  # includes DyNet-style clipping at norm 5
            tot += loss.item()
        val_acc, val_adv = evaluate(model, splits["val"], vocab, dev, target, lambd)
        test_acc, test_adv = evaluate(model, splits["test"], vocab, dev, target, lambd)
        rec = {"epoch": ep, "loss": tot / len(train_rows), "val_acc": val_acc, "test_acc": test_acc,
               "val_adv_acc": val_adv, "test_adv_acc": test_adv, "secs": round(time.time() - t0, 1)}
        hist.append(rec)
        log("  " + json.dumps(rec))
        for key, score in (("test", test_acc), ("val", val_acc)):
            if score > best[key][0]:
                best[key] = (score, ep)
                torch.save(model.state_dict(), os.path.join(out, f"best_by_{key}.pt"))
    with open(os.path.join(out, "vocab.json"), "w") as fh:
        json.dump(vocab.itos, fh)
    summary = {"data": data, "target": target, "lambda": lambd, "n_adv": n_adv if lambd > 0 else 0,
               "epochs": epochs, "history": hist}
    for key in ("test", "val"):
        ep = best[key][1]
        r = hist[ep - 1]
        summary[f"selected_by_{key}"] = {"epoch": ep, "test_acc": r["test_acc"], "test_adv_acc": r["test_adv_acc"]}
    with open(os.path.join(out, "train_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


@torch.no_grad()
def encode_all(model, rows, vocab, dev, bs=512):
    model.eval()
    H, Y, Z = [], [], []
    for x, lens, y, z in batches(rows, vocab, bs, shuffle=False):
        H.append(model.encoder(x.to(dev), lens.to(dev)).cpu())
        Y.append(y)
        Z.append(z)
    return torch.cat(H), torch.cat(Y), torch.cat(Z)


def attack(data, model_dir, which="test", target="z", epochs=100, batch_size=32, lr=0.01, seed=16, log=print):
    """Train a fresh MLP on frozen representations to predict the protected attribute."""
    set_seed(seed)
    dev = device()
    splits, vocab = load_data(data)
    summ = json.load(open(os.path.join(model_dir, "train_summary.json")))
    model = AdvModel(len(vocab), n_adv=summ["n_adv"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"best_by_{which}.pt"), map_location=dev))
    reps = {k: encode_all(model, splits[k], vocab, dev) for k in ("train", "val", "test")}
    col = 2 if target == "z" else 1
    att = MLP().to(dev)
    opt = sgd(att.parameters(), lr)
    ce = nn.CrossEntropyLoss(reduction="sum")
    Htr, Ttr = reps["train"][0], reps["train"][col]
    g = torch.Generator().manual_seed(seed)
    hist = []
    for ep in range(1, epochs + 1):
        att.train()
        perm = torch.randperm(len(Htr), generator=g)
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            loss = ce(att(Htr[b].to(dev)), Ttr[b].to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(att.parameters(), 5.0)
            opt.step()
        att.eval()
        accs = {}
        for k in ("val", "test"):
            H, T = reps[k][0].to(dev), reps[k][col].to(dev)
            accs[k] = 100 * (att(H).argmax(1) == T).float().mean().item()
        hist.append({"epoch": ep, "val_acc": accs["val"], "test_acc": accs["test"]})
        if ep % 10 == 0 or ep == 1:
            log(f"  attacker epoch {ep}: val {accs['val']:.2f}  test {accs['test']:.2f}")
    best_test = max(hist, key=lambda r: r["test_acc"])
    best_val = max(hist, key=lambda r: r["val_acc"])
    res = {"model": model_dir, "encoder_selected_by": which, "target": target, "epochs": epochs,
           "paper_protocol_max_test": best_test["test_acc"], "paper_protocol_epoch": best_test["epoch"],
           "heldout_test_at_best_val": best_val["test_acc"], "heldout_epoch": best_val["epoch"],
           "history": hist}
    with open(os.path.join(model_dir, f"attack_{target}_enc-{which}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def run_all(data, epochs, tag, att_epochs=100, max_train=None, lambdas=(1.0,), attr="race", paper=True):
    out = os.path.join("models", tag)
    res_dir = os.path.join("results", tag)
    os.makedirs(res_dir, exist_ok=True)
    logf = open(os.path.join(res_dir, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    R = {"data": data, "epochs": epochs, "attacker_epochs": att_epochs, "attribute": attr,
         "paper": PAPER if paper else None}
    log("== E1: encoder trained on sentiment only")
    s1 = train(data, "y", 0.0, epochs, f"{out}/sentiment_only", max_train=max_train, log=log)
    log(f"== E2: encoder trained on {attr} only")
    s2 = train(data, "z", 0.0, epochs, f"{out}/race_only", max_train=max_train, log=log)
    log("== E3: post-hoc attacker on the sentiment-only encoder (leakage, no defence)")
    a1 = {w: attack(data, f"{out}/sentiment_only", which=w, epochs=att_epochs, log=log) for w in ("test", "val")}
    R["E1_sentiment_only"] = {k: s1[f"selected_by_{k}"] for k in ("test", "val")}
    R["E2_race_only"] = {k: s2[f"selected_by_{k}"] for k in ("test", "val")}
    R["E3_leakage_no_adversary"] = {"paper_protocol": a1["test"]["paper_protocol_max_test"],
                                    "heldout": a1["val"]["heldout_test_at_best_val"]}
    for lam in lambdas:
        name = f"adv_lambda{lam:g}"
        log(f"== E4: adversarial training, lambda = {lam:g}")
        s = train(data, "y", lam, epochs, f"{out}/{name}", max_train=max_train, log=log)
        a = {w: attack(data, f"{out}/{name}", which=w, epochs=att_epochs, log=log) for w in ("test", "val")}
        pp, ho = s["selected_by_test"], s["selected_by_val"]
        R[f"E4_{name}"] = {
            "paper_protocol": {"task_acc": pp["test_acc"], "adversary_acc": pp["test_adv_acc"],
                               "leakage": a["test"]["paper_protocol_max_test"],
                               "delta": a["test"]["paper_protocol_max_test"] - pp["test_adv_acc"]},
            "heldout": {"task_acc": ho["test_acc"], "adversary_acc": ho["test_adv_acc"],
                        "leakage": a["val"]["heldout_test_at_best_val"],
                        "delta": a["val"]["heldout_test_at_best_val"] - ho["test_adv_acc"]},
        }
    R["minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(res_dir, "results.json"), "w") as fh:
        json.dump(R, fh, indent=2)
    table = results_table(R, lambdas)
    with open(os.path.join(res_dir, "results.md"), "w") as fh:
        fh.write(table)
    log("\n" + table)
    return R


def results_table(R, lambdas=(1.0,)):
    f = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    P = R.get("paper") or {k: None for k in PAPER}
    attr = R.get("attribute", "race")
    rows = [
        ("Sentiment, encoder trained alone (acc)", P["sentiment_only_acc"],
         R["E1_sentiment_only"]["test"]["test_acc"], R["E1_sentiment_only"]["val"]["test_acc"]),
        (f"{attr[:1].upper() + attr[1:]}, encoder trained alone (acc)", P["race_only_acc"],
         R["E2_race_only"]["test"]["test_acc"], R["E2_race_only"]["val"]["test_acc"]),
        ("Leakage: attacker on sentiment encoder", P["leakage_no_adversary"],
         R["E3_leakage_no_adversary"]["paper_protocol"], R["E3_leakage_no_adversary"]["heldout"]),
    ]
    for lam in lambdas:
        e = R[f"E4_adv_lambda{lam:g}"]
        p = lam == 1.0
        rows += [
            (f"Adversarial (lambda={lam:g}): sentiment acc", P["adv_task_acc"] if p else None,
             e["paper_protocol"]["task_acc"], e["heldout"]["task_acc"]),
            (f"Adversarial (lambda={lam:g}): online adversary acc", None,
             e["paper_protocol"]["adversary_acc"], e["heldout"]["adversary_acc"]),
            (f"Adversarial (lambda={lam:g}): post-hoc attacker (leakage)", P["adv_leakage"] if p else None,
             e["paper_protocol"]["leakage"], e["heldout"]["leakage"]),
            (f"Adversarial (lambda={lam:g}): delta = attacker - adversary", P["adv_delta"] if p else None,
             e["paper_protocol"]["delta"], e["heldout"]["delta"]),
        ]
    out = [f"Results ({R['data']}, {R['epochs']} encoder epochs, {R['attacker_epochs']} attacker epochs; "
           "chance = 50.0)", "",
           "| Measurement | Paper | Ours, paper protocol | Ours, held-out selection |",
           "|---|---|---|---|"]
    out += [f"| {a} | {f(b)} | {f(c)} | {f(d)} |" for a, b, c, d in rows]
    out += ["", "Paper protocol: best epoch chosen on the test set, as in the original code.",
            "Held-out selection: epoch chosen on a separate validation set, then scored on test."]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--target", choices=["y", "z"], default="y")
    t.add_argument("--lambd", type=float, default=0.0)
    t.add_argument("--n-adv", type=int, default=1)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--out", required=True)
    t.add_argument("--max-train", type=int, default=None)
    a = sub.add_parser("attack")
    a.add_argument("--data", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--which", choices=["test", "val"], default="test")
    a.add_argument("--epochs", type=int, default=100)
    r = sub.add_parser("run-all")
    r.add_argument("--data", required=True)
    r.add_argument("--epochs", type=int, default=20)
    r.add_argument("--attacker-epochs", type=int, default=100)
    r.add_argument("--tag", required=True)
    r.add_argument("--max-train", type=int, default=None, help="subsample training (quick tests)")
    r.add_argument("--lambdas", type=float, nargs="+", default=[1.0])
    r.add_argument("--attribute", default="race", help="name of the protected attribute, for the table")
    r.add_argument("--no-paper", action="store_true", help="new data: leave out the paper's numbers")
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.data, args.target, args.lambd, args.epochs, args.out, n_adv=args.n_adv, max_train=args.max_train)
    elif args.cmd == "attack":
        res = attack(args.data, args.model, which=args.which, epochs=args.epochs)
        print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))
    else:
        run_all(args.data, args.epochs, args.tag, args.attacker_epochs, args.max_train, tuple(args.lambdas),
                attr=args.attribute, paper=not args.no_paper)


if __name__ == "__main__":
    main()
