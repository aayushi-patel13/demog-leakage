"""Data loading for the four-quadrant datasets (TwitterAAE or Reddit).

A dataset folder holds pos_aa.txt, pos_wh.txt, neg_aa.txt, neg_wh.txt, one
tokenised text per line. For the Reddit data "aa" = Nigerian English and
"wh" = US English; the code is shared.

Splits follow the original data_handler.get_labeled_data(): per quadrant the
first `train` lines are training, the next `test` lines are test. Anything
after that (up to `val` lines) is an extra validation set used for model
selection; if it is missing, validation is carved from the end of training.
Labels: y (main task) pos = 1, neg = 0; z (protected) aa = 1, wh = 0.
"""
import json
import os
import random

import torch

QUADS = [("pos_aa", 1, 1), ("pos_wh", 1, 0), ("neg_aa", 0, 1), ("neg_wh", 0, 0)]
PAD, UNK = "<pad>", "<unk>"


# z = 1 files: aa (TwitterAAE), ng (Reddit Nigerian), z1 (PAN17); z = 0: wh, us, z0
ALIASES = {"pos_aa": ["pos_aa", "pos_ng", "pos_z1"], "pos_wh": ["pos_wh", "pos_us", "pos_z0"],
           "neg_aa": ["neg_aa", "neg_ng", "neg_z1"], "neg_wh": ["neg_wh", "neg_us", "neg_z0"]}


def quad_file(folder, name):
    for alias in ALIASES[name]:
        path = os.path.join(folder, alias + ".txt")
        if os.path.exists(path):
            return path, alias
    raise FileNotFoundError(f"no {ALIASES[name]} file in {folder}")


def read_quadrants(folder):
    data = {}
    for name, _, _ in QUADS:
        path, _ = quad_file(folder, name)
        with open(path, encoding="utf-8") as fh:
            data[name] = [ln.split() for ln in fh.read().splitlines() if ln.strip()]
    return data


def make_splits(folder, train=41500, test=2500, val=2500, seed=16):
    """Per-quadrant split. A split.json in the folder (written for the Reddit
    data, whose train/test are author-disjoint) overrides the paper's sizes."""
    data = read_quadrants(folder)
    fixed = None
    if os.path.exists(os.path.join(folder, "split.json")):
        fixed = json.load(open(os.path.join(folder, "split.json")))
    n_min = min(len(v) for v in data.values())
    if not fixed and n_min < train + test:  # small data: 90/10 per quadrant
        test = max(1, int(round(n_min * 0.1)))
        train = n_min - test
    splits = {"train": [], "test": [], "val": []}
    for name, y, z in QUADS:
        rows = data[name]
        if fixed:
            alias = quad_file(folder, name)[1]
            train, test = fixed[alias]["train"], fixed[alias]["test"]
            val = min(val, train // 10)
        tr = rows[:train]
        te = rows[train:train + test]
        va = [] if fixed else rows[train + test:train + test + val]
        if len(va) < val:  # carve validation from the end of training
            k = min(val, len(tr) // 10)
            va, tr = tr[-k:], tr[:-k]
        splits["train"] += [(t, y, z) for t in tr]
        splits["test"] += [(t, y, z) for t in te]
        splits["val"] += [(t, y, z) for t in va]
    random.Random(seed).shuffle(splits["train"])
    return splits


class Vocab:
    def __init__(self, sentences, min_count=1):
        from collections import Counter
        c = Counter(w for s in sentences for w in s)
        self.itos = [PAD, UNK] + sorted(w for w, n in c.items() if n >= min_count)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(w, 1) for w in toks] or [1]


def batches(rows, vocab, batch_size, shuffle, seed=0, max_len=60):
    idx = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [rows[j] for j in idx[i:i + batch_size]]
        seqs = [vocab.encode(t)[:max_len] for t, _, _ in chunk]
        lens = torch.tensor([len(s) for s in seqs])
        x = torch.zeros(len(seqs), int(lens.max()), dtype=torch.long)
        for k, s in enumerate(seqs):
            x[k, :len(s)] = torch.tensor(s)
        y = torch.tensor([r[1] for r in chunk])
        z = torch.tensor([r[2] for r in chunk])
        yield x, lens, y, z
