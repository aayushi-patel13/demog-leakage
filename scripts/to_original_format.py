"""Write a four-quadrant dataset in the format the original code reads.

The authors' trainer.py expects data/processed/sent_race/ with files
pos_pos, pos_neg, neg_pos, neg_neg (one tweet per line, as word ids separated
by spaces) and vocab (one word per line; the line number is the id), which is
what their make_data.py writes. Quadrant names map as in make_data.py:
pos_pos = happy & AAE, pos_neg = happy & White, neg_pos = sad & AAE,
neg_neg = sad & White.

Usage: python scripts/to_original_format.py <our_folder> <original_data_folder>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data import read_lines  # noqa: E402

NAMES = {"pos_pos": "pos_aa", "pos_neg": "pos_wh", "neg_pos": "neg_aa", "neg_neg": "neg_wh"}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/processed/sent_race_faithful"
    out = sys.argv[2] if len(sys.argv) > 2 else "original/data/processed/sent_race"
    os.makedirs(out, exist_ok=True)
    sents = {k: [ln.split() for ln in read_lines(os.path.join(src, v + ".txt"))] for k, v in NAMES.items()}
    vocab = sorted({w for rows in sents.values() for s in rows for w in s})
    idx = {w: i for i, w in enumerate(vocab)}
    with open(os.path.join(out, "vocab"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(vocab))
    for k, rows in sents.items():
        with open(os.path.join(out, k), "w", encoding="utf-8", newline="\n") as fh:
            for s in rows:
                fh.write(" ".join(str(idx[w]) for w in s) + "\n")
    print(f"wrote {out}: vocab {len(vocab):,}; " + ", ".join(f"{k} {len(v):,}" for k, v in sents.items()))


if __name__ == "__main__":
    main()
