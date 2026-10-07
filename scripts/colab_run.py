"""All training runs, for a Colab GPU session. Paste into one Colab cell:

    from google.colab import drive; drive.mount('/content/drive')
    !git clone -q https://github.com/aayushi-patel13/demog-leakage.git /content/demog-leakage 2>/dev/null || git -C /content/demog-leakage pull -q
    %cd /content/demog-leakage
    !python scripts/colab_run.py

Reads processed_data.zip (TwitterAAE) and, when present, reddit_data.zip from
MyDrive/comp8240, and writes every result to MyDrive/comp8240/results so it
survives a disconnect. A run whose results.md is already in Drive is skipped,
so after a disconnect the same cell simply carries on with what is left.
Order: TwitterAAE replication, Reddit (tokens, masked), TwitterAAE faithful;
with --extra, then the adversarial check (more seeds, one longer run).
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile

RUNS = [
    ("twitteraae", "data/processed/sent_race", []),
    ("reddit_tokens", "data/reddit/processed/tokens", ["--attribute", "variety (NG vs US)", "--no-paper"]),
    ("reddit_masked", "data/reddit/processed/masked", ["--attribute", "variety (NG vs US)", "--no-paper"]),
    ("twitteraae_faithful", "data/processed/sent_race_faithful", []),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default="/content/drive/MyDrive/comp8240")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--only", nargs="*", help="run only these tags")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--extra", action="store_true",
                    help="also run the adversarial check (two more seeds and a 60-epoch run, about an hour)")
    ap.add_argument("--max-train", type=int, default=None, help="testing only")
    ap.add_argument("--attacker-epochs", type=int, default=None, help="testing only")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("No GPU. In Colab: Runtime > Change runtime type > T4 GPU > Save, then run the cell again.")
    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)

    D = args.drive
    for name, dest in (("processed_data.zip", "data/processed"), ("reddit_data.zip", ".")):
        z = os.path.join(D, name)
        if os.path.exists(z):
            zipfile.ZipFile(z).extractall(dest)
            print(f"unpacked {name}", flush=True)
        else:
            print(f"({name} not in {D}; its runs are skipped)", flush=True)
    os.makedirs(os.path.join(D, "results"), exist_ok=True)
    if os.path.islink("results") or os.path.isfile("results"):
        os.remove("results")
    elif os.path.isdir("results"):
        shutil.rmtree("results")
    os.symlink(os.path.join(D, "results"), "results")

    done = []
    for tag, data, extra in RUNS:
        if args.only and tag not in args.only:
            continue
        if not os.path.exists(os.path.join(data, "pos_wh.txt" if "processed/sent" in data else "pos_ng.txt")):
            print(f"\n== {tag}: no data, skipped", flush=True)
            continue
        if os.path.exists(os.path.join("results", tag, "results.md")):
            print(f"\n== {tag}: already finished (results in Drive), skipped", flush=True)
            done.append(tag)
            continue
        print(f"\n== {tag}: training on {data}", flush=True)
        cmd = [sys.executable, "src/experiments.py", "run-all", "--data", data, "--epochs", str(args.epochs),
               "--tag", tag] + extra
        if args.max_train:
            cmd += ["--max-train", str(args.max_train)]
        if args.attacker_epochs:
            cmd += ["--attacker-epochs", str(args.attacker_epochs)]
        subprocess.run(cmd, check=True)
        done.append(tag)
    if args.extra:
        tag = "twitteraae_advcheck"
        if os.path.exists(os.path.join("results", tag, "results.md")):
            print(f"\n== {tag}: already finished (results in Drive), skipped", flush=True)
        else:
            print(f"\n== {tag}: adversarial training with more seeds and a longer run", flush=True)
            cmd = [sys.executable, "src/experiments.py", "adv-check", "--data", "data/processed/sent_race",
                   "--tag", tag]
            if args.max_train:
                cmd += ["--max-train", str(args.max_train), "--epochs", "2", "--long-epochs", "4"]
            if args.attacker_epochs:
                cmd += ["--attacker-epochs", str(args.attacker_epochs)]
            subprocess.run(cmd, check=True)
        done.append(tag)
    for tag in done:
        print(f"\n######## {tag}\n" + open(os.path.join("results", tag, "results.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
