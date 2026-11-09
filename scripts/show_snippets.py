"""Print short data snippets for the report figures (IDs masked). Run from the repo root."""
import glob
import json
import os
import re
import zipfile


def mask_digits(s):
    return re.sub(r"\d", "X", s)


def section(title):
    print("\n" + "=" * 8, title, "=" * 8)


section("TwitterAAE raw (first 2 lines, IDs masked)")
try:
    z = zipfile.ZipFile(glob.glob("data/raw/*TwitterAAE*.zip")[0])
    name = [n for n in z.namelist() if n.endswith("twitteraae_all")][0]
    print("file in zip:", name)
    with z.open(name) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            parts = line.decode("utf-8", "replace").rstrip("\r\n").split("\t")
            parts[0], parts[2], parts[4] = mask_digits(parts[0]), mask_digits(parts[2]), mask_digits(parts[4])
            print(" | ".join(parts))
except Exception as e:
    print("not found:", e)

section("TwitterAAE processed (clean), 2 lines per quadrant")
for q in ["pos_aa", "pos_wh", "neg_aa", "neg_wh"]:
    p = f"data/processed/sent_race/{q}.txt"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            print(q, "->", [next(f).strip() for _ in range(2)])

section("Reddit raw (one comment per group, selected fields, author masked)")
for path in ["data/reddit/raw/ng_Nigeria.jsonl", "data/reddit/raw/us_chicago.jsonl"]:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                if 8 <= len(c.get("body", "").split()) <= 30 and c.get("author") not in ("[deleted]", "AutoModerator"):
                    keep = {k: c.get(k) for k in ["subreddit", "created_utc", "score", "body"]}
                    keep["author"] = "XXXX"
                    print(os.path.basename(path), "->", json.dumps(keep, ensure_ascii=False))
                    break

section("Reddit processed (tokens and masked), 2 lines each")
for variant in ["tokens", "masked"]:
    for p in sorted(glob.glob(f"data/reddit/processed/{variant}/*.txt"))[:4]:
        with open(p, encoding="utf-8") as f:
            print(variant, os.path.basename(p), "->", [next(f).strip() for _ in range(2)])

section("PAN 2017 raw (truth file + start of one author file)")
en = next(iter(glob.glob("data/raw/pan17/**/en", recursive=True)), None)
if en:
    with open(os.path.join(en, "truth.txt"), encoding="utf-8") as f:
        for _ in range(2):
            print("truth.txt ->", mask_digits(next(f).strip()))
    xml = sorted(glob.glob(os.path.join(en, "*.xml")))[0]
    with open(xml, encoding="utf-8") as f:
        print(os.path.basename(mask_digits(xml)), "->", f.read(700))
else:
    print("not found")

section("PAN 2017 processed, 2 lines per quadrant")
for p in sorted(glob.glob("data/processed/pan17_gb_us/*.txt"))[:4]:
    with open(p, encoding="utf-8") as f:
        print(os.path.basename(p), "->", [next(f).strip() for _ in range(2)])
