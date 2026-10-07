"""Mock PAN 2017 English folder (truth.txt + one XML per author), for testing."""
import os
import random
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "data/mock/pan17/en"
os.makedirs(out, exist_ok=True)
rng = random.Random(3)
GB = ["brilliant", "cheers", "mate", "quid", "rubbish", "lovely", "proper", "telly"]
US = ["awesome", "dude", "y'all", "bucks", "gonna", "totally", "trash", "tv"]
COMMON = ["the", "a", "to", "and", "my", "you", "is", "today", "work", "going", "this", "so", "game"]
HAPPY, SAD = ["\U0001F602", "\U0001F60D", ":)", "\U0001F60A"], ["\U0001F612", ":(", "\U0001F622", "\U0001F614"]
truth = []
for v, words in (("great britain", GB), ("united states", US), ("canada", US)):
    for a in range(60):
        aid = f"{v[:2]}{a:04d}"
        truth.append(f"{aid}:::{rng.choice(['male', 'female'])}:::{v}")
        docs = []
        for i in range(100):
            t = " ".join(rng.choice(words) if rng.random() < 0.3 else rng.choice(COMMON)
                         for _ in range(rng.randint(3, 12)))
            r = rng.random()
            if r < 0.35:
                t += " " + rng.choice(HAPPY)
            elif r < 0.55:
                t += " " + rng.choice(SAD)
            if rng.random() < 0.05:
                t = "RT @someone: " + t
            docs.append(f"<document><![CDATA[{t}]]></document>")
        with open(os.path.join(out, aid + ".xml"), "w", encoding="utf-8") as fh:
            fh.write('<author lang="en">\n<documents>\n' + "\n".join(docs) + "\n</documents>\n</author>\n")
with open(os.path.join(out, "truth.txt"), "w") as fh:
    fh.write("\n".join(truth) + "\n")
print("wrote", out)
