"""Generate a small mock TwitterAAE file in the exact on-disk format, for testing.

The mock has the same 10 tab-separated columns as the real corpus and stores
emojis as literal escape sequences (e.g. \\ud83d\\ude02), as the real files do.
Dialect and sentiment are planted in the vocabulary so that the full pipeline
(preparation, training, attacking) can be exercised end to end.

It is NOT data and its numbers mean nothing; it only proves the code runs.
"""
import argparse
import os
import random

AA_WORDS = ["finna", "tryna", "ion", "dis", "dat", "bout", "yall", "lowkey", "bruh", "wassup",
            "nah", "gon", "aint", "sholl", "bae", "turnt", "hella", "deadass"]
WH_WORDS = ["literally", "totally", "awesome", "super", "gonna", "dude", "omg", "haha",
            "seriously", "definitely", "anyway", "honestly", "basically", "whatever"]
POS_WORDS = ["love", "great", "happy", "best", "fun", "amazing", "good", "yay", "blessed", "lol"]
NEG_WORDS = ["hate", "sad", "worst", "tired", "sick", "bad", "ugh", "annoyed", "miss", "hurt"]
NEUTRAL = ["the", "a", "to", "and", "my", "you", "is", "in", "it", "at", "on", "today", "work",
           "school", "game", "night", "phone", "home", "going", "just", "this", "that", "so"]
BS = chr(92)  # a literal backslash; the corpus stores emojis as \\uXXXX text

def esc(cp):
    """UTF-16 escape text for a code point, as stored in the TwitterAAE files."""
    u = chr(cp).encode("utf-16-be")
    return "".join(BS + "u" + u[i:i + 2].hex() for i in range(0, len(u), 2))


HAPPY_ESC = [esc(0x1F602), esc(0x1F60D), esc(0x1F60A), ":)", ":D", esc(0x1F601)]
SAD_ESC = [esc(0x1F612), esc(0x1F62D), esc(0x1F629), ":(", esc(0x1F614), esc(0x1F622)]


def make_tweet(rng, group, sent):
    n = rng.randint(4, 14)
    words = []
    for _ in range(n):
        r = rng.random()
        if r < 0.22:
            words.append(rng.choice(AA_WORDS if (group == "aa") == (rng.random() < 0.8) else WH_WORDS))
        elif r < 0.34:
            words.append(rng.choice(POS_WORDS if (sent == "pos") == (rng.random() < 0.7) else NEG_WORDS))
        else:
            words.append(rng.choice(NEUTRAL))
    if rng.random() < 0.3:
        words.insert(0, "@user" + str(rng.randint(1, 999)))
    emo = rng.choice(HAPPY_ESC if sent == "pos" else SAD_ESC)
    if rng.random() < 0.02:  # conflicting emojis, as in real data
        emo += " " + rng.choice(SAD_ESC if sent == "pos" else HAPPY_ESC)
    words.insert(rng.randint(0, len(words)), emo)
    return " ".join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mock/TwitterAAE-full-v1")
    ap.add_argument("--rows", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "twitteraae_all")
    last = None
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(args.rows):
            r = rng.random()
            if r < 0.35:
                group, aa = "aa", rng.uniform(0.81, 0.99)
            elif r < 0.7:
                group, aa = "wh", rng.uniform(0.0, 0.1)
            else:
                group, aa = rng.choice(["aa", "wh"]), rng.uniform(0.2, 0.6)
            wh = (1 - aa) * (rng.uniform(0.9, 1.0) if group == "wh" else rng.uniform(0.3, 0.6))
            rest = max(0.0, 1 - aa - wh)
            his, oth = rest * 0.6, rest * 0.4
            has_emoji = rng.random() < 0.6
            sent = rng.choice(["pos", "neg"])
            text = make_tweet(rng, group, sent) if has_emoji else " ".join(rng.choice(NEUTRAL) for _ in range(8))
            if last and rng.random() < 0.01:
                text = last  # exact duplicates
            last = text
            fh.write("\t".join([str(10**17 + i), "2013-06-01T12:00:00", str(rng.randint(1, 10**8)),
                                "[40.7, -74.0]", "360610001001", text,
                                f"{aa:.6f}", f"{his:.6f}", f"{oth:.6f}", f"{wh:.6f}"]) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
