"""Generate a mock TwitterAAE file in the exact on-disk format, for testing.

Matches the real twitteraae_all: 10 tab-separated columns
  tweet_id, "timestamp", "user_id", [lon, lat], "blockgroup", "text",
  P(AA), P(Hispanic), P(Asian/other), P(White)
with the text stored as a JSON string (emojis as \\uXXXX surrogate pairs,
newlines as \\n). Dialect and sentiment are planted in the vocabulary, and the
original code's corner cases are included: tweets with both happy and sad
emojis, two different happy emojis, sob-only sad tweets, exact duplicates
(also across confidence levels), tweets identical except for the mention and
emoji, newlines and other line-break characters, quotes, URLs, non-English text.

It is NOT data and its numbers mean nothing; it only proves the code runs.
"""
import argparse
import json
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
HAPPY = ["\U0001F602", "\U0001F60D", "\U0001F60A", ":)", ":D", "\U0001F601", "☺️"]
SAD = ["\U0001F612", "\U0001F629", ":(", "\U0001F614", "\U0001F622", "☹️"]
SOB = "\U0001F62D"
OTHER = ["daqui a pouco vou ir arrumar meu cabelo kkkkkk", "Интересный ремикс мэш-ап",
         "We no speak americano &amp; Marc Reason - Lambada"]


def make_tweet(rng, group, sent, emoji_rate):
    words = []
    for _ in range(rng.randint(3, 14)):
        r = rng.random()
        if r < 0.22:
            words.append(rng.choice(AA_WORDS if (group == "aa") == (rng.random() < 0.8) else WH_WORDS))
        elif r < 0.34:
            words.append(rng.choice(POS_WORDS if (sent == "pos") == (rng.random() < 0.7) else NEG_WORDS))
        else:
            words.append(rng.choice(NEUTRAL))
    if rng.random() < 0.3:
        words.insert(0, "@user" + str(rng.randint(1, 999)))
    if rng.random() < 0.2:
        words.append("http://t.co/" + str(rng.randint(10**5, 10**6)))
    if rng.random() < emoji_rate:
        if sent == "neg" and rng.random() < 0.3:
            emo = SOB                                   # sob-only sad tweet
        else:
            emo = rng.choice(HAPPY if sent == "pos" else SAD)
        x = rng.random()
        if x < 0.03:
            emo += " " + rng.choice(SAD if sent == "pos" else HAPPY)   # both classes
        elif x < 0.08:
            emo += rng.choice(HAPPY if sent == "pos" else SAD)         # two emojis, same class
        words.insert(rng.randint(0, len(words)), emo)
    text = " ".join(words)
    if rng.random() < 0.05:
        text = text.replace(" ", "\n", 1)
    elif rng.random() < 0.02:   # characters that str.splitlines() treats as line breaks
        text = text.replace(" ", rng.choice(["\r", "\u2028", "\x85", "\x0b"]), 1)
    if rng.random() < 0.02:
        text = '"' + text + '" he said'
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mock/TwitterAAE-full-v1")
    ap.add_argument("--rows", type=int, default=400000)
    ap.add_argument("--emoji-rate", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "twitteraae_all")
    recent = []
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
            if rng.random() < 0.03:
                text = rng.choice(OTHER)
            elif recent and rng.random() < 0.01:
                text = rng.choice(recent)  # exact duplicate, possibly at another confidence
            elif rng.random() < 0.01:      # same words, another mention and emoji
                text = rng.choice(["@a", "@b", "@c"]) + " same words every time here " + \
                    rng.choice(HAPPY + SAD)
            else:
                text = make_tweet(rng, group, rng.choice(["pos", "neg"]), args.emoji_rate)
            recent = (recent + [text])[-50:]
            row = [str(286475921177853953 + i), '"Wed Jan 02 14:15:56 +0000 2013"',
                   '"%d"' % rng.randint(10**6, 10**9), "[-74.26275, 40.57988]", '"340230090002"',
                   json.dumps(text), repr(round(aa, 12)), repr(round(rest * 0.6, 12)),
                   repr(round(rest * 0.4, 12)), repr(round(wh, 12))]
            fh.write("\t".join(row) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
