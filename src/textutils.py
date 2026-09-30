"""Text utilities shared by the TwitterAAE and Reddit pipelines.

Python 3 port of the text handling in Elazar & Goldberg's original repository
(src/data/twitter_utils.py). The emoji lists are the ones given in the paper's
appendix and in that file; they are written here as real Unicode characters
rather than the escaped UTF-16 surrogate strings the Python 2 code used.
"""
import re

import twokenize

MENTION = "_TWITTER-ENTITY_"

# ---- Sentiment emoji lists (original repo, twitter_utils.py) -----------------
HAPPY_ASCII = [":)", ":-)", ": )", ":D", "=)", "(:", "(-:", "(="]
HAPPY_EMOJI = [
    "\U0001F600",  # grinning
    "\U0001F603",  # smiley
    "\U0001F604",  # smile
    "\U0001F601",  # grin
    "\U0001F606",  # laughing / satisfied
    "\U0001F605",  # sweat_smile
    "\U0001F602",  # joy
    "\U0001F923",  # rofl
    "☺️",  # relaxed
    "\U0001F60A",  # blush
    "\U0001F642",  # slightly_smiling_face
    "\U0001F60D",  # heart_eyes
    "\U0001F618",  # kissing_heart
    "\U0001F61C",  # stuck_out_tongue_winking_eye
    "\U0001F61D",  # stuck_out_tongue_closed_eyes
    "\U0001F608",  # smiling_imp
    "\U0001F639",  # joy_cat
    "\U0001F63A",  # smiley_cat
]
SAD_ASCII = [":(", ":-(", ": (", "=(", "):", ")-:", ") :", ")="]
SAD_EMOJI = [
    "\U0001F612",  # unamused
    "\U0001F61E",  # disappointed
    "\U0001F614",  # pensive
    "☹️",  # frowning_face
    "\U0001F623",  # persevere
    "\U0001F62B",  # tired_face
    "\U0001F629",  # weary
    "\U0001F624",  # triumph
    "\U0001F620",  # angry
    "\U0001F621",  # rage
    "\U0001F622",  # cry
    "\U0001F62D",  # sob
    "\U0001F628",  # fearful
    "\U0001F630",  # cold_sweat
    "\U0001F912",  # face_with_thermometer
    "\U0001F47F",  # imp
    "\U0001F922",  # nauseated_face
    "\U0001F61F",  # worried
    "\U0001F641",  # slightly_frowning_face
    "\U0001F616",  # confounded
    "\U0001F626",  # frowning
    "\U0001F627",  # anguished
    "\U0001F63F",  # crying_cat_face
]
HAPPY = HAPPY_ASCII + HAPPY_EMOJI
SAD = SAD_ASCII + SAD_EMOJI

# ---- Decoding of escaped text ------------------------------------------------
# The TwitterAAE files store non-ASCII characters as literal escape sequences
# (e.g. "😂"); the original code relied on this. We decode them so the
# rest of the pipeline works on real characters, and pass real Unicode through.
_ESC = re.compile(r"\\u([0-9a-fA-F]{4})")


def decode_escapes(text):
    if "\\u" not in text:
        return text

    def repl(m):
        return chr(int(m.group(1), 16))

    s = _ESC.sub(repl, text)
    # join UTF-16 surrogate pairs into real code points
    try:
        s = s.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeDecodeError:
        s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return s


def emoji_hits(text, emoji_list):
    """Indices of entries of emoji_list that occur in text."""
    return [i for i, e in enumerate(emoji_list) if e in text]


# ---- Emoji removal and tokenisation (mirrors normalize_text) ------------------
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoticons, supplemental
    "\U0001F1E6-\U0001F1FF"  # flags
    "☀-➿"          # misc symbols and dingbats (incl. relaxed, frowning)
    "️‍"           # variation selector, zero-width joiner
    "]+"
)
_EMOTICON_RE = re.compile(
    r"(:\)+)|(:\(+)|(:p)|(:D)|(;\)+)|(=\)+)|(=\(+)|(:-\)+)|(:-\(+)|"
    r"(\(+:)|(\)+:)|(\(+=)|(\)+=)|(\(+-:)|(\)+-:)|(: \))|(: \()|(\) :)"
)


def remove_emojis(text):
    text = _EMOJI_RE.sub(" ", text)
    text = _EMOTICON_RE.sub(" ", text)
    return text.strip()


def normalize_text(text):
    """Remove emojis and emoticons, tokenise with twokenize, map @mentions."""
    no_emojis = remove_emojis(text)
    if not no_emojis:
        return []
    toks = twokenize.simpleTokenize(no_emojis)
    out = []
    for t in toks:
        t = t.replace("\n", "")
        if not t:
            continue
        out.append(MENTION if t.startswith("@") else t)
    return out
