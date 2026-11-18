"""Cleaning, tokenisation (English + Hindi/Marathi), sentence and chunk building."""
import bisect
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .utils import fmt_time

STOP = set(ENGLISH_STOP_WORDS) | {
    "um", "uh", "okay", "yeah", "gonna", "wanna", "like", "just", "really", "actually", "basically",
    "going", "know", "right", "let", "lets", "say", "thing", "things", "want", "need", "use", "using",
    "used", "one", "will", "so", "now", "also", "kind", "sort", "got",
}
STOP |= set("""है हैं था थे थी और का की के को में से पर यह वो ये तो भी कि जो हम मैं आप तुम नहीं
कर करना करते कुछ इस उस एक ही या लिए तक जाता होता होती होते रहा रही रहे सकता अपने अपना बहुत फिर
क्या कैसे जब तब अगर लेकिन बात हो गया गई गए दिया किया लिया वाला वाली वाले मेरा मेरी आपका उनके इन
उन कोई सब सारे जैसे जैसा ऐसे ऐसा ना हां हाँ तुम्हारा हमारा मुझे तुझे उसे इसे हूं हूँ थीं
करके करें कीजिए चाहिए जी अब यहां वहां इसलिए क्योंकि कौन कहां कितना आहे आणि या ते तो ती हे मी तू आम्ही
नाही होते होता पण की त्या ह्या""".split())

_SPLIT = re.compile(r"[\s.,!?;:\"'()\[\]{}|।\-–—/\\]+")
_NOISE = re.compile(r"\[[^\]]*\]|>>+|♪+")
_SUFFIXES = ("ing", "ed", "es", "s", "ly")


def stem(tok: str) -> str:
    """Very light English stemmer (keeps tokens readable, no external data needed)."""
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def clean_text(t: str) -> str:
    t = _NOISE.sub(" ", t.replace("\n", " "))
    return re.sub(r"\s+", " ", t).strip()


def tokenize(text: str):
    """Unicode-safe tokenizer: splits on separators (\\w breaks Devanagari vowel signs)."""
    toks = []
    for t in _SPLIT.split(text.lower()):
        if not t or t.isdigit() or t in STOP:
            continue
        if t.isascii():
            if len(t) >= 3 and t[0].isalpha():
                toks.append(stem(t))
        elif len(t) >= 2:
            toks.append(t)
    return toks


def detect_script(text: str) -> str:
    dev = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    return "devanagari" if dev > 0.3 * max(1, sum(1 for c in text if c.isalpha())) else "latin"


def build_sentences(segments, max_words: int = 40, window: int = 25):
    """Segments -> [{text, start}] sentences; windows unpunctuated auto-captions."""
    text, offsets, starts = "", [], []
    for s in segments:
        c = clean_text(s["text"])
        if not c:
            continue
        offsets.append(len(text))
        starts.append(s["start"])
        text += c + " "
    out = []
    for m in re.finditer(r"[^.!?।]+[.!?।]*", text):
        sent = m.group().strip()
        if len(sent.split()) < 4:
            continue
        i = max(bisect.bisect_right(offsets, m.start()) - 1, 0)
        start = starts[i]
        words = sent.split()
        if len(words) > max_words:
            for j in range(0, len(words), window):
                part = " ".join(words[j:j + window])
                if len(part.split()) >= 4:
                    out.append({"text": part, "start": start})
        else:
            out.append({"text": sent, "start": start})
    return out


def build_chunks(sentences, max_words: int = 1200):
    chunks, cur, n = [], [], 0
    for s in sentences:
        w = len(s["text"].split())
        if cur and n + w > max_words:
            chunks.append({"text": " ".join(x["text"] for x in cur), "start": cur[0]["start"]})
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append({"text": " ".join(x["text"] for x in cur), "start": cur[0]["start"]})
    return chunks


def timestamped_text(sentences, every_words: int = 60) -> str:
    """Transcript with a [mm:ss] marker roughly every `every_words` words (lets the LLM cite times)."""
    out, n = [], 0
    for s in sentences:
        if n == 0 or n >= every_words:
            out.append(f"\n[{fmt_time(s['start'])}]")
            n = 0
        out.append(s["text"])
        n += len(s["text"].split())
    return " ".join(out).strip()
