"""Extractive (TF-IDF + TextRank) and abstractive (LLM, parallel map-reduce) summarisation."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .core import Doc
from .textproc import timestamped_text, tokenize
from .utils import fmt_time

LENGTHS = {"short": (5, "3-5 bullet points"), "medium": (8, "6-8 bullet points"),
           "detailed": (12, "10-12 bullet points")}
RULES = ("Use ONLY information from the transcript; do not add outside facts. "
         "The transcript contains [mm:ss] markers - cite the marker where each point is said.")


def lang_note(language) -> str:
    return "" if language in (None, "", "Same as video") else f" Write the answer in {language}."


def extractive_summary(sentences, n: int = 8):
    """TextRank over a TF-IDF cosine graph; returns sentences in original order."""
    if len(sentences) <= n:
        return sentences
    try:
        X = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False).fit_transform(
            [s["text"] for s in sentences])
    except ValueError:
        return sentences[:n]
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0)
    rs = sim.sum(1, keepdims=True)
    rs[rs == 0] = 1
    P = sim / rs
    N = len(sentences)
    r = np.ones(N) / N
    for _ in range(60):
        r = 0.15 / N + 0.85 * (P.T @ r)
    return [sentences[i] for i in sorted(np.argsort(r)[-n:])]


def extractive_markdown(doc: Doc, length: str = "medium") -> str:
    picks = extractive_summary(doc.sentences, LENGTHS[length][0])
    return "\n".join(f"- {s['text']} [{fmt_time(s['start'])}]" for s in picks)


def chapter_text(doc: Doc, ch) -> str:
    return timestamped_text(doc.sentences[ch["start_idx"]:ch["end_idx"]])


def summarize_chapters(doc: Doc, llm, language="Same as video", workers: int = 4):
    """One short LLM summary per chapter (parallel). Stores ch['summary']."""
    def one(ch):
        text = " ".join(chapter_text(doc, ch).split()[:1800])
        return llm.generate(f"TASK:CHAPTER_SUMMARY\nSummarize this section of a video titled '{ch['title']}' "
                            f"in 3-4 bullet points. {RULES}{lang_note(language)}\n\nTRANSCRIPT:\n{text}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ch, s in zip(doc.chapters, ex.map(one, doc.chapters)):
            ch["summary"] = s
    return doc.chapters


def abstractive_summary(doc: Doc, llm, length: str = "medium", language="Same as video") -> str:
    """Single call for short videos; chapter-level map then reduce for long ones."""
    spec = LENGTHS[length][1]
    fmt = (f"Start with a line 'TL;DR:' (2 sentences), then {spec}, each ending with a [mm:ss] timestamp, "
           "then a final line 'Key takeaway:'.")
    if doc.words <= 6000 or not doc.chapters:
        text = timestamped_text(doc.sentences)
        return llm.generate(f"TASK:SUMMARY\nSummarize this YouTube video transcript. {fmt} {RULES}"
                            f"{lang_note(language)}\n\nTRANSCRIPT:\n{text}")
    if not all("summary" in c for c in doc.chapters):
        summarize_chapters(doc, llm, language)
    joined = "\n\n".join(f"## [{fmt_time(c['start'])}] {c['title']}\n{c['summary']}" for c in doc.chapters)
    return llm.generate(f"TASK:SUMMARY_REDUCE\nCombine these section summaries into one coherent summary of the "
                        f"whole video. {fmt} Keep the existing [mm:ss] timestamps. {RULES}{lang_note(language)}"
                        f"\n\n{joined}")
