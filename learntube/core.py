"""Document model + analysis (sentences, chunks, keywords, retrieval index)."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .retrieval import HybridIndex, make_passages
from .textproc import build_chunks, build_sentences, detect_script, tokenize


@dataclass
class Doc:
    video_id: Optional[str]
    title: str
    sentences: list
    chunks: list
    keywords: list
    words: int
    minutes: float
    script: str = "latin"
    chapters: list = field(default_factory=list)
    index: Optional[HybridIndex] = None

    @property
    def key(self) -> str:
        return self.video_id or f"pasted-{abs(hash(self.sentences[0]['text'])) % 10**8}"


def extract_keywords(sentences, top_k: int = 12):
    docs = [s["text"] for s in sentences]
    groups = [" ".join(docs[i:i + 5]) for i in range(0, len(docs), 5)] or [""]
    try:
        vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                              ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform(groups)
    except ValueError:
        return []
    scores = np.asarray(X.sum(0)).ravel()
    terms = vec.get_feature_names_out()
    picked = []
    for i in np.argsort(scores)[::-1]:
        t = terms[i]
        if " " not in t and any(t in p.split() for p in picked):
            continue
        picked.append(t)
        if len(picked) >= top_k:
            break
    return picked


def analyze(segments, video_id=None, title: str = "", embed_fn=None) -> Doc:
    sentences = build_sentences(segments)
    if len(sentences) < 3:
        raise ValueError("Transcript too short to analyse.")
    last = segments[-1]
    text_sample = " ".join(s["text"] for s in sentences[:50])
    doc = Doc(
        video_id=video_id, title=title or "Untitled video", sentences=sentences,
        chunks=build_chunks(sentences), keywords=extract_keywords(sentences, 12),
        words=sum(len(s["text"].split()) for s in sentences),
        minutes=(last["start"] + last.get("duration", 0)) / 60,
        script=detect_script(text_sample),
    )
    doc.index = HybridIndex(make_passages(sentences, video_id=video_id, title=doc.title), embed_fn=embed_fn)
    return doc
