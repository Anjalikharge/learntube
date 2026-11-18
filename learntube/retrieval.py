"""Hybrid retrieval: BM25 + TF-IDF (+ optional dense embeddings) fused with Reciprocal Rank Fusion."""
import math
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .textproc import tokenize


def make_passages(sentences, window: int = 6, stride: int = 3, video_id=None, title=""):
    out = []
    for i in range(0, len(sentences), stride):
        g = sentences[i:i + window]
        out.append({"text": " ".join(s["text"] for s in g), "start": g[0]["start"],
                    "video_id": video_id, "title": title})
    return out


class HybridIndex:
    MODES = ("tfidf", "bm25", "dense", "hybrid")

    def __init__(self, passages, embed_fn=None, k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        texts = [p["text"] for p in passages]
        # --- TF-IDF ---
        self.vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                   ngram_range=(1, 2), sublinear_tf=True)
        try:
            self.X = self.vec.fit_transform(texts)
        except ValueError:
            self.vec, self.X = None, None
        # --- BM25 ---
        self.tokens = [tokenize(t) for t in texts]
        self.dl = np.array([len(t) for t in self.tokens], dtype=float)
        self.avgdl = max(self.dl.mean(), 1e-9) if len(self.dl) else 1.0
        self.post = defaultdict(list)
        for i, toks in enumerate(self.tokens):
            counts = defaultdict(int)
            for t in toks:
                counts[t] += 1
            for t, c in counts.items():
                self.post[t].append((i, c))
        # --- dense (optional) ---
        self.embed_fn, self.E = embed_fn, None
        if embed_fn is not None:
            try:
                self.E = embed_fn(texts)
            except Exception:
                self.embed_fn, self.E = None, None

    @property
    def has_dense(self) -> bool:
        return self.E is not None

    def _tfidf(self, q):
        if self.X is None:
            return np.zeros(len(self.passages))
        return (self.X @ self.vec.transform([q]).T).toarray().ravel()

    def _bm25(self, q):
        n = len(self.passages)
        scores = np.zeros(n)
        for t in set(tokenize(q)):
            plist = self.post.get(t)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return scores

    def _dense(self, q):
        if self.E is None:
            return np.zeros(len(self.passages))
        return self.E @ self.embed_fn([q])[0]

    def _ranked(self, scores, n=20):
        idx = [int(i) for i in np.argsort(scores)[::-1][:n] if scores[i] > 0]
        return idx

    def search(self, query: str, k: int = 4, mode: str = "hybrid"):
        """Return top-k passages (dicts with an added 'score')."""
        if mode == "tfidf":
            sc = self._tfidf(query)
        elif mode == "bm25":
            sc = self._bm25(query)
        elif mode == "dense" and self.has_dense:
            sc = self._dense(query)
        else:  # hybrid: Reciprocal Rank Fusion
            fused = defaultdict(float)
            signals = [self._bm25(query), self._tfidf(query)]
            if self.has_dense:
                signals.append(self._dense(query))
            for s in signals:
                for rank, i in enumerate(self._ranked(s)):
                    fused[i] += 1.0 / (60 + rank)
            order = sorted(fused, key=fused.get, reverse=True)[:k]
            return [{**self.passages[i], "score": round(fused[i], 5)} for i in order]
        idx = self._ranked(sc, k)
        return [{**self.passages[i], "score": round(float(sc[i]), 5)} for i in idx]
