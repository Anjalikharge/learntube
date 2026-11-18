from learntube.core import analyze
from learntube.ingest import segments_from_text
from learntube.retrieval import HybridIndex, make_passages
from helpers import SAMPLE
import numpy as np


def _doc():
    return analyze(segments_from_text(SAMPLE))


def test_all_modes_return_relevant_passage():
    doc = _doc()
    for mode in ("tfidf", "bm25", "hybrid"):
        got = doc.index.search("how does regularization relate to overfitting", k=3, mode=mode)
        assert got and any("regularization" in p["text"].lower() for p in got)


def test_hybrid_beats_nothing_for_other_topic():
    doc = _doc()
    got = doc.index.search("sql injection and input validation", k=3)
    assert any("sql injection" in p["text"].lower() for p in got)


def test_unknown_query_returns_empty():
    assert _doc().index.search("zzzqqq xylophone", k=3) == []


def test_dense_signal_is_fused_when_available():
    doc = _doc()
    rng = np.random.default_rng(0)

    def fake_embed(texts):
        m = rng.normal(size=(len(texts), 16))
        return m / np.linalg.norm(m, axis=1, keepdims=True)
    idx = HybridIndex(make_passages(doc.sentences), embed_fn=fake_embed)
    assert idx.has_dense
    assert idx.search("overfitting", k=3, mode="hybrid")


def test_embedding_failure_falls_back_to_lexical():
    doc = _doc()

    def broken(texts):
        raise RuntimeError("no network")
    idx = HybridIndex(make_passages(doc.sentences), embed_fn=broken)
    assert not idx.has_dense and idx.search("overfitting", k=2)
