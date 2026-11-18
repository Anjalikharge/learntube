"""RAG chat with timestamp citations; works over one video or a whole library of videos."""
from .retrieval import HybridIndex, make_passages
from .summarize import lang_note
from .utils import fmt_time


def embed_fn_for(llm, enabled: bool = True):
    """Dense embeddings only when an API provider that supports them is active."""
    if enabled and llm is not None and llm.provider in ("gemini", "openai"):
        return llm.embed
    return None


def build_library_index(docs, embed_fn=None) -> HybridIndex:
    passages = []
    for d in docs:
        passages += make_passages(d.sentences, video_id=d.video_id, title=d.title)
    return HybridIndex(passages, embed_fn=embed_fn)


def _label(p, multi: bool) -> str:
    t = fmt_time(p["start"])
    return f"{p['title']} @ {t}" if multi and p.get("title") else t


def answer(question: str, index: HybridIndex, llm, history=None, mode: str = "hybrid", k: int = 5,
           language="Same as video", multi: bool = False):
    """Return (answer_text, passages, note)."""
    query = question
    if history and len(question.split()) <= 6:  # short follow-up: add the previous question for context
        prev = [t for r, t in history if r == "user"]
        if prev:
            query = prev[-1] + " " + question
    ctx = index.search(query, k=k, mode=mode)
    if not ctx:
        return "I couldn't find that in the video transcript.", [], ""
    ctx_txt = "\n\n".join(f"[{_label(p, multi)}] {p['text']}" for p in ctx)
    if llm is None or not llm.available():
        return "Most relevant parts of the video:\n\n" + ctx_txt, ctx, ""
    hist = "\n".join(f"{r}: {t}" for r, t in (history or [])[-6:])
    try:
        ans = llm.generate(
            "TASK:CHAT\nAnswer the question using ONLY the transcript excerpts. If the answer is not there, say "
            "so. Cite sources like [mm:ss] (or [Title @ mm:ss] if several videos)."
            f"{lang_note(language)}\n\nEXCERPTS:\n{ctx_txt}\n\nCHAT SO FAR:\n{hist}\n\nQUESTION: {question}", 0.2)
        return ans, ctx, ""
    except Exception as e:
        return "Most relevant parts of the video:\n\n" + ctx_txt, ctx, f"LLM unavailable ({e})."
