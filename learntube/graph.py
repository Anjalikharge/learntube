"""LangGraph workflow: input -> transcript (with Whisper fallback) -> analysis -> chapters -> summary -> grounding check.

Conditional edges decide: stop on error, use the Whisper fallback when captions are missing, and pick the
LLM or the offline extractive summariser. Every node is timed so the UI (and the report) can show a trace.
"""
import time
from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph

from .chapters import build_chapters
from .chat import embed_fn_for
from .core import Doc, analyze
from .ingest import fetch_title, fetch_transcript, get_video_id, segments_from_text, transcribe_with_whisper
from .summarize import abstractive_summary, extractive_markdown
from .verify import grounding_check


class State(TypedDict, total=False):
    # inputs
    url: str
    pasted: str
    length: str
    language: str
    use_whisper: bool
    use_dense: bool
    # working data
    video_id: str
    title: str
    segments: list
    doc: Doc
    summary: str
    summary_engine: str
    grounding: dict
    note: str
    error: str
    trace: List[dict]


def build_graph(llm):
    def traced(name, fn):
        def wrapper(state: State):
            t0 = time.time()
            out = fn(state) or {}
            entry = {"step": name, "sec": round(time.time() - t0, 3), "ok": not out.get("error")}
            return {**out, "trace": (state.get("trace") or []) + [entry]}
        return wrapper

    def parse_input(s):
        if (s.get("pasted") or "").strip():
            return {"segments": segments_from_text(s["pasted"]), "video_id": None,
                    "title": s.get("title") or "Pasted transcript"}
        if not (s.get("url") or "").strip():
            return {"error": "Enter a YouTube URL or paste a transcript."}
        try:
            return {"video_id": get_video_id(s["url"])}
        except ValueError as e:
            return {"error": str(e)}

    def fetch(s):
        if s.get("segments"):
            return {}
        try:
            _, segs = fetch_transcript(s["url"])
            return {"segments": segs, "title": fetch_title(s["video_id"])}
        except Exception as e:
            return {"note": str(e)}

    def transcribe(s):
        try:
            _, segs = transcribe_with_whisper(s["url"])
            return {"segments": segs, "title": fetch_title(s["video_id"]), "note": "Transcribed with Whisper."}
        except Exception as e:
            return {"error": f"No captions and Whisper fallback failed: {e}\n(Original: {s.get('note', '')})"}

    def do_analyze(s):
        try:
            doc = analyze(s["segments"], s.get("video_id"), s.get("title", ""),
                          embed_fn=embed_fn_for(llm, s.get("use_dense", True)))
            return {"doc": doc}
        except Exception as e:
            return {"error": str(e)}

    def chapters(s):
        build_chapters(s["doc"], llm, s.get("language", "Same as video"))
        return {}

    def summarize_llm(s):
        try:
            md = abstractive_summary(s["doc"], llm, s.get("length", "medium"), s.get("language", "Same as video"))
            return {"summary": md, "summary_engine": f"{llm.provider}:{llm.model}"}
        except Exception as e:
            return {"summary": extractive_markdown(s["doc"], s.get("length", "medium")),
                    "summary_engine": "extractive (LLM failed)", "note": f"LLM unavailable ({e}). Extractive summary shown."}

    def summarize_offline(s):
        return {"summary": extractive_markdown(s["doc"], s.get("length", "medium")),
                "summary_engine": "extractive (TextRank)"}

    def verify(s):
        translated = s.get("language", "Same as video") not in ("Same as video", "", None)
        if translated:
            return {"grounding": {"skipped": "Output language differs from the video language."}}
        return {"grounding": grounding_check(s["summary"], s["doc"], llm)}

    # ---- routing ----
    def after_input(s):
        return "stop" if s.get("error") else "fetch"

    def after_fetch(s):
        if s.get("segments"):
            return "analyze"
        return "transcribe" if s.get("use_whisper") else "fail"

    def after_whisper(s):
        return "stop" if s.get("error") else "analyze"

    def after_analyze(s):
        return "stop" if s.get("error") else "chapters"

    def pick_summary(s):
        return "llm" if llm.available() else "offline"

    def fail(s):
        return {"error": s.get("note") or "Could not fetch transcript."}

    g = StateGraph(State)
    for name, fn in [("parse_input", parse_input), ("fetch_transcript", fetch), ("whisper_fallback", transcribe),
                     ("fail", fail), ("analyze", do_analyze), ("chapters", chapters),
                     ("summarize_llm", summarize_llm), ("summarize_offline", summarize_offline),
                     ("grounding_check", verify)]:
        g.add_node(name, traced(name, fn))
    g.add_edge(START, "parse_input")
    g.add_conditional_edges("parse_input", after_input, {"fetch": "fetch_transcript", "stop": END})
    g.add_conditional_edges("fetch_transcript", after_fetch,
                            {"analyze": "analyze", "transcribe": "whisper_fallback", "fail": "fail"})
    g.add_edge("fail", END)
    g.add_conditional_edges("whisper_fallback", after_whisper, {"analyze": "analyze", "stop": END})
    g.add_conditional_edges("analyze", after_analyze, {"chapters": "chapters", "stop": END})
    g.add_conditional_edges("chapters", pick_summary, {"llm": "summarize_llm", "offline": "summarize_offline"})
    g.add_edge("summarize_llm", "grounding_check")
    g.add_edge("summarize_offline", "grounding_check")
    g.add_edge("grounding_check", END)
    return g.compile()


def run(llm, **inputs) -> State:
    """Convenience: run the whole graph and return the final state."""
    return build_graph(llm).invoke(inputs)
