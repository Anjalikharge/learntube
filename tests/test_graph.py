from learntube import graph, ingest
from learntube.llm import LLM, LLMConfig
from helpers import FakeLLM, SAMPLE

OFFLINE = LLM(LLMConfig(provider="none"))


def steps(state):
    return [t["step"] for t in state["trace"]]


def test_offline_end_to_end_with_pasted_transcript():
    s = graph.run(OFFLINE, pasted=SAMPLE, length="medium")
    assert not s.get("error")
    assert s["summary"] and s["summary_engine"].startswith("extractive")
    assert s["doc"].chapters and s["grounding"]["items"]
    assert steps(s) == ["parse_input", "fetch_transcript", "analyze", "chapters", "summarize_offline", "grounding_check"]


def test_invalid_url_stops_early():
    s = graph.run(OFFLINE, url="https://example.com/x")
    assert "Invalid YouTube URL" in s["error"] and steps(s) == ["parse_input"]


def test_empty_input_error():
    assert graph.run(OFFLINE)["error"]


def test_fetch_failure_without_whisper(monkeypatch):
    monkeypatch.setattr(graph, "fetch_transcript", lambda u: (_ for _ in ()).throw(RuntimeError("captions disabled")))
    s = graph.run(OFFLINE, url="dQw4w9WgXcQ", use_whisper=False)
    assert "captions disabled" in s["error"] and steps(s)[-1] == "fail"


def test_whisper_fallback_route(monkeypatch):
    monkeypatch.setattr(graph, "fetch_transcript", lambda u: (_ for _ in ()).throw(RuntimeError("no captions")))
    monkeypatch.setattr(graph, "fetch_title", lambda v: "T")
    segs = ingest.segments_from_text(SAMPLE)
    monkeypatch.setattr(graph, "transcribe_with_whisper", lambda u: ("dQw4w9WgXcQ", segs))
    s = graph.run(OFFLINE, url="dQw4w9WgXcQ", use_whisper=True)
    assert not s.get("error") and "whisper_fallback" in steps(s) and s["doc"].video_id == "dQw4w9WgXcQ"


def test_whisper_failure_reports_both_errors(monkeypatch):
    monkeypatch.setattr(graph, "fetch_transcript", lambda u: (_ for _ in ()).throw(RuntimeError("no captions")))
    monkeypatch.setattr(graph, "transcribe_with_whisper", lambda u: (_ for _ in ()).throw(RuntimeError("no ffmpeg")))
    s = graph.run(OFFLINE, url="dQw4w9WgXcQ", use_whisper=True)
    assert "no ffmpeg" in s["error"] and "no captions" in s["error"]


def test_llm_route_and_grounding():
    llm = FakeLLM()
    s = graph.run(llm, pasted=SAMPLE, length="short")
    assert s["summary_engine"].startswith("gemini") and "TL;DR" in s["summary"]
    assert s["grounding"]["supported_pct"] >= 50  # fake summary is built from the transcript itself


def test_llm_failure_falls_back_to_extractive():
    class Broken(FakeLLM):
        def _call(self, prompt, t):
            if prompt.startswith("TASK:CHAPTER_TITLES"):
                return "[]"
            raise RuntimeError("quota")
    llm = Broken()
    llm.generate.__func__  # noqa
    import learntube.llm as m
    orig = m.time.sleep
    m.time.sleep = lambda s: None
    try:
        s = graph.run(llm, pasted=SAMPLE)
    finally:
        m.time.sleep = orig
    assert s["summary_engine"].startswith("extractive") and "quota" in s["note"]


def test_translated_output_skips_lexical_grounding():
    s = graph.run(FakeLLM(), pasted=SAMPLE, language="Hindi")
    assert "skipped" in s["grounding"]


def test_long_video_uses_map_reduce():
    long_text = SAMPLE + "\n" + "\n".join(
        f"[{20 + i // 60:02d}:{i % 60:02d}] " + " ".join(["analysis of pipeline stage number %d" % i] * 8) + "." for i in range(900))
    llm = FakeLLM()
    s = graph.run(llm, pasted=long_text)
    assert s["doc"].words > 6000
    assert any(p.startswith("TASK:CHAPTER_SUMMARY") for p in llm.prompts)
    assert any(p.startswith("TASK:SUMMARY_REDUCE") for p in llm.prompts)
