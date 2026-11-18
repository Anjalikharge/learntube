from learntube.chapters import build_chapters
from learntube.core import analyze
from learntube.ingest import segments_from_text
from helpers import FakeLLM, SAMPLE
from learntube.llm import LLM, LLMConfig


def test_chapters_found_and_ordered():
    doc = analyze(segments_from_text(SAMPLE))
    chs = build_chapters(doc, LLM(LLMConfig(provider="none")))
    assert 3 <= len(chs) <= 8
    starts = [c["start"] for c in chs]
    assert starts == sorted(starts) and chs[0]["start_idx"] == 0
    assert chs[-1]["end_idx"] == len(doc.sentences)
    assert all(c["title"] for c in chs)


def test_boundaries_near_topic_changes():
    doc = analyze(segments_from_text(SAMPLE))
    chs = build_chapters(doc, LLM(LLMConfig(provider="none")), max_chapters=4)
    # topic switches occur every 26 sentences; some boundary must fall within 8 sentences of each switch
    for switch in (26, 52, 78):
        assert any(abs(c["start_idx"] - switch) <= 8 for c in chs), (switch, [c["start_idx"] for c in chs])


def test_llm_titles_used():
    doc = analyze(segments_from_text(SAMPLE))
    chs = build_chapters(doc, FakeLLM())
    assert chs[0]["title"] == "Topic 1"


def test_short_transcript_single_chapter():
    doc = analyze(segments_from_text("Machine learning is fun. " * 4 + "Data matters a lot here. " * 4))
    assert len(build_chapters(doc)) == 1
