from learntube.chapters import build_chapters
from learntube.core import analyze
from learntube.export import anki_csv, study_pack_markdown
from learntube.ingest import segments_from_text
from learntube.learn import (coverage_text, generate_flashcards, generate_notes, generate_quiz, mindmap_mermaid)
from learntube.llm import LLM, LLMConfig
from helpers import FakeLLM, SAMPLE

OFF = LLM(LLMConfig(provider="none"))


def _doc():
    d = analyze(segments_from_text(SAMPLE), None, "Demo lecture")
    build_chapters(d, OFF)
    return d


def test_offline_quiz_valid():
    q, note = generate_quiz(_doc(), OFF, 4)
    assert q and all(len(x["options"]) == 4 and 0 <= x["answer_index"] < 4 for x in q) and not note


def test_llm_quiz_validated():
    q, _ = generate_quiz(_doc(), FakeLLM(), 5)
    assert len(q) == 5


def test_invalid_llm_quiz_items_dropped():
    class Bad(FakeLLM):
        def _call(self, p, t):
            return '[{"question": "x", "options": ["a"], "answer_index": 9}]' if p.startswith("TASK:QUIZ") else "ok"
    q, note = generate_quiz(_doc(), Bad(), 3)
    assert q and all(len(x["options"]) == 4 for x in q)  # fell back to offline cloze


def test_flashcards_and_anki_export():
    cards, _ = generate_flashcards(_doc(), OFF, 6)
    assert cards and all(c["front"] and c["back"] for c in cards)
    assert anki_csv(cards).count("\n") == len(cards)
    llm_cards, _ = generate_flashcards(_doc(), FakeLLM(), 10)
    assert len(llm_cards) == 10


def test_notes_cover_every_chapter():
    d = _doc()
    md, _ = generate_notes(d, FakeLLM())
    assert md.count("## [") == len(d.chapters)
    md_off, _ = generate_notes(d, OFF)
    assert md_off.count("## [") == len(d.chapters)


def test_coverage_text_reaches_end_of_long_video():
    d = _doc()
    d.words = 20000  # pretend long
    txt = coverage_text(d, budget=1200)
    last_topic_term = d.sentences[-1]["text"].split()[0]
    assert len(txt.split()) < 2500 and d.chapters[-1]["start_idx"] < len(d.sentences)
    assert txt.count("[") >= len(d.chapters)


def test_mindmap_is_valid_mermaid_without_bad_chars():
    m = mindmap_mermaid(_doc())
    assert m.startswith("mindmap") and "root((" in m
    assert not any(c in m.split("root((", 1)[1].split("\n", 1)[1] for c in "()[]{}\"")


def test_study_pack_contains_sections():
    d = _doc()
    md = study_pack_markdown(d, "- point [00:10]", "", [{"front": "f", "back": "b"}], [])
    assert "## Chapters" in md and "## Summary" in md and "## Flashcards" in md
