import pytest
from learntube.core import analyze
from learntube.ingest import get_video_id, segments_from_text
from learntube.llm import parse_json
from learntube.textproc import tokenize
from learntube.utils import fmt_time, linkify
from helpers import SAMPLE


@pytest.mark.parametrize("u", ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ?t=5",
                               "https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ",
                               "https://www.youtube.com/embed/dQw4w9WgXcQ"])
def test_video_id_formats(u):
    assert get_video_id(u) == "dQw4w9WgXcQ"


def test_invalid_url():
    with pytest.raises(ValueError):
        get_video_id("https://example.com/nope")


def test_pasted_transcript_keeps_timestamps():
    segs = segments_from_text(SAMPLE)
    assert len(segs) > 50 and segs[1]["start"] > 0
    assert segments_from_text("just plain text without stamps")[0]["start"] == 0


def test_plain_text_no_timestamps():
    doc = analyze(segments_from_text("Machine learning is great. " * 30))
    assert doc.sentences and doc.video_id is None


def test_hindi_tokenizer_keeps_devanagari():
    toks = tokenize("मशीन लर्निंग डेटा से सीखती है और यह बहुत उपयोगी है")
    assert "मशीन" in toks and "लर्निंग" in toks and "है" not in toks


def test_unpunctuated_captions_are_windowed():
    words = ("neural networks learn weights by gradient descent " * 40).split()
    segs = [{"text": " ".join(words[i:i + 6]), "start": float(i), "duration": 3.0} for i in range(0, len(words), 6)]
    doc = analyze(segs)
    assert max(len(s["text"].split()) for s in doc.sentences) <= 40


def test_time_helpers():
    assert fmt_time(3725) == "1:02:05"
    assert "(https://youtu.be/abc?t=75)" in linkify("see [01:15]", "abc")
    assert linkify("see [01:15]", None) == "see [01:15]"


def test_parse_json_handles_fences():
    assert parse_json("```json\n[{\"a\": 1}]\n```") == [{"a": 1}]
    assert parse_json("Sure! {\"x\": 2} done") == {"x": 2}
