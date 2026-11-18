"""Streamlit smoke test (offline, no network): runs the real app with a pasted transcript."""
from streamlit.testing.v1 import AppTest
from helpers import ROOT, SAMPLE


def _run():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    return at


def test_app_loads_with_hint():
    at = _run()
    assert not at.exception
    assert any("Analyze video" in i.value for i in at.info)


def test_analyze_pasted_transcript_and_all_tabs():
    at = _run()
    at.text_area[0].set_value(SAMPLE)
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert not at.error
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Chapters"]) >= 3
    assert any("Study" in b.label or "Generate" in b.label for b in at.button)
    gen = {b.label: b for b in at.button}
    gen["Generate quiz"].click()
    at.run()
    assert not at.exception, at.exception
    gen = {b.label: b for b in at.button}
    gen["Generate flashcards"].click()
    at.run()
    assert not at.exception, at.exception
    gen = {b.label: b for b in at.button}
    gen["Generate study notes"].click()
    at.run()
    assert not at.exception, at.exception


def test_invalid_url_shows_error():
    at = _run()
    at.text_input[1].set_value("https://example.com/nope") if len(at.text_input) > 1 else at.text_input[0].set_value("https://example.com/nope")
    at.button[0].click()
    at.run()
    assert at.error
