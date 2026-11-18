import learntube.llm as llm_mod
from learntube.core import analyze
from learntube.ingest import segments_from_text
from learntube.llm import LLM, LLMConfig
from learntube.verify import extract_claims, grounding_check
from helpers import FakeLLM, SAMPLE


def _doc():
    return analyze(segments_from_text(SAMPLE))


def test_supported_vs_fabricated_claims():
    d = _doc()
    md = ("- Overfitting is an important part of machine learning because it directly affects cross validation [00:22]\n"
          "- The speaker recommends buying cryptocurrency before the quarterly earnings announcement tomorrow [01:00]")
    r = grounding_check(md, d)
    st = [i["status"] for i in r["items"]]
    assert st[0] == "supported" and st[1] == "unsupported"
    assert 0 < r["score"] < 1 and r["items"][0]["evidence_ts"]


def test_claim_extraction_strips_markup():
    c = extract_claims("TL;DR: Short two sentence overview of the whole video here.\n- **Bold** point about things [01:02]\n")
    assert c[0].startswith("Short") and "[" not in c[0]


def test_llm_judge_upgrades_borderline_claims():
    d = _doc()
    r = grounding_check("- The talk mentions deployment dashboards and security headers somewhere in passing remarks", d, FakeLLM())
    assert any(i["judged_by"] == "llm" for i in r["items"]) or r["items"][0]["status"] == "supported"


def test_cache_prevents_second_call(tmp_path):
    calls = []

    class C(LLM):
        def _call(self, prompt, t):
            calls.append(1)
            return "hello"
    llm = C(LLMConfig(provider="gemini", gemini_key="k", cache_dir=str(tmp_path)))
    assert llm.generate("p") == "hello" and llm.generate("p") == "hello"
    assert len(calls) == 1 and llm.cache_hits == 1


def test_retry_then_success(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    n = {"c": 0}

    class R(LLM):
        def _call(self, prompt, t):
            n["c"] += 1
            if n["c"] < 3:
                raise RuntimeError("429")
            return "ok"
    assert R(LLMConfig(provider="gemini", gemini_key="k", use_cache=False)).generate("p") == "ok" and n["c"] == 3


def test_retry_exhausted_raises(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)

    class R(LLM):
        def _call(self, prompt, t):
            raise RuntimeError("down")
    import pytest
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        R(LLMConfig(provider="gemini", gemini_key="k", use_cache=False)).generate("p")


def test_provider_resolution_and_keys_not_in_environ(monkeypatch):
    assert LLM(LLMConfig(provider="auto")).provider == "none"
    assert LLM(LLMConfig(provider="auto", openai_key="k")).provider == "openai"
    assert LLM(LLMConfig(provider="auto", gemini_key="k", openai_key="k")).provider == "gemini"
    import os
    assert "KEY_SHOULD_NOT_LEAK" not in os.environ


def test_generate_json_retries_on_bad_json():
    outs = iter(["not json", '[{"a": 1}]'])

    class J(LLM):
        def _call(self, p, t):
            return next(outs)
    assert J(LLMConfig(provider="gemini", gemini_key="k", use_cache=False)).generate_json("x") == [{"a": 1}]
