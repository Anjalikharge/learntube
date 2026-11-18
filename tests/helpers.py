import json
import re
from pathlib import Path

from learntube.llm import LLM, LLMConfig

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = (ROOT / "eval" / "sample_transcript.txt").read_text(encoding="utf-8")


class FakeLLM(LLM):
    """Deterministic stand-in for a real provider (no network)."""

    def __init__(self):
        super().__init__(LLMConfig(provider="gemini", gemini_key="x", use_cache=False))
        self.prompts = []

    def _call(self, prompt, temperature):
        self.prompts.append(prompt)
        task = re.match(r"TASK:(\w+)", prompt)
        task = task.group(1) if task else ""
        body = prompt.split("TRANSCRIPT:\n", 1)[-1] if "TRANSCRIPT:\n" in prompt else prompt
        plain = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", body)
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", plain) if len(s.split()) > 5]
        if task == "CHAPTER_TITLES":
            n = int(re.search(r"array of (\d+) strings", prompt).group(1))
            return json.dumps([f"Topic {i + 1}" for i in range(n)])
        if task in ("SUMMARY", "SUMMARY_REDUCE", "CHAPTER_SUMMARY", "NOTES"):
            pts = sents[:4] or ["Nothing"]
            return "TL;DR: " + " ".join(pts[:2]) + "\n" + "\n".join(f"- {p} [00:10]" for p in pts) + "\nKey takeaway: " + pts[0]
        if task == "QUIZ":
            return json.dumps([{"question": f"Question {i}?", "options": ["A", "B", "C", "D"], "answer_index": i % 4,
                                "explanation": "because"} for i in range(5)])
        if task == "FLASHCARDS":
            return json.dumps([{"front": f"Front {i}", "back": f"Back {i}"} for i in range(10)])
        if task == "JUDGE":
            return "supported"
        if task == "CHAT":
            return "The video explains this around [00:10]."
        return "ok"
