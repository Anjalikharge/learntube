"""Learning tools: study notes, quiz, flashcards, mind map (LLM with offline fallbacks)."""
import random
import re
from concurrent.futures import ThreadPoolExecutor

from .core import Doc, extract_keywords
from .summarize import RULES, chapter_text, extractive_summary, lang_note
from .textproc import timestamped_text
from .utils import fmt_time


def coverage_text(doc: Doc, budget: int = 6000) -> str:
    """Timestamped text that covers the WHOLE video (a share of words from every chapter)."""
    if doc.words <= budget:
        return timestamped_text(doc.sentences)
    chs = doc.chapters or [{"start_idx": 0, "end_idx": len(doc.sentences)}]
    per = max(200, budget // len(chs))
    parts = [" ".join(timestamped_text(doc.sentences[c["start_idx"]:c["end_idx"]]).split()[:per]) for c in chs]
    return "\n\n".join(parts)


# ---------------------------------------------------------------- notes
def offline_notes(doc: Doc) -> str:
    md = [f"# Study Notes: {doc.title}\n"]
    for ch in doc.chapters or [{"index": 1, "start": 0, "start_idx": 0, "end_idx": len(doc.sentences),
                                "title": "Overview", "keywords": doc.keywords[:4]}]:
        sents = doc.sentences[ch["start_idx"]:ch["end_idx"]]
        md.append(f"## [{fmt_time(ch['start'])}] {ch['title']}\n**Keywords:** {', '.join(ch['keywords'])}\n")
        md += [f"- {s['text']}" for s in extractive_summary(sents, 3)]
        md.append("")
    return "\n".join(md)


def generate_notes(doc: Doc, llm, language="Same as video"):
    """Returns (markdown, note). LLM notes are written per chapter, so the whole video is covered."""
    if not llm.available():
        return offline_notes(doc), ""
    try:
        chs = doc.chapters or [{"index": 1, "start": 0, "start_idx": 0, "end_idx": len(doc.sentences),
                                "title": doc.title}]

        def one(ch):
            text = " ".join(chapter_text(doc, ch).split()[:1500])
            return llm.generate(f"TASK:NOTES\nWrite concise study notes in Markdown for this section: bullet "
                                f"points, **bold** key terms, keep [mm:ss] markers. {RULES}{lang_note(language)}"
                                f"\n\nTRANSCRIPT:\n{text}")
        with ThreadPoolExecutor(max_workers=4) as ex:
            bodies = list(ex.map(one, chs))
        md = [f"# Study Notes: {doc.title}\n"]
        for ch, b in zip(chs, bodies):
            md.append(f"## [{fmt_time(ch['start'])}] {ch['title']}\n{b}\n")
        return "\n".join(md), ""
    except Exception as e:
        return offline_notes(doc), f"LLM unavailable ({e}). Offline notes shown."


# ---------------------------------------------------------------- quiz
def _valid_q(q):
    return (isinstance(q, dict) and isinstance(q.get("options"), list) and len(q["options"]) == 4
            and isinstance(q.get("answer_index"), int) and 0 <= q["answer_index"] < 4 and q.get("question"))


def offline_quiz(doc: Doc, n: int = 5):
    kws = [k for k in extract_keywords(doc.sentences, 25) if " " not in k] or ["concept"]
    quiz = []
    for s in extractive_summary(doc.sentences, n * 3):
        hit = next((k for k in kws if re.search(rf"\b{re.escape(k)}\w*\b", s["text"], re.I)), None)
        if not hit:
            continue
        pat = re.compile(rf"\b{re.escape(hit)}\w*\b", re.I)
        answer = pat.search(s["text"]).group()
        pool = [k for k in kws if k.lower() != hit.lower()]
        distract = random.sample(pool, min(3, len(pool)))
        while len(distract) < 3:
            distract.append(f"None of these ({len(distract) + 1})")
        opts = distract + [answer]
        random.shuffle(opts)
        quiz.append({"question": "Fill in the blank: " + pat.sub("_____", s["text"], count=1), "options": opts,
                     "answer_index": opts.index(answer), "explanation": f"Original sentence: {s['text']}",
                     "ts": s["start"]})
        if len(quiz) >= n:
            break
    return quiz


def generate_quiz(doc: Doc, llm, n: int = 5, difficulty: str = "medium", language="Same as video"):
    """Returns (questions, note). Each: {question, options[4], answer_index, explanation}."""
    if llm.available():
        try:
            raw = llm.generate_json(
                f"TASK:QUIZ\nCreate {n} {difficulty}-difficulty multiple-choice questions that test understanding "
                "of this video, spread across the WHOLE transcript. Return ONLY a JSON array; each item: "
                '{"question": str, "options": [4 strings], "answer_index": 0-3, "explanation": str}. '
                f"Use ONLY the transcript.{lang_note(language)}\n\nTRANSCRIPT:\n{coverage_text(doc)}", 0.4)
            quiz = [q for q in raw if _valid_q(q)]
            if quiz:
                return quiz[:n], ""
        except Exception as e:
            return offline_quiz(doc, n), f"LLM unavailable ({e}). Offline cloze quiz shown."
    return offline_quiz(doc, n), ""


# ---------------------------------------------------------------- flashcards
def offline_flashcards(doc: Doc, n: int = 10):
    cards = []
    for q in offline_quiz(doc, n):
        cards.append({"front": q["question"].replace("Fill in the blank: ", ""),
                      "back": q["options"][q["answer_index"]], "ts": q.get("ts", 0)})
    return cards


def generate_flashcards(doc: Doc, llm, n: int = 10, language="Same as video"):
    """Returns (cards, note). Each: {front, back}."""
    if llm.available():
        try:
            raw = llm.generate_json(
                f"TASK:FLASHCARDS\nCreate {n} study flashcards (question on the front, short answer on the back) "
                'covering the whole video. Return ONLY a JSON array of {"front": str, "back": str}. '
                f"Use ONLY the transcript.{lang_note(language)}\n\nTRANSCRIPT:\n{coverage_text(doc)}", 0.3)
            cards = [c for c in raw if isinstance(c, dict) and c.get("front") and c.get("back")]
            if cards:
                return cards[:n], ""
        except Exception as e:
            return offline_flashcards(doc, n), f"LLM unavailable ({e}). Offline flashcards shown."
    return offline_flashcards(doc, n), ""


# ---------------------------------------------------------------- mind map
def _label(s: str) -> str:
    s = re.sub(r"[()\[\]{}\"'`:;#<>|]", " ", s)
    return re.sub(r"\s+", " ", s).strip()[:48] or "topic"


def mindmap_mermaid(doc: Doc) -> str:
    """Mermaid 'mindmap' source: video -> chapters -> keywords (works offline)."""
    lines = ["mindmap", f"  root(({_label(doc.title)}))"]
    chs = doc.chapters or [{"title": "Main topics", "keywords": doc.keywords[:6]}]
    for ch in chs:
        lines.append(f"    {_label(ch['title'])}")
        for k in ch["keywords"][:4]:
            lines.append(f"      {_label(k)}")
    return "\n".join(lines)
