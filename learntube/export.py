"""Exports: Markdown study pack and Anki-importable flashcards."""
import csv
import io

from .core import Doc
from .utils import fmt_time, linkify


def anki_csv(cards) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t")
    for c in cards:
        w.writerow([c["front"], c["back"]])
    return buf.getvalue()


def study_pack_markdown(doc: Doc, summary_md: str = "", notes_md: str = "", cards=None, quiz=None) -> str:
    out = [f"# {doc.title}", ""]
    if doc.video_id:
        out += [f"Source: https://www.youtube.com/watch?v={doc.video_id}", ""]
    out += ["## Chapters"]
    out += [f"- [{fmt_time(c['start'])}] {c['title']}" for c in doc.chapters] or ["- (single section)"]
    if summary_md:
        out += ["", "## Summary", linkify(summary_md, doc.video_id)]
    if notes_md:
        out += ["", linkify(notes_md, doc.video_id)]
    if cards:
        out += ["", "## Flashcards"] + [f"- **{c['front']}** -> {c['back']}" for c in cards]
    if quiz:
        out += ["", "## Quiz"]
        for i, q in enumerate(quiz, 1):
            out.append(f"{i}. {q['question']}")
            out += [f"   - {'**' if j == q['answer_index'] else ''}{o}{'**' if j == q['answer_index'] else ''}"
                    for j, o in enumerate(q["options"])]
    return "\n".join(out)
