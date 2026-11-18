"""Evaluation harness (for the report's Results chapter).

Usage:  python -m learntube.evaluate eval/sample.json [--out results.md]

Input JSON: a list of {"name", "url" OR "transcript_file", "reference_summary", "qa": [{"q", "answer_terms": [..]}]}
Reports per video: compression ratio, ROUGE-1/2/L (extractive vs LLM summary), grounding score, chapter count,
latency, and retrieval quality (Hit@k, MRR) for TF-IDF vs BM25 vs hybrid (+dense when an API key is set).
A retrieved passage counts as relevant if it contains ANY of the question's answer_terms.
"""
import argparse
import json
import statistics
import time

from .chapters import build_chapters
from .core import Doc, analyze
from .ingest import fetch_title, fetch_transcript, segments_from_text
from .llm import LLM
from .summarize import abstractive_summary, extractive_markdown
from .verify import grounding_check


def rouge(candidate: str, reference: str):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return {k: round(v.fmeasure, 3) for k, v in sc.score(reference, candidate).items()}


def retrieval_eval(doc: Doc, qa, k: int = 4, modes=None):
    modes = modes or [m for m in ("tfidf", "bm25", "dense", "hybrid") if m != "dense" or doc.index.has_dense]
    res = {}
    for mode in modes:
        hits, rr = 0, []
        for item in qa:
            terms = [t.lower() for t in item["answer_terms"]]
            got = doc.index.search(item["q"], k=k, mode=mode)
            rank = next((i + 1 for i, p in enumerate(got) if any(t in p["text"].lower() for t in terms)), None)
            hits += rank is not None
            rr.append(1 / rank if rank else 0.0)
        n = max(len(qa), 1)
        res[mode] = {"hit@k": round(hits / n, 3), "mrr": round(sum(rr) / n, 3)}
    return res


def evaluate_video(item, llm: LLM, k: int = 4):
    t0 = time.time()
    if item.get("transcript_file"):
        segs = segments_from_text(open(item["transcript_file"], encoding="utf-8").read())
        vid, title = None, item["name"]
    else:
        vid, segs = fetch_transcript(item["url"])
        title = fetch_title(vid) or item["name"]
    doc = analyze(segs, vid, title, embed_fn=llm.embed if llm.provider in ("gemini", "openai") else None)
    build_chapters(doc, llm)
    row = {"name": item["name"], "minutes": round(doc.minutes, 1), "words": doc.words, "chapters": len(doc.chapters)}
    ref = item.get("reference_summary", "")
    summaries = {"extractive": extractive_markdown(doc, "medium")}
    if llm.available():
        t1 = time.time()
        summaries["llm"] = abstractive_summary(doc, llm, "medium")
        row["llm_seconds"] = round(time.time() - t1, 1)
    row["summaries"] = {}
    for name, text in summaries.items():
        words = len(text.split())
        r = {"words": words, "compression": round(words / max(doc.words, 1), 3),
             "grounding": grounding_check(text, doc)["score"]}
        if ref:
            r.update(rouge(text, ref))
        row["summaries"][name] = r
    if item.get("qa"):
        row["retrieval"] = retrieval_eval(doc, item["qa"], k)
    row["total_seconds"] = round(time.time() - t0, 1)
    return row


def to_markdown(rows) -> str:
    out = ["## Summarisation results", "",
           "| Video | Min | Words | Chapters | Method | Words out | Compression | ROUGE-1 | ROUGE-2 | ROUGE-L | Grounding |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for m, s in r["summaries"].items():
            out.append(f"| {r['name']} | {r['minutes']} | {r['words']} | {r['chapters']} | {m} | {s['words']} | "
                       f"{s['compression']} | {s.get('rouge1', '-')} | {s.get('rouge2', '-')} | "
                       f"{s.get('rougeL', '-')} | {s['grounding']} |")
    out += ["", "## Retrieval results (Hit@k / MRR)", "", "| Video | Mode | Hit@k | MRR |", "|---|---|---|---|"]
    for r in rows:
        for mode, s in r.get("retrieval", {}).items():
            out.append(f"| {r['name']} | {mode} | {s['hit@k']} | {s['mrr']} |")
    modes = {}
    for r in rows:
        for mode, s in r.get("retrieval", {}).items():
            modes.setdefault(mode, []).append(s)
    if len(rows) > 1 and modes:
        out += ["", "**Average across videos**", "", "| Mode | Hit@k | MRR |", "|---|---|---|"]
        for mode, ss in modes.items():
            out.append(f"| {mode} | {statistics.mean(x['hit@k'] for x in ss):.3f} | "
                       f"{statistics.mean(x['mrr'] for x in ss):.3f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="")
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    llm = LLM()
    rows = [evaluate_video(i, llm, a.k) for i in json.load(open(a.config, encoding="utf-8"))]
    md = to_markdown(rows)
    print(md)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(md)
        json.dump(rows, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=2)


if __name__ == "__main__":
    main()
