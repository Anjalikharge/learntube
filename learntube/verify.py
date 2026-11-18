"""Grounding check: is each claim in the summary supported by the transcript?

Lexical support = share of the claim's content words found in the best-matching transcript passage.
Borderline claims can optionally be judged by the LLM.
"""
import re

from .core import Doc
from .textproc import tokenize
from .utils import fmt_time

_TS = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")


def extract_claims(summary_md: str):
    claims = []
    for line in summary_md.splitlines():
        line = _TS.sub("", line).strip()
        line = re.sub(r"^[-*\u2022]+\s*|^\d+[.)]\s*|^#+\s*", "", line).replace("**", "").strip()
        line = re.sub(r"^(TL;DR|Key takeaway)\s*:\s*", "", line, flags=re.I)
        if len(line.split()) >= 5:
            claims.append(line)
    return claims


def grounding_check(summary_md: str, doc: Doc, llm=None, strong: float = 0.6, weak: float = 0.35,
                    max_llm_calls: int = 8):
    """Return {'score', 'supported_pct', 'items': [{claim, coverage, status, evidence_ts, evidence}]}."""
    items = []
    for claim in extract_claims(summary_md):
        ctoks = set(tokenize(claim))
        if not ctoks:
            continue
        best, best_cov = None, 0.0
        for p in doc.index.search(claim, k=3, mode="bm25"):
            cov = len(ctoks & set(tokenize(p["text"]))) / len(ctoks)
            if cov > best_cov:
                best, best_cov = p, cov
        status = "supported" if best_cov >= strong else "partial" if best_cov >= weak else "unsupported"
        items.append({"claim": claim, "coverage": round(best_cov, 2), "status": status,
                      "evidence_ts": fmt_time(best["start"]) if best else "", "evidence": best["text"] if best else "",
                      "judged_by": "lexical"})
    if llm is not None and llm.available():
        calls = 0
        for it in items:
            if it["status"] == "supported" or calls >= max_llm_calls or not it["evidence"]:
                continue
            try:
                v = llm.generate(f"TASK:JUDGE\nDoes the EXCERPT support the CLAIM? Answer with one word: "
                                 f"supported, partial or unsupported.\nCLAIM: {it['claim']}\nEXCERPT: {it['evidence']}")
                calls += 1
                v = v.strip().lower().split()[0].strip(".,:;")
                if v in ("supported", "partial", "unsupported"):
                    it["status"], it["judged_by"] = v, "llm"
            except Exception:
                break
    n = len(items)
    sup = sum(1 for i in items if i["status"] == "supported")
    part = sum(1 for i in items if i["status"] == "partial")
    return {"score": round((sup + 0.5 * part) / n, 2) if n else 0.0,
            "supported_pct": round(100 * sup / n) if n else 0, "items": items}
