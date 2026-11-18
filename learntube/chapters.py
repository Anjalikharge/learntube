"""Automatic chapter detection (TextTiling-style topic segmentation) with timestamps."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .core import Doc, extract_keywords
from .textproc import tokenize


def segment(sentences, block: int = 5, smooth: int = 2, min_blocks: int = 4, max_chapters=None):
    """Return sorted sentence indices where chapters start (always includes 0).

    Sentences are grouped into blocks; the cosine similarity between the blocks on each side of a
    gap is computed, and deep similarity valleys become chapter boundaries.
    """
    n = len(sentences)
    nb = (n + block - 1) // block
    if nb < 2 * min_blocks:
        return [0]
    blocks = [" ".join(s["text"] for s in sentences[i:i + block]) for i in range(0, n, block)]
    try:
        X = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                            sublinear_tf=True).fit_transform(blocks).toarray()
    except ValueError:
        return [0]
    sims = []
    for g in range(1, nb):
        L, R = X[max(0, g - smooth):g].sum(0), X[g:g + smooth].sum(0)
        d = np.linalg.norm(L) * np.linalg.norm(R)
        sims.append(float(L @ R / d) if d > 0 else 0.0)
    sims = np.array(sims)
    depth = np.zeros_like(sims)
    for i in range(len(sims)):
        l = i
        while l > 0 and sims[l - 1] >= sims[l]:
            l -= 1
        r = i
        while r < len(sims) - 1 and sims[r + 1] >= sims[r]:
            r += 1
        depth[i] = (sims[l] - sims[i]) + (sims[r] - sims[i])
    cutoff = depth.mean() + 0.5 * depth.std()
    chosen = []
    for i in np.argsort(depth)[::-1]:
        if depth[i] <= 0 or depth[i] < cutoff:
            break
        g = int(i) + 1
        if g < min_blocks or nb - g < min_blocks:
            continue
        if all(abs(g - c) >= min_blocks for c in chosen):
            chosen.append(g)
        if max_chapters and len(chosen) >= max_chapters - 1:
            break
    if not chosen and nb >= 3 * min_blocks:  # flat similarity: fall back to even splits
        parts = max(2, min(max_chapters or 6, nb // (2 * min_blocks)))
        chosen = [round(nb * k / parts) for k in range(1, parts)]
    return [0] + sorted(c * block for c in chosen)


def build_chapters(doc: Doc, llm=None, language: str = "Same as video", max_chapters=None):
    """Fill doc.chapters with [{index,start,start_idx,end_idx,keywords,title}] and return them."""
    n = len(doc.sentences)
    if max_chapters is None:
        max_chapters = max(3, min(12, round(doc.minutes / 6))) if doc.minutes > 0 else max(3, min(10, n // 40))
    bounds = segment(doc.sentences, max_chapters=max_chapters)
    chapters = []
    for i, b in enumerate(bounds):
        e = bounds[i + 1] if i + 1 < len(bounds) else n
        sents = doc.sentences[b:e]
        kws = extract_keywords(sents, 4)
        chapters.append({"index": i + 1, "start": sents[0]["start"], "start_idx": b, "end_idx": e,
                         "keywords": kws, "title": ", ".join(kws[:3]).title() or f"Part {i + 1}"})
    if llm is not None and llm.available():
        try:
            lang = "" if language in (None, "", "Same as video") else f" Write the titles in {language}."
            desc = "\n".join(
                f"{c['index']}. keywords: {', '.join(c['keywords'])}; opening: "
                f"{' '.join(' '.join(s['text'] for s in doc.sentences[c['start_idx']:c['end_idx']][:4]).split()[:60])}"
                for c in chapters)
            titles = llm.generate_json(
                "TASK:CHAPTER_TITLES\nGive a short descriptive title (max 7 words) for each numbered section "
                f"of a video. Return a JSON array of {len(chapters)} strings in order.{lang}\n\n{desc}")
            if isinstance(titles, list) and len(titles) == len(chapters):
                for c, t in zip(chapters, titles):
                    c["title"] = str(t).strip() or c["title"]
        except Exception:
            pass  # keyword titles stay
    doc.chapters = chapters
    return chapters
