"""LearnTube - turn any YouTube video into a course.  Run: streamlit run app.py"""
import html

import streamlit as st
import streamlit.components.v1 as components

from learntube import chat as chatmod
from learntube import graph as graphmod
from learntube import learn
from learntube.chapters import build_chapters  # noqa: F401  (re-exported for convenience)
from learntube.export import anki_csv, study_pack_markdown
from learntube.llm import LLM, LLMConfig
from learntube.summarize import extractive_summary, summarize_chapters
from learntube.utils import fmt_time, linkify, ts_url

st.set_page_config(page_title="LearnTube", page_icon="🎓", layout="wide")
st.title("🎓 LearnTube: turn any YouTube video into a course")
st.caption("Chapters, summary, notes, quiz, flashcards, mind map and cited chat. Works offline; AI makes it better.")

ENGINES = {"Auto": "auto", "Gemini": "gemini", "OpenAI": "openai", "Ollama (local)": "ollama", "Offline only": "none"}
LANGS = ["Same as video", "English", "Hindi", "Marathi"]


def mermaid(code: str, height: int = 460):
    components.html(
        f"""<pre class="mermaid">{html.escape(code)}</pre>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
mermaid.initialize({{startOnLoad: true}});
</script>""", height=height, scrolling=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    engine = st.selectbox("AI engine", list(ENGINES))
    key = st.text_input("API key (kept only in this session)", type="password")
    language = st.selectbox("Output language", LANGS)
    length = st.select_slider("Summary length", ["short", "medium", "detailed"], value="medium")
    use_whisper = st.checkbox("Whisper fallback if no captions", value=False,
                              help="Needs: pip install yt-dlp faster-whisper")
    use_dense = st.checkbox("Semantic search (embeddings)", value=True,
                            help="Used only with Gemini/OpenAI; otherwise keyword search (BM25 + TF-IDF).")
    n_q = st.slider("Quiz questions", 3, 10, 5)
    difficulty = st.select_slider("Quiz difficulty", ["easy", "medium", "hard"], value="medium")
    n_cards = st.slider("Flashcards", 5, 20, 10)

cfg = LLMConfig.from_env()
cfg.provider = ENGINES[engine]
if key:  # per-session key: never written to os.environ
    if cfg.provider == "openai" or (cfg.provider == "auto" and key.startswith("sk-")):
        cfg.openai_key = key
    else:
        cfg.gemini_key = key
llm = LLM(cfg)
st.sidebar.caption(f"Active engine: **{llm.provider}**" + ("" if llm.available() else " (offline mode)"))

# ------------------------------------------------------------------ input
ss = st.session_state
ss.setdefault("library", {})
url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
with st.expander("Transcript not available? Paste it (optional [mm:ss] at the start of lines)"):
    pasted = st.text_area("Transcript text", height=140)

if st.button("Analyze video", type="primary"):
    for k in ("notes", "quiz", "cards", "chat", "result"):
        ss.pop(k, None)
    final = {}
    with st.status("Running the LangGraph pipeline...", expanded=True) as status:
        g = graphmod.build_graph(llm)
        inputs = {"url": url, "pasted": pasted, "length": length, "language": language,
                  "use_whisper": use_whisper, "use_dense": use_dense}
        for state in g.stream(inputs, stream_mode="values"):
            final = state
            if state.get("trace"):
                last = state["trace"][-1]
                status.write(f"{'✅' if last['ok'] else '❌'} {last['step']} ({last['sec']}s)")
        status.update(label="Done" if not final.get("error") else "Failed",
                      state="complete" if not final.get("error") else "error")
    ss["result"] = final
    ss["graph_mermaid"] = g.get_graph().draw_mermaid()
    if final.get("doc"):
        ss["library"][final["doc"].key] = final["doc"]

res = ss.get("result")
if not res:
    st.info("Paste a YouTube link and click **Analyze video**.")
    st.stop()
if res.get("error"):
    st.error(res["error"])
    st.stop()

doc = res["doc"]
if res.get("note"):
    st.warning(res["note"])
gr = res.get("grounding", {})

st.subheader(doc.title)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Words", f"{doc.words:,}")
c2.metric("Duration", f"{doc.minutes:.0f} min" if doc.video_id else "n/a")
c3.metric("Chapters", len(doc.chapters))
c4.metric("Grounded claims", f"{gr['supported_pct']}%" if "supported_pct" in gr else "n/a")
if doc.video_id:
    st.video(f"https://www.youtube.com/watch?v={doc.video_id}")

tabs = st.tabs(["📝 Summary", "📚 Chapters", "📒 Notes", "❓ Quiz", "🃏 Flashcards", "🧠 Mind map",
                "💬 Chat", "✅ Grounding", "⚙️ Pipeline"])

# ---- summary
with tabs[0]:
    st.caption(f"Engine: {res.get('summary_engine', '')}")
    st.markdown(linkify(res["summary"], doc.video_id))
    st.markdown("**Keywords:** " + " ".join(f"`{k}`" for k in doc.keywords))
    cols = st.columns(2)
    cols[0].download_button("Download summary (.md)", res["summary"], "summary.md")
    cols[1].download_button("Download study pack (.md)",
                            study_pack_markdown(doc, res["summary"], ss.get("notes", ("", ""))[0],
                                                (ss.get("cards") or ([], ""))[0], (ss.get("quiz") or ([], ""))[0]),
                            "study_pack.md")

# ---- chapters
with tabs[1]:
    if llm.available() and st.button("Summarize every chapter with AI"):
        with st.spinner("Summarizing chapters in parallel..."):
            try:
                summarize_chapters(doc, llm, language)
            except Exception as e:
                st.warning(f"LLM unavailable ({e})")
    for ch in doc.chapters:
        label = f"[{fmt_time(ch['start'])}] {ch['title']}"
        with st.expander(label, expanded=ch["index"] == 1):
            if doc.video_id:
                st.markdown(f"[▶ Watch from {fmt_time(ch['start'])}]({ts_url(doc.video_id, ch['start'])})")
            st.markdown("**Keywords:** " + ", ".join(ch["keywords"]))
            if ch.get("summary"):
                st.markdown(linkify(ch["summary"], doc.video_id))
            else:
                for s in extractive_summary(doc.sentences[ch["start_idx"]:ch["end_idx"]], 2):
                    st.markdown(f"- {s['text']}")

# ---- notes
with tabs[2]:
    if st.button("Generate study notes"):
        with st.spinner("Writing notes..."):
            ss["notes"] = learn.generate_notes(doc, llm, language)
    if "notes" in ss:
        md, note = ss["notes"]
        if note:
            st.warning(note)
        st.markdown(linkify(md, doc.video_id))
        st.download_button("Download notes (.md)", md, "notes.md")

# ---- quiz
with tabs[3]:
    if st.button("Generate quiz"):
        with st.spinner("Creating questions..."):
            ss["quiz"] = learn.generate_quiz(doc, llm, n_q, difficulty, language)
            ss["quiz_id"] = ss.get("quiz_id", 0) + 1
    if "quiz" in ss:
        quiz, note = ss["quiz"]
        if note:
            st.warning(note)
        if not quiz:
            st.info("Could not build questions from this transcript.")
        else:
            qid = ss["quiz_id"]
            with st.form(f"quiz_form_{qid}"):
                answers = [st.radio(f"**Q{i + 1}. {q['question']}**", q["options"], index=None, key=f"q{qid}_{i}")
                           for i, q in enumerate(quiz)]
                submitted = st.form_submit_button("Check answers")
            if submitted:
                score = 0
                for q, a in zip(quiz, answers):
                    ok = a == q["options"][q["answer_index"]]
                    score += ok
                    (st.success if ok else st.error)(
                        f"{'Correct' if ok else 'Answer: ' + q['options'][q['answer_index']]}. {q.get('explanation', '')}")
                st.subheader(f"Score: {score}/{len(quiz)}")

# ---- flashcards
with tabs[4]:
    if st.button("Generate flashcards"):
        with st.spinner("Creating flashcards..."):
            ss["cards"] = learn.generate_flashcards(doc, llm, n_cards, language)
    if "cards" in ss:
        cards, note = ss["cards"]
        if note:
            st.warning(note)
        for i, c in enumerate(cards, 1):
            with st.expander(f"Card {i}: {c['front']}"):
                st.markdown(c["back"])
        st.download_button("Export for Anki (.tsv)", anki_csv(cards), "flashcards.tsv")

# ---- mind map
with tabs[5]:
    mm = learn.mindmap_mermaid(doc)
    mermaid(mm, 520)
    with st.expander("Mermaid source"):
        st.code(mm)

# ---- chat
with tabs[6]:
    multi_ok = len(ss["library"]) > 1
    scope = st.radio("Search in", ["This video", f"All analysed videos ({len(ss['library'])})"], horizontal=True) \
        if multi_ok else "This video"
    modes = ["hybrid", "bm25", "tfidf"] + (["dense"] if doc.index.has_dense else [])
    mode = st.selectbox("Retrieval", modes, help="hybrid = BM25 + TF-IDF (+ embeddings) fused with RRF")
    hist = ss.setdefault("chat", [])
    for role, msg in hist:
        st.chat_message(role).markdown(linkify(msg, doc.video_id) if scope == "This video" else msg)
    q = st.chat_input("Ask something about the video...")
    if q:
        st.chat_message("user").markdown(q)
        multi = scope != "This video"
        index = (chatmod.build_library_index(list(ss["library"].values()), chatmod.embed_fn_for(llm, use_dense))
                 if multi else doc.index)
        with st.spinner("Thinking..."):
            ans, passages, note = chatmod.answer(q, index, llm, hist, mode, language=language, multi=multi)
        with st.chat_message("assistant"):
            st.markdown(linkify(ans, doc.video_id) if not multi else ans)
            if passages:
                with st.expander("Sources"):
                    for p in passages:
                        link = ts_url(p["video_id"], p["start"]) if p.get("video_id") else ""
                        st.markdown(f"**{p.get('title', '')} [{fmt_time(p['start'])}]({link})**  \n{p['text']}"
                                    if link else f"**[{fmt_time(p['start'])}]**  \n{p['text']}")
        if note:
            st.warning(note)
        hist += [("user", q), ("assistant", ans)]

# ---- grounding
with tabs[7]:
    if "skipped" in gr:
        st.info(gr["skipped"])
    else:
        st.write("Each claim in the summary is checked against the transcript (word overlap with its best-matching "
                 "passage; borderline claims are judged by the AI when available).")
        st.metric("Grounding score", gr.get("score", 0))
        icon = {"supported": "🟢", "partial": "🟡", "unsupported": "🔴"}
        for it in gr.get("items", []):
            with st.expander(f"{icon[it['status']]} {it['claim'][:110]}"):
                st.write(f"Status: **{it['status']}** (coverage {it['coverage']}, by {it['judged_by']})")
                if it["evidence"]:
                    st.caption(f"Best evidence at {it['evidence_ts']}:")
                    st.write(it["evidence"])

# ---- pipeline
with tabs[8]:
    st.write("LangGraph workflow (conditional routing: Whisper fallback, LLM vs offline summary).")
    mermaid(ss.get("graph_mermaid", ""), 520)
    st.dataframe(res.get("trace", []), use_container_width=True)
    st.caption(f"LLM calls: {llm.calls}, cache hits: {llm.cache_hits}")
