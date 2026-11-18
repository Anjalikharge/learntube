# 🎓 LearnTube: turn any YouTube video into a course

Paste a YouTube link and get **chapters with timestamps, a summary, study notes, a quiz, flashcards, a mind map and a cited
chat**, plus a **grounding check** that verifies each summary claim against the transcript.
Works **fully offline** (no API key); Gemini / OpenAI / Ollama make it better.

## Features
| Feature | How it works |
|---|---|
| LangGraph pipeline | input -> transcript (Whisper fallback) -> analysis -> chapters -> summary (LLM or offline) -> grounding check; conditional routing and a timed trace |
| Auto chapters | TextTiling-style topic segmentation (TF-IDF block similarity valleys) + LLM titles |
| Summaries | Extractive TextRank (offline) or LLM; long videos use parallel per-chapter map-reduce; bullets carry clickable `[mm:ss]` links |
| Hybrid RAG chat | BM25 + TF-IDF (+ dense embeddings with Gemini/OpenAI) fused by Reciprocal Rank Fusion; cites timestamps; can search a whole library of videos |
| Grounding check | each summary claim is matched to its best transcript passage and marked supported / partial / unsupported |
| Learning tools | notes (whole video covered), quiz with difficulty, flashcards + Anki export, Mermaid mind map |
| Languages | English, Hindi, Marathi input; output language selectable |
| Reliability | disk cache, retries with backoff, graceful fallback to offline mode, API keys never stored in `os.environ` |
| Evaluation | `python -m learntube.evaluate`: ROUGE, compression, grounding, Hit@k / MRR for TF-IDF vs BM25 vs hybrid |

## Run
```bash
pip install -r requirements.txt
cp .env.example .env        # optional: add GEMINI_API_KEY or OPENAI_API_KEY
streamlit run app.py
```
Docker: `docker build -t learntube . && docker run -p 8501:8501 --env-file .env learntube`
Videos without captions: `pip install -r requirements-whisper.txt` and tick *Whisper fallback* in the sidebar.

## Test
```bash
pip install -r requirements-dev.txt
python -m pytest -q          # 50 offline tests (LLM is mocked)
```

## Evaluate (numbers for your report)
Copy `eval/sample.json`, add your own videos (`url`, a short `reference_summary`, and questions with `answer_terms`), then:
```bash
python -m learntube.evaluate my_eval.json --out results.md
```
`eval/sample.json` uses a synthetic demo transcript, so its numbers are only a demo of the harness.

## Project layout
`learntube/` ingest, textproc, core, chapters, retrieval, summarize, learn, verify, chat, graph, llm, evaluate, export · `app.py` UI · `tests/` · `eval/`

## Known limitations
- YouTube may block transcript requests from cloud servers; run locally or paste the transcript.
- Summary quality depends on caption quality and the chosen LLM.
- Grounding is lexical (plus optional LLM judging); it flags unsupported claims but is not a proof of truth.
