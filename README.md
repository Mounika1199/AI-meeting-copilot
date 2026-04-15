# AI Meeting Copilot

An LLM-powered meeting assistant that answers natural-language questions about meeting transcripts using a multi-stage Retrieval-Augmented Generation (RAG) pipeline with temporal and speaker-aware filtering.

## Features

- **RAG pipeline** — semantic search over chunked transcripts with FAISS indexing and LRU caching
- **Intent detection** — automatically classifies queries as temporal, speaker-specific, or synthesis (summaries / action items)
- **Speaker filtering** — fuzzy name matching to find what a specific participant said
- **Temporal filtering** — extracts clock times and relative references ("after 15:30", "in the last 10 minutes") to scope retrieval
- **Cross-encoder reranking** — BGE reranker re-scores retrieved chunks before context assembly
- **Streaming chat UI** — real-time WebSocket-based response streaming
- **Evaluation dashboard** — built-in RAGAS metrics (faithfulness, relevancy, context precision, context relevance)

## Architecture

```
app.py                  FastAPI app with /ws and /ws/eval WebSocket endpoints
config.py               Environment-driven configuration
core/
  models.py             Loads spaCy, sentence embeddings, reranker, and Ollama LLM
  logging_config.py     Timestamped rotating log setup
pipeline/
  transcript.py         Parses raw transcript text into a DataFrame
  chunking.py           Speaker-aware overlapping chunking with metadata
  embeddings.py         FAISS index builder with LRU transcript cache
  retrieval.py          Full RAG pipeline: temporal → semantic → speaker → rerank → generate
  reranker.py           Cross-encoder reranking wrapper
  prompt.py             Intent-aware prompt builder
utils/
  speaker.py            Speaker extraction and fuzzy matching
  temporal.py           Temporal intent detection and time-window parsing
eval/
  runner.py             Per-question async evaluator using RAGAS
  evaluate.py           Standalone evaluation script
  test_cases.py         Test fixtures for factual, speaker, temporal, and synthesis queries
static/
  test1.html            Main chat UI
  eval.html             Evaluation dashboard
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- A CUDA-capable GPU (recommended) or CPU-only setup with `faiss-cpu`
- The spaCy model: `python -m spacy download en_core_web_md`

## Installation

```bash
git clone https://github.com/Mounika1199/AI-meeting-copilot.git
cd AI_meeting_copilot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_md
```

> **GPU note:** `requirements.txt` installs `faiss-cpu`. For GPU-accelerated FAISS, replace it with `faiss-gpu` after installing the matching CUDA toolkit.

## Configuration

```bash
cp .env.example .env
```

Edit `.env` to match your setup — at minimum set `LLM_MODEL` to a model you have pulled in Ollama:

```bash
ollama pull gemma3:12b-20k   # or any other model
```

Key settings:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `LLM_MODEL` | `gemma3:12b-20k` | Model name as shown by `ollama list` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence transformer for embeddings |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU indices to expose, or leave empty for CPU |
| `SERVER_PORT` | `8018` | Port the FastAPI server listens on |

## Running

```bash
python app.py
```

Then open `http://localhost:8018/static/test1.html` in your browser.

For the evaluation dashboard: `http://localhost:8018/static/eval.html`

## Transcript Format

Paste transcripts in this format:

```
Speaker Name  HH:MM
Their spoken text here.

Another Speaker  HH:MM
Their response.
```

Speakers may optionally include an affiliation separated by `|`:

```
Jane Smith | Acme Corp  14:05
Let's review the roadmap.
```

## Evaluation

Run the standalone evaluation script against the built-in test cases:

```bash
python -m eval.evaluate
```

Or use the in-browser evaluation dashboard to test your own questions interactively.

## License

MIT
