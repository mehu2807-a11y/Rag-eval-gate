# RAG-Eval: Self-Evaluating Retrieval-Augmented Generation System

A RAG pipeline that doesn't just answer questions — it grades itself. Every time
the document set, embedding model, chunking strategy, or prompt changes, an
automated evaluation suite (Ragas) scores answer quality and blocks the change
if quality drops below a threshold. Experiments are logged to MLflow so you can
compare configurations over time.

## Architecture

```
                 ┌─────────────────────┐
                 │   data/ (your docs)  │
                 └──────────┬───────────┘
                            │  watched by scripts/reindex_watch.py
                            ▼
   ┌──────────────────────────────────────────────┐
   │  app/ingest.py                                │
   │  chunk → embed (sentence-transformers)        │
   │  → store in Chroma (vector DB)                │
   └──────────────────────┬─────────────────────────┘
                           │
                           ▼
   ┌──────────────────────────────────────────────┐
   │  app/retrieval.py  →  top-k relevant chunks   │
   └──────────────────────┬─────────────────────────┘
                           │
                           ▼
   ┌──────────────────────────────────────────────┐
   │  app/generate.py  →  LLM answer (Claude/GPT)  │
   └──────────────────────┬─────────────────────────┘
                           │
                           ▼
   ┌──────────────────────────────────────────────┐
   │  app/main.py  (FastAPI)  →  POST /ask         │
   └──────────────────────┬─────────────────────────┘
                           │
                ┌──────────┴───────────┐
                ▼                      ▼
   ┌────────────────────┐   ┌────────────────────────┐
   │ eval/run_eval.py    │◄──┤ .github/workflows/eval │
   │ Ragas: faithfulness,│   │ .yml — CI quality gate │
   │ relevancy, context  │   │ blocks merge if score  │
   │ precision/recall    │   │ drops below threshold  │
   └──────────┬──────────┘   └────────────────────────┘
              ▼
   ┌────────────────────┐
   │ tracking/mlflow_log │  logs params + eval metrics
   │ .py                 │  for every experiment run
   └────────────────────┘
```

## Project layout

```
rag-eval-project/
├── app/
│   ├── config.py        # all tunable settings, read from env vars
│   ├── ingest.py         # chunking + embedding + Chroma indexing
│   ├── retrieval.py      # top-k retrieval
│   ├── generate.py       # LLM call (Anthropic or OpenAI)
│   └── main.py           # FastAPI app: /ask, /reindex, /health
├── data/                  # <-- put YOUR documents here (.txt / .md)
├── eval/
│   ├── testset.json       # <-- put YOUR question/ground-truth pairs here
│   └── run_eval.py         # runs the pipeline against testset.json, scores it
├── tracking/
│   └── mlflow_log.py       # wraps run_eval.py, logs to MLflow
├── scripts/
│   └── reindex_watch.py    # watches data/ and auto re-indexes + re-evals
├── .github/workflows/eval.yml   # CI quality gate
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API key(s)
```

### 1. Add your data

Drop `.txt` or `.md` files into `data/`. Nothing is hardcoded — the pipeline
ingests whatever it finds there. See `data/README.md`.

### 2. Build the index

```bash
python -m app.ingest
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Then:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'
```

### 4. Add your eval set

Fill in `eval/testset.json` with real question / ground-truth-answer pairs
based on your own documents (10–50 pairs is a good start). See the file for
the expected format — it ships with the schema only, no fabricated data.

### 5. Run the evaluation suite

```bash
python -m eval.run_eval
```

This prints a JSON score report (faithfulness, answer relevancy, context
precision, context recall) and exits non-zero if faithfulness drops below the
threshold in `app/config.py` — this is what the CI workflow checks.

### 6. Track experiments in MLflow

```bash
python -m tracking.mlflow_log --run-name "chunk_size_300"
mlflow ui   # view at http://localhost:5000
```

Change a config value (chunk size, embedding model, top_k) in `app/config.py`
or via env vars, re-run, and compare runs side by side in the MLflow UI.

### 7. Auto re-index on new documents

```bash
python -m scripts.reindex_watch
```

Watches `data/` for changes; on any add/edit/delete it re-runs ingestion and
then the eval suite, so you always know if new docs hurt retrieval quality.

### 8. CI/CD quality gate

`.github/workflows/eval.yml` runs `eval/run_eval.py` on every push and pull
request. If the faithfulness score falls below the threshold, the workflow
fails — this blocks a bad prompt/model/embedding change from being merged.
You'll need to add your LLM API key as a repo secret (`ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`) under Settings → Secrets → Actions.

### 9. Docker

```bash
docker build -t rag-eval .
docker run -p 8000:8000 --env-file .env rag-eval
```

## What to write up for an interview

- Which chunk sizes you tried and how faithfulness/context precision changed
- Which embedding model you tried and its retrieval recall
- One example where the eval suite caught a hallucination or a bad retrieval
  before you would have noticed it manually
- The CI run history showing at least one blocked/failed run and the fix
