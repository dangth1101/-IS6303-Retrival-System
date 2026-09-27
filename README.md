# Recipe Retrieval

Sparse (BM25), Dense (embedding) and Hybrid (RRF + Reranking) retrieval over the [Shengtao/recipe](https://huggingface.co/datasets/Shengtao/recipe) dataset, stored in ParadeDB under three Chunking strategies (`fixed`, `sentence`, `semantic`). An evaluation scores four configs on a fixed Query set with Recall@k, MRR, nDCG@k and latency.

Terms like Chunk, Chunking strategy and Fusion baseline are defined in [CONTEXT.md](CONTEXT.md). Design decisions are in [docs/adr/](docs/adr/).

## Prerequisites

- Docker (runs ParadeDB on port 5434)
- [uv](https://docs.astral.sh/uv/) (Python 3.11+ and all Python dependencies)
- Node 20.19+ or 22.12+ with npm (only for the UI)
- [Ollama](https://ollama.com) running on `http://localhost:11434`

Ollama models:

| Model | Used by | Needed for |
|---|---|---|
| `nomic-embed-text` | ingest, API, eval | embedding Chunks and queries |
| `qwen2.5:7b` | `make_queries.py` | only if you regenerate the Query set |

```sh
ollama pull nomic-embed-text
ollama pull qwen2.5:7b   # optional, see step 4
```

Two Hugging Face models download on first use, no account needed: `bert-base-uncased` (tokenizer for the `fixed` strategy) and `BAAI/bge-reranker-base` (~1 GB, the reranker used by Hybrid).

Every setting has a default that matches `docker-compose.yml`. Override with env vars if yours differ: `DATABASE_URL`, `OLLAMA_URL`, `EMBED_MODEL`, `RERANK_MODEL`, `HYBRID_CANDIDATES`, `RERANK_TOP` (all in `api/config.py`) and `QUERY_MODEL` (in `scripts/make_queries.py`).

## 1. Start the database

```sh
docker compose up -d --wait
uv sync
```

A fresh volume builds the schema from `sql/schema.sql` on first start. `uv sync` creates `.venv` with the API and dev dependencies.

## 2. Load the Recipes

```sh
uv run scripts/ingest.py load
```

Downloads `recipe.csv` (~61 MB) into `data/` if it isn't there, then loads 32,719 Recipes with their ingredients and nutrition. Takes under a minute. It refuses to run once any Chunking strategy exists, because a reload renumbers Recipes.

## 3. Build the Chunking strategies

```sh
uv run scripts/ingest.py chunk semantic
uv run scripts/ingest.py chunk fixed
uv run scripts/ingest.py chunk sentence
```

Each one chunks and embeds every Recipe through Ollama, builds the search index and marks the strategy loaded. Rough timings on an M-series MacBook: `semantic` ~70 min, `fixed` ~60 min, `sentence` ~85 min. Most of that is embedding.

- Transient Ollama errors are retried. If a run crashes, run the same command again: it discards the unfinished build and starts over.
- A strategy isn't searchable until its run prints `<name> loaded`.
- To rebuild one: `uv run scripts/ingest.py drop <name> --yes`, then `chunk <name>` again.

You can evaluate with fewer than three. The eval uses whatever is loaded.

## 4. Query set (already committed)

`eval/queries.jsonl` is in the repo: 298 queries, one per sampled Recipe. Use it as is to get the same numbers. Skip this step.

How it was made: `qwen2.5:7b` wrote one query per Recipe (2 of 300 sampled Recipes were skipped after 3 bad tries). Then all 298 were read against their Recipes and 57 were rewritten by hand: wrong facts, broken output, too vague, or copying the title.

To regenerate it anyway (a few seconds per Recipe on a laptop):

```sh
uv run scripts/make_queries.py --out eval/my-queries.jsonl
```

A regenerated set won't match the committed one: it lacks the hand rewrites, and LLM output can differ across machines and Ollama versions. The sample of Recipes is the same, since it uses a fixed seed (603). A rerun skips Recipes already in the output file, so a crash resumes. Pass the new file to the eval with `--queries eval/my-queries.jsonl`.

## 5. Run the API and UI

```sh
cd ui && npm install && npm run build && cd ..
uv run uvicorn api.main:app
```

Open http://localhost:8000. The page runs one query through Sparse, Hybrid and Dense side by side, with a strategy picker and filters in Settings. API docs are at http://localhost:8000/docs, and `/health` reports the DB, Ollama and reranker status.

For UI development with hot reload, see [ui/README.md](ui/README.md).

The API isn't needed for the eval, which calls the retrieval code directly.

## 6. Run the evaluation

```sh
uv run scripts/evaluate.py
```

Runs every query through four configs under every loaded Chunking strategy:

- `sparse`: Sparse retrieval only (BM25)
- `dense`: Dense retrieval only
- `fusion`: the Fusion baseline, Sparse and Dense fused with RRF, no Reranking (eval only)
- `hybrid`: RRF then Reranking, the same code and settings the API serves

A full run (298 queries × 4 configs × 3 strategies) took about 11 minutes. Reranking is most of it.

Each Evaluation run prints a markdown table and writes `eval/runs/<timestamp>/` with `settings.json`, `metrics.json`, `metrics.csv`, `per_query.csv` (ranks and latencies per query) and `table.md`.

For a quick check:

```sh
uv run scripts/evaluate.py --configs sparse dense --strategies fixed --limit 20
```

## 7. Run the failure analyzer

```sh
uv run scripts/failures.py eval/runs/<timestamp>
```

Reads the run folder, doesn't search again. It prints and writes `failures.md` with:

- counts per strategy of `sparse_win` (Sparse top 10, Dense not), `dense_win` (the reverse) and `rerank_hurt` (Hybrid ranks the Recipe worse than the Fusion baseline)
- the first N cases of each bucket with query, Recipe title and ranks (`--cases N`, default 10)
- each config's metrics split into high and low word overlap between query and Recipe

Run folders are gitignored. Keep one for the report with `git add -f eval/runs/<timestamp>`.

## Tests

```sh
uv run pytest
```

Tests that need the database build a throwaway `recipe_test` DB inside the `recipe-paradedb` container and are skipped when it isn't running. They use fake embedders and scorers, so Ollama isn't needed.

## Layout

```
api/            FastAPI app: retrieval, filters, reranking, config
ui/             React + Tailwind search page, built into api/static
scripts/        ingest, search, make_queries, evaluate, failures, metrics
sql/            schema, BM25 + vector indexes, one-off migrations
eval/           the Query set; runs/ holds Evaluation runs (gitignored)
docs/adr/       architecture decisions
tests/          pytest
```
