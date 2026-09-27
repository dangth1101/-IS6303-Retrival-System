"""Evaluation run: score retrieval configs on the Query set under every loaded Chunking strategy.

    uv run scripts/evaluate.py                                    # every config, every loaded strategy
    uv run scripts/evaluate.py --configs dense --strategies fixed --limit 20

Prints a markdown table and writes a run folder under eval/runs/<timestamp>/:
settings.json, metrics.json, metrics.csv, per_query.csv (ranks and latencies), table.md.
Runs in the project env (not a PEP 723 script) because it calls the API's retrieval code.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # `uv run scripts/evaluate.py` puts scripts/ on the path, not the repo root

import httpx  # noqa: E402
import psycopg  # noqa: E402

import metrics  # noqa: E402
from api import config, retrieval, strategies  # noqa: E402
from api.embed import embed_query  # noqa: E402
from api.filters import SearchParams  # noqa: E402

QUERY_SET = ROOT / "eval" / "queries.jsonl"
RUNS = ROOT / "eval" / "runs"
POOL = 100  # Chunks Sparse and Dense pull before deduping to the top metrics.DEPTH Recipes
STAGES = ["embed", "sparse", "dense"]

Embedder = Callable[[str], str]  # query text -> pgvector literal


class Stopwatch(dict):
    """Stage name -> milliseconds, on a monotonic clock."""

    @contextmanager
    def __call__(self, stage: str):
        started = time.perf_counter()
        yield
        self[stage] = (time.perf_counter() - started) * 1000


# A config takes the query's params and embedding, times its own stages, and returns ranked Chunks.

def sparse(conn, params, qvec, watch):
    with watch("sparse"):
        return retrieval.sparse(conn, params, POOL)


def dense(conn, params, qvec, watch):
    with watch("dense"):
        return retrieval.dense(conn, params, qvec, POOL)


CONFIGS = {"sparse": sparse, "dense": dense}
USES_EMBEDDING = {"dense"}  # their latency includes the query embedding


def read_queries(path: Path, limit: int | None = None) -> list[dict]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return lines[:limit] if limit else lines


def search_all(conn, queries: list[dict], strategy_names: list[str], configs: list[str], embedder: Embedder):
    """Ranked Recipe ids and stage timings per (config, strategy, query id).

    The first query runs once more as an uncounted warm-up, so caches and connections are hot.
    Each query is embedded once and shared by every strategy and config that needs it.
    """
    ranked, timings = {}, {}
    needs_embedding = any(c in USES_EMBEDDING for c in configs)
    for i, q in enumerate([queries[0], *queries]):
        query_watch, qvec = Stopwatch(), None
        if needs_embedding:
            with query_watch("embed"):
                qvec = embedder(q["text"])
        for strategy in strategy_names:
            params = SearchParams(q=q["text"], strategy=strategy)
            for c in configs:
                watch = Stopwatch()
                hits = CONFIGS[c](conn, params, qvec, watch)
                if i == 0:
                    continue
                if c in USES_EMBEDDING:
                    watch["embed"] = query_watch["embed"]
                key = (c, strategy, q["query_id"])
                ranked[key] = metrics.dedupe(h.recipe_id for h in hits)
                timings[key] = watch
    return ranked, timings


def resolve_strategies(conn, chosen: list[str] | None) -> list[str]:
    """The chosen strategies if all are loaded, else every loaded one, as the API lists them."""
    if not chosen:
        return [s["name"] for s in strategies.loaded(conn)]
    registry = strategies.registry(conn)
    problems = [p for name in chosen if (p := strategies.strategy_problem(name, registry))]
    if problems:
        raise ValueError("; ".join(problems))
    return chosen


def summarize(relevant, ranked, timings) -> list[dict]:
    rows = []
    for (c, s), scores in metrics.metrics_table(relevant, ranked).items():
        watches = [w for (wc, ws, _), w in timings.items() if (wc, ws) == (c, s)]
        latency = {"total": metrics.latency_summary(sum(w.values()) for w in watches)}
        for stage in STAGES:
            if stage in watches[0]:
                latency[stage] = metrics.latency_summary(w[stage] for w in watches)
        rows.append({"config": c, "strategy": s, **scores, "latency_ms": latency})
    return rows


def markdown(rows: list[dict]) -> str:
    head = ["Config", "Strategy", "R@5", "R@10", "R@20", "MRR", "nDCG@5", "nDCG@10", "p50 ms", "p95 ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * 2 + "---:|" * (len(head) - 2)]
    for r in rows:
        total = r["latency_ms"]["total"]
        cells = [r["config"], r["strategy"], *(f"{r[m]:.3f}" for m in metrics.COLUMNS),
                 f"{total['p50_ms']:.1f}", f"{total['p95_ms']:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    n = rows[0]["queries"] if rows else 0
    return f"{n} queries. Ranks over the top {metrics.DEPTH} unique Recipes.\n\n" + "\n".join(lines) + "\n"


def _git_commit() -> str | None:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or None


def write_run(run_dir: Path, settings: dict, rows: list[dict], relevant, ranked, timings) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")

    stage_cols = [f"{st}_{p}" for st in ["total", *STAGES] for p in ("p50_ms", "p95_ms")]
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "strategy", "queries", *metrics.COLUMNS, *stage_cols])
        for r in rows:
            lat = r["latency_ms"]
            w.writerow([r["config"], r["strategy"], r["queries"], *(round(r[m], 4) for m in metrics.COLUMNS),
                        *(round(lat[st][p], 2) if st in lat else "" for st in ["total", *STAGES]
                          for p in ("p50_ms", "p95_ms"))])

    with open(run_dir / "per_query.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "strategy", "query_id", "recipe_id", "rank", "total_ms",
                    *(f"{st}_ms" for st in STAGES), "top_recipe_ids"])
        for (c, s, qid), top in ranked.items():
            watch = timings[c, s, qid]
            w.writerow([c, s, qid, relevant[qid], metrics.rank_of(relevant[qid], top) or "",
                        round(sum(watch.values()), 2),
                        *(round(watch[st], 2) if st in watch else "" for st in STAGES),
                        " ".join(map(str, top))])

    (run_dir / "table.md").write_text(markdown(rows))


def run(conn, embedder: Embedder, *, configs: list[str] | None = None, strategy_names: list[str] | None = None,
        query_set: Path = QUERY_SET, limit: int | None = None, out: Path = RUNS) -> Path:
    """One Evaluation run. Returns its folder."""
    configs = configs or list(CONFIGS)
    strategy_names = resolve_strategies(conn, strategy_names)
    if not strategy_names:
        raise ValueError("no Chunking strategy is loaded")
    queries = read_queries(query_set, limit)
    if not queries:
        raise ValueError(f"{query_set} has no queries")
    relevant = {q["query_id"]: q["recipe_id"] for q in queries}

    started = datetime.now()
    ranked, timings = search_all(conn, queries, strategy_names, configs, embedder)
    rows = summarize(relevant, ranked, timings)

    settings = {
        "started_at": started.isoformat(timespec="seconds"),
        "seconds": round((datetime.now() - started).total_seconds(), 1),
        "git_commit": _git_commit(),
        "configs": configs,
        "strategies": strategy_names,
        "query_set": str(query_set.relative_to(ROOT) if query_set.is_relative_to(ROOT) else query_set),
        "query_set_sha256": hashlib.sha256(query_set.read_bytes()).hexdigest(),
        "queries": len(queries),
        "limit": limit,
        "chunk_pool": POOL,
        "depth": metrics.DEPTH,
        "embed_model": config.EMBED_MODEL,
        "warmup_queries": 1,
    }
    run_dir = out / started.strftime("%Y%m%d-%H%M%S")
    write_run(run_dir, settings, rows, relevant, ranked, timings)
    return run_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS), help="default: all")
    p.add_argument("--strategies", nargs="+", help="default: every loaded Chunking strategy")
    p.add_argument("--queries", type=Path, default=QUERY_SET, help="Query set (JSONL)")
    p.add_argument("--limit", type=int, help="only the first N queries, for quick runs")
    p.add_argument("--out", type=Path, default=RUNS, help="parent folder for the run folder")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn, httpx.Client(timeout=30) as http:
        try:
            run_dir = run(conn, lambda text: embed_query(http, text), configs=a.configs,
                          strategy_names=a.strategies, query_set=a.queries.resolve(), limit=a.limit, out=a.out)
        except ValueError as e:
            sys.exit(f"evaluate: {e}")
    print((run_dir / "table.md").read_text())
    print(f"Run folder: {run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}")
