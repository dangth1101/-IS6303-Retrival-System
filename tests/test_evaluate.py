import csv
import json

import pytest

import evaluate
import ingest
from api import config
from conftest import add_strategy

QUERIES = [
    {"query_id": "q001", "text": "garlic shrimp", "recipe_id": 1, "recipe_title": "Garlic Shrimp"},
    {"query_id": "q002", "text": "steeped tea", "recipe_id": 3, "recipe_title": "Tea"},
]


@pytest.fixture
def loaded_db(recipes, fake_embed):
    """RECIPES chunked under `sentence`; `fixed` registered but not loaded, so never in a default run."""
    ingest.run(recipes, ["chunk", "sentence"])
    add_strategy(recipes, "fixed", loaded=False)
    return recipes


@pytest.fixture
def query_set(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text("".join(json.dumps(q) + "\n" for q in QUERIES))
    return path


def summary_embedding(conn, recipe_id):
    """A fake embedder's answer: the stored vector of the Recipe's Summary chunk."""
    return conn.execute("SELECT embedding::text FROM chunk WHERE recipe_id = %s AND kind = 'summary'",
                        [recipe_id]).fetchone()[0]


def toast_first(query, texts):
    """A fake reranker scorer: Toast's Chunks beat everything, so Reranking visibly reorders."""
    return [1.0 if t.startswith("Toast") else 0.0 for t in texts]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_an_evaluation_run_writes_its_folder_and_ranks_the_known_item_first(loaded_db, query_set, tmp_path):
    calls, scored = [], []

    def embedder(text):
        calls.append(text)
        return summary_embedding(loaded_db, 1)  # every query "means" Garlic Shrimp to Dense

    def scorer(query, texts):
        scored.append(query)
        return toast_first(query, texts)

    run_dir = evaluate.run(loaded_db, embedder, scorer, query_set=query_set, out=tmp_path / "runs")

    assert {p.name for p in run_dir.iterdir()} == {
        "settings.json", "metrics.json", "metrics.csv", "per_query.csv", "table.md"}
    assert calls == ["garlic shrimp", "garlic shrimp", "steeped tea"]  # warm-up, then one per query
    assert scored == calls  # Hybrid reranks once per query, the warm-up included

    settings = json.loads((run_dir / "settings.json").read_text())
    assert settings["strategies"] == ["sentence"]
    assert settings["configs"] == ["sparse", "dense", "fusion", "hybrid"]
    assert settings["queries"] == 2 and len(settings["query_set_sha256"]) == 64
    assert settings["hybrid_candidates"] == config.HYBRID_CANDIDATES
    assert settings["rerank_top"] == config.RERANK_TOP and settings["rerank_model"] == config.RERANK_MODEL
    assert settings["pg_search_version"] and settings["vector_cluster_max_probe"] == 0.02

    ranks = {(r["config"], r["query_id"]): r["rank"] for r in read_csv(run_dir / "per_query.csv")}
    assert set(ranks) == {(c, q) for c in ("sparse", "dense", "fusion", "hybrid") for q in ("q001", "q002")}
    assert ranks["sparse", "q001"] == ranks["dense", "q001"] == ranks["sparse", "q002"] == "1"
    assert ranks["fusion", "q001"] == "1" and ranks["hybrid", "q001"] == "2"  # Reranking put Toast on top

    rows = {r["config"]: r for r in json.loads((run_dir / "metrics.json").read_text())}
    assert rows["sparse"]["mrr"] == 1.0 and rows["sparse"]["queries"] == 2
    assert set(rows["sparse"]["latency_ms"]) == {"total", "sparse"}
    assert set(rows["dense"]["latency_ms"]) == {"total", "embed", "dense"}
    assert set(rows["fusion"]["latency_ms"]) == {"total", "embed", "sparse", "dense", "rrf"}
    assert set(rows["hybrid"]["latency_ms"]) == {"total", "embed", "sparse", "dense", "rrf", "rerank"}
    assert rows["hybrid"]["short_lists"] == 2  # 3 Recipes in the DB: every list is short of 20
    table = (run_dir / "table.md").read_text()
    assert "| sparse | sentence | 1.000 |" in table and "| hybrid | sentence |" in table


def test_fusion_ranks_the_same_candidates_that_hybrid_reranks(loaded_db, query_set, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RERANK_TOP", 1)  # the served cut before Reranking
    run_dir = evaluate.run(loaded_db, lambda text: summary_embedding(loaded_db, 1), toast_first,
                           configs=["fusion", "hybrid"], query_set=query_set, out=tmp_path)
    tops = {(r["config"], r["query_id"]): r["top_recipe_ids"] for r in read_csv(run_dir / "per_query.csv")}
    assert tops["fusion", "q001"] == tops["hybrid", "q001"] == "1"  # Toast is outside the cut, so can't win


def test_a_run_with_hybrid_needs_a_reranker_scorer(loaded_db, query_set, tmp_path):
    with pytest.raises(ValueError, match="reranker"):
        evaluate.run(loaded_db, lambda text: summary_embedding(loaded_db, 1), configs=["hybrid"],
                     query_set=query_set, out=tmp_path)


def test_a_run_can_pick_configs_strategies_and_a_query_limit(loaded_db, query_set, tmp_path):
    run_dir = evaluate.run(loaded_db, lambda text: pytest.fail("Sparse needs no embedding"),
                           configs=["sparse"], strategy_names=["sentence"], query_set=query_set, limit=1,
                           out=tmp_path)
    rows = read_csv(run_dir / "per_query.csv")
    assert [(r["config"], r["strategy"], r["query_id"]) for r in rows] == [("sparse", "sentence", "q001")]


def test_an_unloaded_strategy_is_refused_by_name(loaded_db, query_set, tmp_path):
    with pytest.raises(ValueError, match="'fixed' is not loaded yet"):
        evaluate.run(loaded_db, lambda text: "", toast_first, strategy_names=["fixed"], query_set=query_set,
                     out=tmp_path)


def test_the_command_defaults_to_every_config_and_loaded_strategy():
    a = evaluate.parse_args([])
    assert a.configs is None and a.strategies is None and a.limit is None
    assert a.queries == evaluate.QUERY_SET
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--configs", "bm25"])
