import csv
import json
from pathlib import Path

import pytest

import failures

QUERIES = [
    {"query_id": "q001", "text": "garlic butter prawns", "recipe_id": 1, "recipe_title": "Garlic Shrimp",
     "word_overlap": 1.0},
    {"query_id": "q002", "text": "hot leaf drink", "recipe_id": 2, "recipe_title": "Tea", "word_overlap": 0.5},
    {"query_id": "q003", "text": "crispy bread", "recipe_id": 3, "recipe_title": "Toast", "word_overlap": 1.0},
    {"query_id": "q004", "text": "stack of fluffy cakes", "recipe_id": 4, "recipe_title": "Pancakes",
     "word_overlap": 0.25},
]

# config -> strategy -> query id -> rank ("" = not in the top 20)
RANKS = {
    "sparse": {"fixed": {"q001": 3, "q002": "", "q003": 10, "q004": 1},
               "semantic": {"q001": 1, "q002": 1, "q003": 1, "q004": 1}},
    "dense": {"fixed": {"q001": 11, "q002": 4, "q003": "", "q004": 2},
              "semantic": {"q001": 1, "q002": 1, "q003": 1, "q004": 1}},
    "fusion": {"fixed": {"q001": 2, "q002": 5, "q003": "", "q004": 1},
               "semantic": {"q001": 1, "q002": 1, "q003": 1, "q004": 1}},
    "hybrid": {"fixed": {"q001": 4, "q002": 5, "q003": 20, "q004": ""},
               "semantic": {"q001": 1, "q002": 1, "q003": 1, "q004": 1}},
}


@pytest.fixture
def run_dir(tmp_path):
    """A hand-written run folder: per_query.csv and settings.json pointing at a Query set."""
    query_set = tmp_path / "queries.jsonl"
    query_set.write_text("".join(json.dumps(q) + "\n" for q in QUERIES))
    run = tmp_path / "20260928-000000"
    run.mkdir()
    (run / "settings.json").write_text(json.dumps({"query_set": str(query_set)}))
    recipe = {q["query_id"]: q["recipe_id"] for q in QUERIES}
    with open(run / "per_query.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "strategy", "query_id", "recipe_id", "rank", "total_ms"])
        for c, by_strategy in RANKS.items():
            for s, by_query in by_strategy.items():
                for q, rank in by_query.items():
                    w.writerow([c, s, q, recipe[q], rank, 1.0])
    return run


def ids(cases):
    return [case["query_id"] for case in cases]


def test_sparse_and_dense_wins_split_at_rank_10(run_dir):
    report = failures.analyze_run(run_dir)
    fixed = report["buckets"]["fixed"]
    assert ids(fixed["sparse_win"]) == ["q001", "q003"]  # 3 vs 11, and 10 vs a miss: 10 still counts as found
    assert ids(fixed["dense_win"]) == ["q002"]  # a Sparse miss vs 4
    # q004: both found, so neither bucket


def test_rerank_hurt_is_hybrid_ranking_worse_than_the_fusion_baseline(run_dir):
    fixed = failures.analyze_run(run_dir)["buckets"]["fixed"]
    # q001: 2 -> 4 hurt. q002: 5 -> 5 tie, not hurt. q003: a Fusion miss -> 20, better. q004: 1 -> miss, hurt.
    assert ids(fixed["rerank_hurt"]) == ["q001", "q004"]


def test_counts_are_per_chunking_strategy(run_dir):
    counts = failures.analyze_run(run_dir)["counts"]
    assert counts == {
        "fixed": {"queries": 4, "sparse_win": 2, "dense_win": 1, "rerank_hurt": 2},
        "semantic": {"queries": 4, "sparse_win": 0, "dense_win": 0, "rerank_hurt": 0},
    }


def test_a_case_carries_the_query_the_recipe_title_and_every_rank(run_dir):
    case = failures.analyze_run(run_dir)["buckets"]["fixed"]["dense_win"][0]
    assert case["text"] == "hot leaf drink" and case["recipe_title"] == "Tea"
    assert case["ranks"] == {"sparse": None, "dense": 4, "fusion": 5, "hybrid": 5}


def test_metrics_are_split_by_low_and_high_word_overlap(run_dir):
    split = failures.analyze_run(run_dir)["overlap"]
    high, low = split["sparse", "fixed", "high"], split["sparse", "fixed", "low"]
    assert high["queries"] == 2 and low["queries"] == 2  # overlap 1.0 is high, 0.5 and 0.25 are low
    assert high["recall@10"] == 1.0 and high["mrr"] == pytest.approx((1 / 3 + 1 / 10) / 2)
    assert low["recall@10"] == 0.5 and low["mrr"] == pytest.approx((0 + 1) / 2)
    assert split["dense", "fixed", "high"]["recall@20"] == 0.5  # 11 counts at R@20, the miss doesn't


def test_the_markdown_lists_the_first_n_cases_by_query_id(run_dir):
    md = failures.markdown(failures.analyze_run(run_dir), cases=1)
    assert "| fixed | 4 | 2 | 1 | 2 |" in md
    assert "garlic butter prawns" in md and "Garlic Shrimp" in md
    assert "crispy bread" not in md  # the 2nd Sparse win, past the first N
    assert "1 more, not listed" in md


def test_the_command_writes_failures_md_into_the_run_folder(run_dir):
    out = failures.run(run_dir, cases=5)
    assert out == run_dir / "failures.md" and "stack of fluffy cakes" in out.read_text()


def test_a_run_without_fusion_and_hybrid_has_no_rerank_bucket(run_dir):
    path = run_dir / "per_query.csv"
    rows = [r for r in csv.DictReader(path.open()) if r["config"] in ("sparse", "dense")]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = failures.analyze_run(run_dir)
    assert "rerank_hurt" not in report["counts"]["fixed"]
    assert report["counts"]["fixed"]["sparse_win"] == 2


def test_a_query_set_changed_since_the_run_is_refused(run_dir):
    settings = json.loads((run_dir / "settings.json").read_text())
    (run_dir / "settings.json").write_text(json.dumps({**settings, "query_set_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="changed since the run"):
        failures.analyze_run(run_dir)
    assert failures.analyze_run(run_dir, query_set=Path(settings["query_set"]))["counts"]  # an explicit one is trusted


def test_only_the_queries_the_run_holds_are_analyzed(run_dir):
    """A `--limit` run holds fewer queries than its Query set; the rest aren't misses."""
    path = run_dir / "per_query.csv"
    rows = [r for r in csv.DictReader(path.open()) if r["query_id"] in ("q001", "q002")]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = failures.analyze_run(run_dir)
    assert report["counts"]["fixed"] == {"queries": 2, "sparse_win": 1, "dense_win": 1, "rerank_hurt": 1}
    assert report["overlap"]["sparse", "fixed", "high"]["queries"] == 1  # q001 only, not q003
    assert report["overlap"]["sparse", "fixed", "high"]["mrr"] == pytest.approx(1 / 3)
