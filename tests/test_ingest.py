import threading

import numpy as np
import psycopg
import pytest

import ingest
from api import retrieval
from api.filters import SearchParams


# --- the sentence cut ---------------------------------------------------------

def test_sentence_cut_groups_three_sentences_and_keeps_the_leftover_as_its_own_chunk():
    s = ["Heat oil.", "Add garlic.", "Stir.", "Add shrimp.", "Cook 3 minutes."]
    assert ingest.sentence_cut([s], sentences=3) == [["Heat oil. Add garlic. Stir.", "Add shrimp. Cook 3 minutes."]]


def test_sentence_cut_has_no_size_cap_and_handles_short_recipes():
    long = "Whisk " + "and whisk " * 100 + "well."
    assert ingest.sentence_cut([[long, "Bake.", "Serve."], ["Serve."], []], sentences=3) == [
        [f"{long} Bake. Serve."], ["Serve."], []]


# --- the fixed cut -------------------------------------------------------------

def tokens(text):
    return len(ingest.wordpiece("bert-base-uncased").encode(text, add_special_tokens=False).ids)


LONG = ("Preheat the oven to 350 degrees F. Grease a 9x13 inch baking dish. In a large bowl, whisk "
        "together the flour, sugar, baking powder and salt. Stir in the milk, melted butter and eggs "
        "until just combined. Pour into the prepared dish and bake 25 to 30 minutes, until golden.")


def test_fixed_cut_makes_windows_of_at_most_56_tokens_and_a_shorter_last_one():
    [windows] = ingest.fixed_cut([ingest.split_sentences(LONG)], tokens=56, tokenizer="bert-base-uncased")
    sizes = [tokens(w) for w in windows]
    assert len(windows) > 1 and max(sizes) <= 56
    # each window is full: the next word would have pushed it past 56
    for w, nxt in zip(windows, windows[1:]):
        assert tokens(f"{w} {nxt.split()[0]}") > 56
    assert sizes[-1] < 56


def test_fixed_cut_only_cuts_between_words_and_ignores_sentence_ends():
    [windows] = ingest.fixed_cut([ingest.split_sentences(LONG)], tokens=56, tokenizer="bert-base-uncased")
    assert " ".join(windows) == LONG
    assert [w.split() for w in windows] == [w.split(" ") for w in windows]  # no split word, no stray space
    assert not windows[0].endswith(".")  # the first cut lands mid-sentence


def test_fixed_cut_keeps_short_recipes_whole_and_handles_empty_ones():
    assert ingest.fixed_cut([["Heat oil.", "Serve."], []], tokens=56, tokenizer="bert-base-uncased") == [
        ["Heat oil. Serve."], []]


# --- against the test DB: a few Recipes, embeddings stubbed --------------------

RECIPES = [  # title, directions, ingredient lines
    ("Garlic Shrimp", "Heat oil. Add garlic. Stir. Add shrimp. Cook 3 minutes.", ["1 lb shrimp", "4 cloves garlic"]),
    ("Toast", "Toast the bread.", ["2 slices bread"]),
    ("Tea", "Boil water. Steep ½ hour. Pour.", ["1 tea bag"]),
]


@pytest.fixture
def recipes(db):
    db.execute("TRUNCATE category, author, recipe, recipe_nutrition, recipe_ingredient RESTART IDENTITY CASCADE")
    db.execute("INSERT INTO category (name) VALUES ('Main')")
    for i, (title, directions, lines) in enumerate(RECIPES):
        rid = db.execute(
            """INSERT INTO recipe (url, title, description, directions, category_id, rating,
                 rating_count, review_count, total_minutes, servings)
               VALUES (%s, %s, 'Nice.', %s, 1, 4.5, 10, 3, 20, 2) RETURNING id""",
            [f"u{i}", title, directions]).fetchone()[0]
        db.execute("INSERT INTO recipe_nutrition (recipe_id, calories) VALUES (%s, 100)", [rid])
        for pos, line in enumerate(lines, 1):
            db.execute("INSERT INTO recipe_ingredient VALUES (%s, %s, %s)", [rid, pos, line])
    return db


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    def embed(texts):
        v = np.random.default_rng(len(texts)).random((len(texts), 768), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    monkeypatch.setattr(ingest, "embed", embed)


@pytest.fixture
def staged(recipes):
    """`sentence` registered and loaded into staging, not yet checked or attached."""
    ingest.register(recipes, "sentence")
    ingest.load_chunks(recipes, "sentence")
    return recipes


def test_a_complete_staging_table_passes_every_check(staged):
    assert ingest.staging_problems(staged, "sentence") == []


@pytest.mark.parametrize("corrupt, problem", [
    ("DELETE FROM chunk_sentence_staging WHERE recipe_id = 2 AND kind = 'summary'",
     "every Recipe has exactly 1 Summary chunk"),
    ("""INSERT INTO chunk_sentence_staging (recipe_id, strategy, kind, position, content, embedding,
          category, total_minutes, rating, rating_count, servings, calories)
        SELECT recipe_id, strategy, kind, 1, content, embedding, category, total_minutes, rating,
               rating_count, servings, calories
        FROM chunk_sentence_staging WHERE recipe_id = 1 AND kind = 'ingredients'""",
     "every Recipe has exactly 1 Ingredients chunk"),
    ("DELETE FROM chunk_sentence_staging WHERE recipe_id = 2 AND kind = 'step'",
     "every Recipe has at least 1 Step chunk"),
    ("UPDATE chunk_sentence_staging SET position = 5 WHERE recipe_id = 1 AND kind = 'step' AND position = 2",
     "Step positions run 1..n with no gaps"),
    ("UPDATE chunk_sentence_staging SET calories = 999 WHERE recipe_id = 3 AND kind = 'summary'",
     "ADR-0001 consistency check returns 0"),
])
def test_each_check_names_what_is_wrong(staged, corrupt, problem):
    staged.execute(corrupt)
    assert ingest.staging_problems(staged, "sentence") == [problem]


def test_a_null_embedding_is_caught(staged):
    # staging copies chunk's NOT NULL, so this can only happen if that constraint is lost
    staged.execute("ALTER TABLE chunk_sentence_staging ALTER embedding DROP NOT NULL")
    staged.execute("UPDATE chunk_sentence_staging SET embedding = NULL WHERE recipe_id = 1 AND kind = 'summary'")
    assert ingest.staging_problems(staged, "sentence") == ["no null embeddings"]


# --- the commands --------------------------------------------------------------

loaded = ingest.is_loaded


def tables(conn):
    return {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'chunk\\_%'")}


def step_count(conn, name):
    return conn.execute("SELECT count(*) FROM chunk WHERE strategy = %s AND kind = 'step'", [name]).fetchone()[0]


def test_chunk_builds_a_loaded_searchable_partition(recipes):
    ingest.run(recipes, ["chunk", "sentence"])
    assert loaded(recipes, "sentence") is True
    assert tables(recipes) == {"chunk_sentence"}
    assert step_count(recipes, "sentence") == 4  # 5 sentences -> 2, 1 -> 1, 3 -> 1
    plan = "\n".join(r[0] for r in recipes.execute(
        "EXPLAIN SELECT id FROM chunk WHERE content ||| 'garlic' AND strategy = 'sentence' "
        "ORDER BY pdb.score(id) DESC LIMIT 10"))
    assert "chunk_sentence_search_idx" in plan


def test_building_one_strategy_leaves_the_others_scores_alone(recipes):
    ingest.run(recipes, ["chunk", "semantic"])
    ingest.run(recipes, ["chunk", "sentence"])
    qvec = recipes.execute("SELECT embedding::text FROM chunk ORDER BY id LIMIT 1").fetchone()[0]

    def top10():
        out = []
        for name in ("semantic", "sentence"):
            params = SearchParams(q="garlic shrimp", strategy=name)
            out += [[(c.chunk_id, c.score) for c in retrieval.sparse(recipes, params, 10)],
                    [(c.chunk_id, c.score) for c in retrieval.dense(recipes, params, qvec, 10)]]
        return out

    before = top10()
    assert all(before)
    ingest.run(recipes, ["chunk", "fixed"])
    assert top10() == before


def test_fixed_records_its_parameters_and_leaves_the_title_out_of_the_count(recipes):
    directions = " ".join(["Stir."] * 28)  # 56 tokens: one full window
    assert tokens(directions) == 56
    recipes.execute("UPDATE recipe SET title = 'A Very Long Title For Toast', directions = %s WHERE id = 2",
                    [directions])
    ingest.run(recipes, ["chunk", "fixed"])
    assert loaded(recipes, "fixed") is True
    assert recipes.execute("SELECT method, parameters FROM chunking_strategy WHERE name = 'fixed'").fetchone() == (
        "fixed", {"tokens": 56, "tokenizer": "bert-base-uncased"})
    steps = recipes.execute("SELECT content FROM chunk WHERE strategy = 'fixed' AND kind = 'step' "
                            "AND recipe_id = 2").fetchall()
    assert steps == [(f"A Very Long Title For Toast\n{directions}",)]


def test_a_crashed_build_is_cleaned_up_by_rerunning_it(recipes, monkeypatch):
    monkeypatch.setattr(ingest, "BATCH_SIZE", 1)
    real_embed, calls = ingest.embed, []

    def dies_on_second_batch(texts):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_embed(texts)

    monkeypatch.setattr(ingest, "embed", dies_on_second_batch)
    with pytest.raises(KeyboardInterrupt):
        ingest.run(recipes, ["chunk", "sentence"])
    assert loaded(recipes, "sentence") is False and tables(recipes) == {"chunk_sentence_staging"}

    monkeypatch.setattr(ingest, "embed", real_embed)
    ingest.run(recipes, ["chunk", "sentence"])
    assert loaded(recipes, "sentence") is True
    assert tables(recipes) == {"chunk_sentence"}
    assert step_count(recipes, "sentence") == 4


def test_a_failed_check_leaves_the_strategy_unloaded_and_names_the_check(recipes, monkeypatch):
    monkeypatch.setitem(ingest.STRATEGIES, "sentence",
                        ingest.Strategy("sentence", {}, lambda batch: [[] for _ in batch]))
    with pytest.raises(SystemExit, match="every Recipe has at least 1 Step chunk"):
        ingest.run(recipes, ["chunk", "sentence"])
    assert loaded(recipes, "sentence") is False and tables(recipes) == {"chunk_sentence_staging"}


def test_chunk_refuses_a_loaded_strategy_and_points_to_drop(recipes):
    ingest.run(recipes, ["chunk", "sentence"])
    with pytest.raises(SystemExit, match="sentence is loaded; run `drop sentence` first"):
        ingest.run(recipes, ["chunk", "sentence"])


def test_chunk_refuses_an_unknown_name_and_lists_the_known_ones(recipes):
    with pytest.raises(SystemExit, match="unknown strategy 'foo'; known: fixed, semantic, sentence"):
        ingest.run(recipes, ["chunk", "foo"])


def test_a_second_concurrent_build_fails_fast(recipes, test_db_url, monkeypatch):
    started, release = threading.Event(), threading.Event()
    real_embed = ingest.embed

    def slow_embed(texts):
        started.set()
        release.wait(10)
        return real_embed(texts)

    monkeypatch.setattr(ingest, "embed", slow_embed)
    with psycopg.connect(test_db_url, autocommit=True) as first:
        build = threading.Thread(target=ingest.run, args=(first, ["chunk", "sentence"]))
        build.start()
        try:
            assert started.wait(10)
            with pytest.raises(SystemExit, match="already being built"):
                ingest.run(recipes, ["chunk", "sentence"])
        finally:
            release.set()
            build.join()
    assert loaded(recipes, "sentence") is True


def test_drop_refuses_a_loaded_strategy_without_yes_and_shows_its_size(recipes):
    ingest.run(recipes, ["chunk", "sentence"])
    with pytest.raises(SystemExit, match="sentence is loaded with 10 chunks"):
        ingest.run(recipes, ["drop", "sentence"])
    assert loaded(recipes, "sentence") is True

    ingest.run(recipes, ["drop", "sentence", "--yes"])
    assert loaded(recipes, "sentence") is None and tables(recipes) == set()


def test_drop_removes_an_unfinished_build_without_yes(recipes, monkeypatch):
    monkeypatch.setitem(ingest.STRATEGIES, "sentence",
                        ingest.Strategy("sentence", {}, lambda batch: [[] for _ in batch]))
    with pytest.raises(SystemExit):
        ingest.run(recipes, ["chunk", "sentence"])
    ingest.run(recipes, ["drop", "sentence"])
    assert loaded(recipes, "sentence") is None and tables(recipes) == set()


def test_load_refuses_while_strategies_exist_and_names_them(recipes):
    ingest.run(recipes, ["chunk", "sentence"])
    ingest.run(recipes, ["chunk", "semantic"])
    with pytest.raises(SystemExit, match="drop strategies first: semantic, sentence"):
        ingest.run(recipes, ["load"])
    assert recipes.execute("SELECT count(*) FROM recipe").fetchone()[0] == len(RECIPES)


def test_the_old_index_command_is_gone(recipes):
    with pytest.raises(SystemExit):
        ingest.run(recipes, ["index"])
