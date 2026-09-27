# 04: Build the `sentence` strategy with the new ingest

**What to build:** A developer runs one command to build a Chunking strategy end to end, and `sentence` (3 sentences per Step chunk) becomes a loaded, searchable strategy. Building it doesn't change any other strategy's scores.

The ingest script holds every strategy's definition in one place (name, method, parameters, cut). It builds into a staging table, checks completeness, then indexes and attaches the table as the partition and marks it loaded in one transaction. A crashed build is cleaned up by rerunning it. A loaded strategy is only removed by an explicit drop (which needs confirmation). Reloading the source data is refused while any strategy exists.

`sentence` goes first because it's the simplest cut and needs no tokenizer, so the lifecycle is proven on the easy case.

Spec: [../../chunking-strategies/spec.md](../../chunking-strategies/spec.md), "Slice 2". Decisions: ingest ticket 08, settings ticket 03, registry ticket 05.

**Blocked by:** 02 (Semantic search runs on the partitioned `chunk`)

**Status:** ready-for-agent

- [ ] `chunk sentence` finishes with `loaded_at` set, and its Step chunk count is close to 110,314
- [ ] Isolation: semantic's top-10 BM25 and dense ids and scores, captured before the build, are identical after it
- [ ] `EXPLAIN` for a `sentence` search uses its own partition's index
- [ ] Killing the build mid-load, then rerunning it, cleans up and finishes
- [ ] `chunk sentence` on a loaded `sentence` refuses and points to `drop`
- [ ] `drop <name>` on a loaded strategy refuses without `--yes`
- [ ] `load` refuses while strategies exist, and names them
- [ ] A second concurrent `chunk sentence` fails fast
- [ ] The old `index` command is gone
- [ ] The completeness checks (1 Summary, 1 Ingredients, at least 1 Step per Recipe; gap-free positions; no null embeddings; ADR-0001 check) have tests
