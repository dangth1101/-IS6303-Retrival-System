# 02: First Evaluation run: Sparse and Dense

**What to build:** One eval command scores Sparse and Dense retrieval on the Query set under every loaded Chunking strategy. It prints a markdown table (Recall@5/10/20, MRR, nDCG@5/10, latency p50/p95) and writes a run folder with the same numbers plus per-query ranks. This is the tracer bullet: all the plumbing lands here.

Spec: [../spec.md](../spec.md), "Implementation Decisions" (Four configs, Chunk → Recipe dedupe, Metrics, Latency, Run output, Eval command options).

**Blocked by:** 01 (Generate the Query set)

**Status:** ready-for-agent

- [ ] Metrics module is pure: relevant Recipe per query + ranked Recipe ids per (config, strategy, query) in, metrics table out
- [ ] Sparse and Dense pull 100 Chunks, dedupe to Recipes by first appearance, keep 20
- [ ] Strategies default to every loaded one from the registry. Configs, strategies, query-set path and a query limit can be chosen
- [ ] Latency is timed in process per stage (query embedding, Sparse, Dense), with one warm-up query not counted
- [ ] The run folder holds settings (including the query-set file hash), metrics JSON and CSV, per-query ranks and latencies CSV, and the markdown table
- [ ] Run folders are gitignored
- [ ] Metric tests with hand-worked values: ranks 1 and 7, a missing Recipe, a duplicate Recipe counted once, an empty result counted as a miss
- [ ] An end-to-end test on `recipe_test` with a fake embedder: two queries, the expected files, and a known-item query ranking its Recipe first
- [ ] A real run over the 300 queries and 3 strategies finishes, and the table is pasted into the Done notes
