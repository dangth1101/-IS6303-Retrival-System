# 04: Failure analyzer

**What to build:** A command reads an Evaluation run's folder, without searching again, and lists the queries where the methods disagree: Sparse found the Recipe and Dense didn't, the reverse, and where Reranking made the rank worse. The output is ready to use for report case studies.

Spec: [../spec.md](../spec.md), "Implementation Decisions" (Failure analyzer).

**Blocked by:** 03 (Add Hybrid and the Fusion baseline)

**Status:** ready-for-agent

- [ ] Sparse win = Sparse rank ≤ 10 and Dense rank > 10. Dense win is the reverse
- [ ] Rerank hurt = Hybrid rank worse than Fusion baseline rank
- [ ] Counts per bucket per Chunking strategy
- [ ] Markdown case lists with query, Recipe title and ranks, in a stable order (first N by query id)
- [ ] Metrics are also split by low vs high word overlap, to show the synthetic-query bias
- [ ] Tests use a hand-written per-query ranks file and check bucket membership and counts
