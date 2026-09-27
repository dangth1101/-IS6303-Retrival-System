# 03: Add Hybrid and the Fusion baseline

**What to build:** The Evaluation run gains Hybrid retrieval (RRF + Reranking, as the API serves it) and the Fusion baseline (RRF without Reranking, eval only), so one table compares all four configs and shows what Reranking adds.

Spec: [../spec.md](../spec.md), "Implementation Decisions" (Four configs, Glossary).

**Blocked by:** 02 (First Evaluation run: Sparse and Dense)

**Status:** ready-for-agent

- [ ] Both configs reuse the existing retrieval functions and the API's candidate and rerank settings, not copies of them
- [ ] The reranker loads once before timing. Latency adds RRF and Reranking stages
- [ ] A config with fewer than 20 unique Recipes counts the missing positions as misses
- [ ] `CONTEXT.md` adds **Fusion baseline**, **Query set** and **Evaluation run**. "Reranking is always part of Hybrid retrieval" still holds for served search
- [ ] No API or UI changes
- [ ] The end-to-end test covers all four configs with a fake reranker scorer
- [ ] A real run of 4 configs × 3 strategies finishes, and the table is pasted into the Done notes
