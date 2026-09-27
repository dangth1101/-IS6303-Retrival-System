# 03: Pick a Chunking strategy in the UI and API

**What to build:** A user picks the Chunking strategy next to the search box, and all three result columns re-run under it. Every search endpoint requires a `strategy` (no default). Unknown or not-yet-loaded names get a clear 422. The API lists loaded strategies, reports chunk counts per strategy in health, and echoes the strategy in every response. The `CHUNKER` setting is gone. A strategy that finishes loading while the API is running shows up without a restart.

Spec: [../../chunking-strategies/spec.md](../../chunking-strategies/spec.md), "Slice 3". Decisions: API/UI ticket 07.

**Blocked by:** 02 (Semantic search runs on the partitioned `chunk`)

**Status:** ready-for-agent

- [ ] A search with no `strategy`, an unknown one or an unloaded one returns 422 with the messages from the spec; `chunker=` is rejected
- [ ] `GET /strategies` returns only loaded strategies, ordered by name
- [ ] `/health` shows `chunks` per loaded strategy, and the Settings drawer's Services section shows them
- [ ] Responses include `strategy`
- [ ] The picker sits next to the search box, pre-selects `semantic` (or the last choice), re-runs the current query on change, and column headers show the strategy
- [ ] With no loaded strategy, search is disabled with a message
- [ ] Inserting a loaded registry row in a test DB makes it appear in `/strategies` without a restart
- [ ] The built UI bundle served by the API is rebuilt
