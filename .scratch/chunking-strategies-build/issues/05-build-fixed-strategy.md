# 05: Build the `fixed` strategy

**What to build:** `fixed` (56-token windows counted with nomic-embed-text's WordPiece tokenizer, cut only between words, ignoring sentence ends, no overlap) is added to the strategy definitions and built with the same command as `sentence`. It becomes a loaded, searchable strategy, and building it doesn't move the other strategies' scores.

Spec: [../../chunking-strategies/spec.md](../../chunking-strategies/spec.md), "Strategies in code". Decisions: settings ticket 03.

**Blocked by:** 04 (Build the `sentence` strategy with the new ingest)

**Status:** ready-for-agent

- [ ] The cut has tests: no mid-word cuts, windows of at most 56 tokens, the title prefix not counted, and a Recipe's last window can be shorter
- [ ] `chunk fixed` finishes with `loaded_at` set, and its Step chunk count is close to 100,536
- [ ] Isolation: semantic's and sentence's top-10 ids and scores are identical before and after the build
- [ ] The registry row records `{"tokens": 56, "tokenizer": "bert-base-uncased"}`
