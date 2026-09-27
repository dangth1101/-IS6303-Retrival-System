# 05: README and reproducible setup

**What to build:** A grader can go from a fresh clone to the same eval table by following a README: start the DB, pull the Ollama models, ingest, build the Chunking strategies, generate or reuse the Query set, run the eval and the failure analyzer.

Spec: [../spec.md](../spec.md), user stories 29–30.

**Blocked by:** None (can start immediately). Update the eval steps once 02–04 land.

**Status:** ready-for-agent

- [ ] The README covers prerequisites (Docker, uv, Node, Ollama) and every Ollama model needed (`nomic-embed-text`, `qwen2.5:7b`)
- [ ] Steps: DB up, `ingest.py load`, `chunk <name>` per strategy (with rough timings), API + UI, eval, analyzer
- [ ] It says the Query set is committed, so regenerating it is optional
- [ ] Every command in it is run once from a clean shell and works
