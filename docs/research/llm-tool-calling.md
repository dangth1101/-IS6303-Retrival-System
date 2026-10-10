# Tool calling and structured output on qwen2.5:7b and Gemini via LangGraph

Research for issue #2 (part of #1). Checked 2026-10-10.

Versions on PyPI that day: `langchain` 1.4.4, `langchain-core` 1.6.9, `langgraph` 1.2.14, `langchain-ollama` 1.1.0 (2026-04-07), `langchain-google-genai` 4.4.0 (2026-09-01). Local Ollama is 0.40.1 with `qwen2.5:7b` Q4_K_M.

## Two ways to get structured data out of the model

- **Tool calling**: the model returns a tool name plus JSON arguments. LangChain: `model.bind_tools([...])`, results in `msg.tool_calls`. LangGraph runs them with `ToolNode`. ([LangChain models](https://docs.langchain.com/oss/python/langchain/models))
- **Structured output**: the whole reply is forced into one schema. LangChain: `model.with_structured_output(Schema, method=...)`, where `method` is `json_schema` (provider's native feature), `function_calling` (a forced tool call) or `json_mode` (schema only in the prompt). `include_raw=True` returns `raw`, `parsed`, `parsing_error` instead of raising. ([LangChain models](https://docs.langchain.com/oss/python/langchain/models))

## Ollama + qwen2.5:7b

- Package `langchain-ollama`; class `ChatOllama`; `init_chat_model` provider string `ollama`. ([Ollama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama), [init_chat_model reference](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model))
- Nothing to set beyond a running Ollama with the model pulled. `base_url` defaults to the Ollama client default (localhost:11434). ([chat_models.py](https://github.com/langchain-ai/langchain/blob/master/libs/partners/ollama/langchain_ollama/chat_models.py))
- `validate_model_on_init=True` fails at startup if the model isn't pulled. Default is `False`. (same source)
- qwen2.5 carries the `tools` tag on Ollama. Native context 32K. ([ollama.com/library/qwen2.5](https://ollama.com/library/qwen2.5))
- Qwen2.5 uses Hermes-style tool calls (`<tool_call>{json}</tool_call>`). Ollama's template emits them and Ollama parses them into `tool_calls`. ([Qwen docs, v2.5](https://qwen.readthedocs.io/en/v2.5/framework/function_call.html))
- Since May 2025 Ollama's parser reads each model's tool-call prefix and can stream around tool calls, instead of waiting for the whole reply and trying to parse it as JSON. ([Ollama blog, 2025-05-28](https://ollama.com/blog/streaming-tool))
- **`tool_choice` is ignored.** The `ChatOllama.bind_tools` docstring says it "is currently ignored as it is not supported by Ollama". You can't force a tool call on Ollama. ([chat_models.py](https://github.com/langchain-ai/langchain/blob/master/libs/partners/ollama/langchain_ollama/chat_models.py))
- `with_structured_output` defaults to `method="json_schema"` since `langchain-ollama` 0.3.0. That passes the schema as Ollama's `format`, which constrains output to the schema. ([chat_models.py](https://github.com/langchain-ai/langchain/blob/master/libs/partners/ollama/langchain_ollama/chat_models.py), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [Ollama blog, 2024-12-06](https://ollama.com/blog/structured-outputs))
- Ollama's advice for structured output: temperature 0, and also put the schema in the prompt. (same sources)
- Unparseable tool arguments: `langchain-ollama` tries `json.loads`, then `ast.literal_eval`. If both fail it raises `OutputParserException`; it doesn't return an `invalid_tool_calls` entry. (same source)
- **Context window.** Ollama's default context depends on VRAM: 4k under 24 GiB, 32k at 24–48 GiB. It recommends at least 64k for agents and tools. ([Ollama context length](https://docs.ollama.com/context-length)) The Ollama tool-calling blog also says 32k+ "anecdotally" improves tool calling. ([Ollama blog](https://ollama.com/blog/streaming-tool)) On this 16 GB M4 Mac, `/api/ps` shows `context_length: 4096` for qwen2.5:7b. `ChatOllama(num_ctx=...)` sets it per request.

### Known failure modes for qwen2.5 (from Qwen's own docs)

From [Qwen docs, v2.5](https://qwen.readthedocs.io/en/v2.5/framework/function_call.html):

- "It is not guaranteed that the model generation will always follow the protocol even with proper prompting or templates."
- Extra text around the tool call.
- Arguments as a Python dict repr instead of JSON.
- Valid JSON that doesn't match the schema.
- Calling the same function over and over.
- Inventing required arguments instead of asking the user.
- Their advice: parse defensively, improve prompts, and fine-tune if needed.

### Local smoke test (2026-10-10, this machine)

Ran in a throwaway `uv run --with`, nothing installed into the project. The script is not committed. The tool schema was the real `SearchParams` JSON schema without `strategy`, plus a `select_recipe(position)` tool. Temperature 0 once and 0.8 three times, so 8 prompts × 4 runs = 32 calls. Default 4k context.

Tool calling (`bind_tools`):

- 32/32 calls came back as clean `tool_calls` or plain text. No malformed JSON, and every `search_recipes` argument set passed `SearchParams` validation.
- General question ("how long should I rest a steak"): answered directly, no tool, 4/4.
- Numeric filters were mapped right: `min_rating 4.5`, `max_calories 400`, `min_protein_g 25`, `max_sodium_mg 500` + `max_total_minutes 60`, `k 10`.
- **Wrong tool, 4/4**: "tell me more about the second one", asked after a 3-item result list, called `search_recipes(q="Spicy Sausage Penne", k=1)` instead of `select_recipe(2)`. It picked the right Recipe but the wrong tool.
- Ambiguous field, 4/4: "under 30 minutes" became `max_cook_minutes` instead of `max_total_minutes`.
- Latency: about 1.1–2.7 s per call once warm (8 s cold load). Prompt was about 725–835 tokens with both tools.

Structured output (`with_structured_output(..., method="json_schema")`, 6 search prompts, temperature 0):

- 6/6 parsed, but 3/6 were wrong in meaning:
  - Invented `category` values (`"dinner"`, `"chicken"`, `"desserts"`, `"dishes"`).
  - On "10 chocolate desserts" it filled 14 made-up range bounds and missed `k=10`.
  - "less than 500mg sodium" became `max_sodium_mg: 5e-305`.
- Lesson: constrained decoding guarantees the shape, not the meaning. With a schema of ~25 optional fields, the 7B model fills fields it shouldn't. Tool calling did better than structured output on the same schema.

## Gemini

- Package `langchain-google-genai` (4.x uses the `google-genai` SDK). Class `ChatGoogleGenerativeAI`. Reads `GOOGLE_API_KEY`, then `GEMINI_API_KEY`. ([Gemini integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai))
- **Use the provider prefix.** In `init_chat_model`, a bare `gemini-...` model name is inferred as `google_vertexai`, not `google_genai` (the docs say this "changes in next major"). Write `google_genai:<model>`. ([init_chat_model reference](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model))
- Tool calling is supported. Gemini has modes AUTO (default), ANY (must call a function), NONE and VALIDATED (schema adherence), so `tool_choice="any"` or a tool name works here, unlike Ollama. ([Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling), updated 2026-09-23)
- Google's advice: precise descriptions, specific types and enums, keep 10–20 tools at most, and check calls before running them. (same page)
- Gemini 3 models use "thought signatures" in tool calls, which must be passed back unchanged. The SDK handles them; `langchain-google-genai` handles them from 3.1.0. (same page, [Gemini integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai))
- `with_structured_output` defaults to `method="json_schema"` (Gemini native). `function_calling` is the other option. ([Gemini integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai))
- Native structured output supports a subset of JSON Schema: types incl. null, `properties`, `required`, `additionalProperties`, `enum`, `minimum`/`maximum`, array `items`/`minItems`/`maxItems`. `minLength` isn't listed. Google says: "While output is syntactically correct JSON, always validate values in your application." ([Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output), updated 2026-09-23)
- Structured output combined with function calling in one request is Gemini 3 only and preview. (same page)

### Which model, and the free tier

- Stable text models as of 2026-10-09: `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`. `gemini-3.1-pro-preview` and `gemini-3-flash-preview` are preview. ([Gemini models](https://ai.google.dev/gemini-api/docs/models))
- `gemini-2.0-flash` is shut down. The 2.5 models are "limited to users who have actively used them in the past", so a new key may not get them. (same page)
- Free tier ("Free of charge") exists for 3.8 Flash, 3.6 Flash, 3.5 Flash-Lite, 3.1 Flash-Lite and 3 Flash preview. 3.1 Pro preview has no free tier. ([Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), updated 2026-10-09)
- On the free tier, prompts are used to improve Google's products. Fine for recipe questions, but say so in the report. (same page)
- Limits are per Google Cloud project, not per key. Daily quota resets at midnight Pacific. ([Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), updated 2026-10-09)
- The public page no longer prints free-tier RPM/TPM/RPD numbers. It points to AI Studio (login needed). See Unconfirmed.

## Swapping providers by config

- `init_chat_model("ollama:qwen2.5:7b", temperature=0)` or `init_chat_model("google_genai:gemini-3.8-flash", temperature=0)`. One config key holding the `provider:model` string is enough. ([LangChain models](https://docs.langchain.com/oss/python/langchain/models))
- `init_chat_model` passes extra kwargs to the provider class, so provider-only settings (`num_ctx` for Ollama) need a small per-provider branch.
- Runtime switching is also possible: with `configurable_fields=("model", "model_provider")` you can pass `config={"configurable": {"model": ...}}` per call. Don't use `"any"`: it lets callers change `api_key` and `base_url`. ([init_chat_model reference](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model))
- `init_chat_model` defaults `max_retries` to 6. That matters for Gemini 429s: a quota hit retries for a while before failing. ([LangChain models](https://docs.langchain.com/oss/python/langchain/models))
- Install: `langchain`, `langgraph`, `langchain-ollama`, plus `langchain-google-genai` only if Gemini is used. Env: `GOOGLE_API_KEY` for Gemini; nothing for Ollama beyond its URL (already `OLLAMA_URL` in `api/config.py`).

## LangGraph error handling that helps

- `ToolNode`'s default `handle_tool_errors` catches invalid arguments from the model and returns a descriptive error message so the model can retry. Errors raised inside the tool body are re-raised. ([ToolNode reference](https://reference.langchain.com/python/langgraph.prebuilt/tool_node/ToolNode))
- For `create_agent`, `response_format` picks the provider's native structured output when the model profile says it's supported, else a tool-call strategy. `handle_errors=True` (default) re-prompts on validation errors. ([LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output))

## What this means for the Assistant

Recommendation: use **tool calling** for search (not `with_structured_output`). Give the model a **smaller tool schema** than `SearchParams`, and keep "which Recipe" out of the LLM where you can.

- One config key, e.g. `ASSISTANT_MODEL=ollama:qwen2.5:7b`. Gemini is `google_genai:gemini-3.8-flash`, or a Flash-Lite model if quota is tight. Built with `init_chat_model`, `temperature=0`.
- Set `num_ctx` explicitly for Ollama (e.g. 16k–32k). The 4k default on this Mac will silently squeeze a Selected Recipe's full text plus history.
- The search tool exposes `q`, `k` and a few filters people actually say: `max_total_minutes`, `min_rating`, `max_calories`, `min_protein_g`, `max_sodium_mg`. Leave out `strategy` (server picks) and `category` (the model invents values) unless you give an `enum` of real categories. Build the full `SearchParams` in code and validate it there.
- Make Selected Recipe a code path where possible. A click selects directly, with no LLM. For "the second one", return numbered results with IDs in the tool result and test it. qwen2.5:7b picked the wrong tool 4/4 in the smoke test.
- Don't rely on `tool_choice`: Ollama ignores it. If a turn must search, route in the graph rather than forcing it.
- On bad arguments: `ToolNode` returns the validation error to the model once. If it still fails, fall back to a plain search with just `q`. Catch `OutputParserException` from `ChatOllama` too.
- `/health`: Ollama reachable and model pulled (`validate_model_on_init`). For Gemini, only whether `GOOGLE_API_KEY` is set, since a real call costs quota.
- Put the wrong-tool and wrong-field cases into the ~30 hand-judged Conversations, so the eval measures them.
- Outside the current setup: `qwen3` is what Ollama's tool docs now use as the example model. It may call tools better, but it's untested here and would change the "local default" decision. Worth one comparison run, not a switch.

## Unconfirmed

- Free-tier RPM / TPM / RPD for any current Gemini model. The public rate-limits page has no free-tier numbers; it points to AI Studio, which needs a login. Check at https://aistudio.google.com/rate-limit once a key exists.
- Which Gemini model names are current. LangChain's docs use `gemini-3.7-flash`, which isn't on Google's models page (it lists 3.6 and 3.8 Flash). Trust Google's page.
- Gemini not tested live (no key; `.env` not read). Whether Gemini accepts the full `SearchParams` schema as-is (`minLength`/`maxLength` on `q`, `anyOf` with null) is unconfirmed.
- Smoke test numbers are 8 prompts on one machine, one system prompt. They point the way but aren't a measurement.
- How Ollama enforces `format` internally (grammar vs other). Its docs don't say.
- Whether `num_ctx` passed per request through `ChatOllama` overrides the 4k server default without reloading the model. Expected yes (it's a model option), but not checked.
- The KV-cache cost of a 32k context for qwen2.5:7b is my estimate (about 1.8 GB at fp16 from the model shape), not measured.
