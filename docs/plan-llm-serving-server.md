# Plan: Proper LLM Serving Server

**Status:** Proposed plan only — no code has been written on this branch yet.
**Branch:** `feat/llm-serving-server` (derived from `main`). Code lands only after this plan is reviewed.
**Source of truth (upstream):** local checkout `/Users/vaijanath.rao/Work/openSource/llama.cpp`
at `b9952` (Java pins `b9941`; version-gated items need the `GIT_TAG` bump per `AGENTS.md`).

This document scopes what it would take to turn the current bindings into a **proper
multi-model, multi-client LLM server** with KV-cache management, utilization
observability, and admin control. It is a gap plan: the repo already ships a
working server layer, so most work is *closing parity gaps* and *surfacing upstream
state into the Java API* — not building a server from zero.

---

## 1. Current state (what already exists)

| Capability | Where | Notes |
|-------------|-------|-------|
| Standalone server | `server/NativeServer.java` | Loads its own model from forwarded args (like `llama-server`). |
| Attached server | `server/NativeServer.java` | Serves an already-loaded `LlamaModel` over HTTP — no second copy of weights. |
| **Router mode (multi-model)** | `server/NativeServer.java` | `--models-dir` + per-model worker subprocess; `GET/POST /models`, per-request model selection. |
| OpenAI-compatible routes | `server/OpenAiCompatServer.java` | chat, completions, embeddings, rerank, messages, responses, infill, **models, metrics, slots, health**. |
| Utilization metrics | `value/ServerMetrics.java` | idle/processing/busy slots, cumulative + window token usage, timings, per-slot metrics. |
| Model-parameter plumbing | `parameters/ModelParameters.java` | `--models-dir/-max/-autoload/-preset`, KV cache (`--cache-ram`, `--cache-idle-slots`), prompt cache (`--prompt-cache`/`-ro`/`-all`), `--parallel`, `--cont-batching`, API-key passthrough. |

**Conclusion:** multi-model (router), KV cache, and basic utilization already work.
What is missing is *route parity*, *deeper utilization data*, *admin control
endpoints*, and *multi-client guardrails*.

---

## 2. Target capabilities & gaps

### 2.1 Route parity (`OpenAiCompatServer` vs upstream `server.cpp`)
Reachable today only via `NativeServer`'s own router; **not** wired into the
OpenAI-compatible frontend:

| Route | Upstream | Complexity |
|-------|----------|------------|
| `POST /v1/audio/transcriptions` | `server.cpp` | Small (proxy) |
| `POST /v1/chat/completions/control` | `server.cpp` (pause/resume/abort/set-params) | Medium (real-time `SERVER_TASK_TYPE_CONTROL`) |
| `GET /v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/v1/messages/count_tokens` | `server.cpp` | Small (chat-template-aware token count) |
| `/v1/stream/:conv_id`, `/v1/streams/lookup`, `DELETE` | `server.cpp` | Medium (stream handle registry) |
| `/apply-template`, `/tokenize`, `/detokenize`, `/lora-adapters` (GET/POST) | `server.cpp` | Small (JNI already exists, just not routed) |

These are direct analogues of the §A1/A2/A4 + §E gaps in
`docs/plan-upstream-feature-gaps.md` — same work, surfaced on the OAI router.

### 2.2 Utilization / observability
`ServerMetrics` currently exposes slot counts and token throughput, but **not**:
- **GPU / VRAM utilization** per device (upstream `/metrics` emits `ggml_device` + KV-cache usage — needs parsing).
- **Prompt-cache hit/miss stats** and KV-cache occupancy.
- **Queue depth** (deferred tasks are exposed; need a queue-length gauge).
- **Per-model breakdown in router mode** (router spawns one worker per model; metrics are per-worker and must be aggregated/keyed by model alias).

Proposed `ServerMetrics` extensions (parse new upstream JSON fields, keep the
existing accessors stable):
- `getDeviceUtilization()` / `getVramUsedBytes()` per device.
- `getKvCacheUsage()` / `getPromptCacheHits()`.
- `getQueueDepth()`.
- `getModelMetrics(alias)` (router mode).

Surfaced both on `/metrics` JSON and via a richer Java `ServerMetrics` API.

### 2.3 Multi-model management / admin
Router mode works, but there is no **runtime model admin** from Java:
- List / load / unload models in the pool (upstream supports `POST /models` load + `GET /models`).
- Per-model **health** (`/health` per worker) and **load** (latency, queue).
- Runtime tuning of a loaded model (sampling defaults, KV budget) — today only at launch via `ModelParameters`.

Proposed: extend `NativeServer` with a small admin facade (`ModelPool`):
- `listModels()`, `loadModel(alias, params)`, `unloadModel(alias)`.
- `getModelHealth(alias)`, `getModelMetrics(alias)`.
All delegate to existing upstream endpoints / args.

### 2.4 KV-cache admin
Covered at launch (`--cache-ram`, `--cache-idle-slots`, `--prompt-cache*`).
Missing **runtime** control:
- `POST /slots` save / `DELETE /slots/:id` erase (upstream supports; not routed in OAI frontend).
- Prompt-cache stats readout (see 2.2).
- Idle-eviction controls already exist (`setClearIdle` / `setCacheRamMib`).

### 2.5 Multi-client guardrails
- **Authentication:** upstream `--api-key` / `--api-key-file` exist and pass through `NativeServer` args today; no Java-side helper to set/rotate them — add convenience on `OpenAiServerConfig`.
- **Rate limiting / per-client quotas:** upstream has limited/no native support; a Java `Filter` on `OpenAiCompatServer` (token-bucket per API key) is the realistic path.
- **Concurrency budgeting:** `--parallel` + `--cont-batching` exist; expose sane defaults + a `maxConcurrentClients` knob mapped to slots.

---

## 3. Proposed milestones

- **M1 — Route parity (Small/Medium).** Wire the §2.1 routes into `OpenAiCompatServer` + add the matching `NativeServer` plumbing. Highest leverage, lowest risk. **Status: routing wired.** The utility routes `/tokenize`, `/detokenize`, `/apply-template`, `/lora-adapters` (GET+POST), `/count_tokens` (and the `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/v1/messages/count_tokens` aliases) plus the remaining `/v1/audio/transcriptions`, `/v1/chat/completions/control`, `/v1/streams/lookup` and `/v1/stream/:conv_id` (GET/DELETE) are all registered on `OpenAiCompatServer` and delegate to `OpenAiBackend` seam methods. Audio transcription and the control channel return `501 Not Implemented` from the default backend until the native Whisper/control support lands (gated on the §4 `GIT_TAG` bump); the stream-handle registry is a pure-Java route likewise stubbed until integrated with the streaming pipeline.
- **M2 — Utilization/observability.** Extend `ServerMetrics` (GPU/VRAM, queue, prompt-cache, per-model) and surface on `/metrics` + Java API. **Status: done (best-effort on current metrics shape).** `getQueueDepth`, `getPromptCacheHits`, `getKvCacheUsedTokens` / `getKvCacheCapacityTokens` / `getKvCacheUsage`, `SlotMetrics.getCacheHitRate`, `getDeviceCount`, `getDevices`, `getDeviceUtilization(i)` and `getVramUsedBytes(i)` are all added. The per-device GPU/VRAM fields are defensive getters over the upstream `devices` array and return zero until the pinned `GIT_TAG` is bumped to a llama.cpp that emits per-device data (see §4); `getModelMetrics(alias)` per-model aggregation is deferred to M3 (it needs the router `ModelPool`).
- **M3 — Multi-model admin.** `ModelPool` facade: list/load/unload + per-model health/metrics in router mode. **Status: done.** `ModelPool` wraps a running `NativeServer` and delegates to the upstream `GET/POST /models`, `GET /health` and `GET /metrics` endpoints (no new native calls). It offers `listModels()`, `loadModel(alias, params)`, `unloadModel(alias)`, `getModelHealth(alias)` (round-trip latency + status) and `getModelMetrics(alias)` (a typed `ServerMetrics`). In router mode the upstream server aggregates per-worker state, so `getModelHealth` / `getModelMetrics` currently reflect the server as a whole; per-worker breakdown keyed by alias is a follow-up once the router embeds worker addresses in its `/models` response. Model-free HTTP tests added in `ModelPoolTest` against a stub server.
- **M4 — KV-cache admin.** `POST/DELETE /slots`, prompt-cache stats readout. **Status: routes wired.** `POST /slots` (save) and `DELETE /slots/:id` (erase) are registered on `OpenAiCompatServer` and delegate to `OpenAiBackend.saveSlots` / `eraseSlot` seam methods. Both return `501 Not Implemented` from the default backend until the native slot save/erase support lands (gated on the §4 `GIT_TAG` bump); prompt-cache stats readout is already covered by M2's `getPromptCacheHits` / `SlotMetrics.getCacheHitRate`.
- **M5 — Multi-client.** API-key plumbing on `OpenAiServerConfig`, optional token-bucket rate limiter, concurrency defaults. **Status: done.** `OpenAiServerConfig` gains `rateLimitRps` and `maxConcurrentClients` knobs; a `RateLimitFilter` (token-bucket per presented bearer key, or per client address when auth is off, plus a `Semaphore` concurrency gate) is attached to every route and returns HTTP 429 when a client exceeds its budget. `OpenAiCompatServer.setApiKey(String)` rotates the bearer key at runtime without a restart. Both limits are no-ops when disabled (the filter is not attached). Pure-Java, no native changes.

---

## 4. Version-bump dependency

Java pins **b9941**. Router mode, `/models`, `/metrics`, `/slots`, `/health`,
and the existing KV-cache flags are all within b9941. The §2.1 control /
input-tokens / stream / audio routes and the deeper per-device utilization
fields may have landed **after** b9941 — those should be checked against
`b9941` before M1/M2 implementation; if newer, they fold into the same
`GIT_TAG` bump milestone described in `docs/plan-upstream-feature-gaps.md §F`.

---

## 5. Implementation notes

- `NativeServer` router uses **worker subprocesses**; Java-side work is overwhelmingly
  route/JSON plumbing + metrics parsing. Native heavy-lifting stays upstream — minimize
  JNI changes (consistent with the `AGENTS.md` "minimize modifications to server.hpp" rule).
- Keep `ServerMetrics` accessor signatures stable; add new methods rather than changing
  existing ones, so current consumers (dashboards, tests) don't break.
- Each milestone ships with integration tests against a real (small) model where feasible,
  and model-free HTTP/metrics tests otherwise (mirror `server/*IntegrationTest.java`).
- Follow the established pattern: `mvn spotless:apply` + javadoc build before any commit.
