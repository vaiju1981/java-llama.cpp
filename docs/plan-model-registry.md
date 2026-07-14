# Plan: Model Registry & "Pull + Manage" Layer (Ollama-like, but better)

**Status:** Proposed plan only — no code has been written on this branch yet.
**Depends on:** M1–M5 of `docs/plan-llm-serving-server.md` (route parity, observability,
multi-model `ModelPool`, KV-cache admin, guardrails) — all committed.
**Source of truth (upstream):** local checkout `/Users/vaijanath.rao/Work/openSource/llama.cpp`
at `b9964` (Java pins `b9964`; see `AGENTS.md`).

The goal: make java-llama.cpp usable the way Ollama is — `pull`, `list`, `show`, `rm`, `run` a
named model — but *better*: it already speaks OpenAI **and** Ollama protocols, exposes
observability (`ServerMetrics`), enforces guardrails (rate limit / concurrency), and administers
KV-cache slots. This plan adds the missing **model registry + pull** so a model is referenced by a
stable name, not a raw path/URL.

---

## 1. Current state (what already exists)

| Capability | Where | Notes |
|-------------|-------|-------|
| Download model from URL | `parameters/ModelParameters.java` `setModelUrl(url)` | Native llama.cpp fetch (needs `LLAMA_CURL=ON` build; `setOffline(true)` disables). |
| Offline guard | `ModelParameters.setOffline(boolean)` | Maps to upstream `--offline`. |
| Router model list parser | `json/RouterModelsResponseParser.java` | Parses upstream router `/models` wire format. |
| Multi-model admin | `server/ModelPool.java` (M3) | `listModels()`, `loadModel(alias, params)`, `unloadModel(alias)`, `getModelHealth`, `getModelMetrics` over a running `NativeServer`. |
| Model advertisement | `server/OpenAiServerConfig.modelIds` + `OpenAiSseFormatter.modelsJson` | `GET /v1/models` lists configured ids; `GET /api/tags` (Ollama) already exists. |
| Native server CLI | `server/NativeServer.java` `main` | Forwards llama-server args verbatim; router mode spawns worker subprocesses. |

**Conclusion:** the *transport* for pulling (URL download) and the *admin* surface (`ModelPool`)
already exist. What is missing is a **registry abstraction** (name → source/quantization/paths)
and a **pull/lifecycle CLI** that ties them together. Ollama's value is mostly this naming +
lifecycle convenience; we can match it and exceed it with the observability/guardrails already built.

---

## 2. Target capabilities & gaps

### 2.1 Registry (local manifest)
No local index of "what models do I have and where did they come from." Proposed `ModelRegistry`:
- Persists a JSON manifest (`~/.jllama/models.json` or a `--registry` path) mapping
  `name` → `{ localPath, sourceUrl, quantization, sizeBytes, aliases, pulledAt }`.
- Pure Java, no native changes. Enables `list`, `show`, `rm` without touching a running server.
- `ServerMetrics`-style typed view + raw JSON passthrough (mirror existing pattern).

### 2.2 Pull (resolve name/URL → download → register)
- `pull("org/model")` / `pull("https://…/model.gguf")` / `pull("name@q8_0")`.
- For a bare **URL** or **HF repo**, reuse `ModelParameters.setModelUrl` (native curl download).
- For a **short name** (Ollama-style `org/model` or an alias), resolve to a concrete GGUF URL via
  a small resolver:
  - Direct HuggingFace GGUF URL (`hf.co/{org}/{repo}/resolve/{rev}/{file}.gguf`) — buildable from
    the name + an optional quantization tag.
  - Optional curated alias table (`config/registry-aliases.json`) mapping friendly names
    (`llama3.2`, `qwen2.5:7b`) to URLs — the Ollama-library equivalent.
- Gated HF models: pass an HF token via the existing system-property / env mechanism
  (`HUGGING_FACE_HUB_TOKEN` / `HF_TOKEN`); the resolver injects `?download=…&token=…` or an
  `Authorization` header through the native fetch (upstream supports `Authorization` on the URL).
- Honors `setOffline` (refuses to pull; errors clearly).

### 2.3 Management CLI
- Extend or add a `ModelRegistryCli` (`main`) with `pull`, `list`, `show <name>`, `rm <name>`,
  `cp <src> <dst>` (alias), matching Ollama's verbs. Thin wrapper over `ModelRegistry` +
  `ModelPool`. No native changes beyond what `ModelParameters` already exposes.
- Optionally enrich `GET /v1/models` and `GET /api/tags` with registry metadata (quantization,
  size, pulledAt) so dashboards show more than just ids.

### 2.4 Lifecycle integration with `ModelPool` (router mode)
- `ModelPool.loadModel(alias, params)` already exists; extend it to (a) resolve `alias` against the
  registry to fill `ModelParameters` (path or URL + quantization) and (b) lazily `pull` on first
  load when the local file is absent and `offline` is off. This is the "first request pulls the
  model" behaviour Ollama users expect in router mode.

### 2.5 "Better than Ollama" differentiators (already present, call out in docs/CLI)
- Per-model health + metrics (`ModelPool.getModelHealth/getModelMetrics`), queue depth, KV-cache
  usage, prompt-cache hits — Ollama exposes none of this.
- Per-key rate limiting + concurrency gate (M5) — Ollama has none.
- Dual OpenAI + Ollama protocol support — broader client drop-in.
- KV-cache slot save/erase (M4) — finer control than Ollama.

---

## 3. Proposed milestones

- **R1 — Local registry manifest.** `ModelRegistry` (load/save JSON, add/remove/lookup by name,
  typed view + raw JSON). Pure Java, model-free unit tests. No server/native dependency.
- **R2 — Name/URL resolver + pull.** Resolve short names / aliases / URLs to a GGUF source;
  `pull(name)` downloads via `ModelParameters.setModelUrl` and records it in the registry.
  Honors offline. Model-free tests using a stub HTTP file server (download + register); real
  pull validated against a tiny public GGUF URL in CI.
- **R3 — Management CLI.** `ModelRegistryCli` with `pull` / `list` / `show` / `rm` / `cp`, plus
  registry-metadata enrichment of `GET /v1/models` and `GET /api/tags`.
- **R4 — `ModelPool` + registry integration.** `loadModel` resolves aliases via the registry and
  lazily pulls on first load; per-model quantization propagated to `ModelParameters`.
- **R5 — Ollama-registry compatibility (optional).** Mirror Ollama's library/manifest shapes so
  existing Ollama tooling/configs can be reused; document the mapping.

  **Status: implemented.** `OllamaRegistryCompat` (pure Java, model-free) projects the local
  `ModelRegistry` into Ollama's `GET /api/tags` body, enumerating *every* registered model (the
  `ollama list` shape) rather than just the one served model. Field mapping:

  | jllama `ModelRegistryEntry` | Ollama `/api/tags` field |
  |-----------------------------|--------------------------|
  | `name`                      | `models[].name` / `models[].model` |
  | `pulledAt`                  | `models[].modified_at` (ISO-8601) |
  | `sizeBytes`                 | `models[].size` |
  | `quantization`              | `models[].details.quantization_level` |
  | `localPath` / `sourceUrl`   | `models[].digest` (stable, content-derived id — **not** a real sha256) |

  The `digest` is a stable, non-cryptographic id (Ollama uses a sha256 of the GGUF blob; jllama does
  not re-hash here). This is a wire-shape mirror for tooling interop, not a full Ollama blob-store
  replica; full byte-identical Ollama manifests are out of scope.

---

## 4. Dependencies / risks

- **`LLAMA_CURL=ON` build required for URL pull.** The native lib must be built with
  `cmake -B build -DLLAMA_CURL=ON` (see `AGENTS.md` build commands). Document this clearly; the
  CLI should fail with a actionable message if a pull is attempted without curl support.
- **HF URL construction** needs the repo + revision + file; the resolver should validate the
  resulting URL (HEAD/redirect) before handing it to the native downloader. Gated models need a
  token path.
- **No schema-from-POJO / codegen** in this project (per `LlamaModel` javadoc) — keep the manifest
  a hand-written small JSON contract, evolve by adding optional fields.
- **Minimize JNI changes** (AGENTS rule): R1–R3 are pure Java; R4 only reuses existing
  `ModelParameters` setters. No new native methods needed for pull (the download is already native).
- **Restricted networks:** CI/sandbox may block `huggingface.co`. Model-free tests use a stub
  server; the real-pull test is gated behind a system property (mirror AGENTS' model-file pattern).

---

## 5. Implementation notes

- Mirror the established patterns: typed wrapper over a `JsonNode` (like `ServerMetrics` /
  `ModelMeta`), `RawJson` passthrough, `mvn spotless:apply` before commits, model-free unit tests
  against stub HTTP servers (like `ModelPoolTest`).
- `ModelRegistry` file location: default `~/.jllama/models.json`; override via
  `-Dnet.ladenthin.llama.registry.path`. Keep it human-editable.
- Keep `ModelPool` the single admin seam; the CLI and the HTTP `/models` enrichment both go through
  it so there is one source of truth for "what is loaded".
- Each milestone ships model-free tests; the real-pull (`R2`/`R4`) integration test is gated
  behind a system property and only runs where outbound HF is allowed.
