<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# TODO — java-llama.cpp

Open work items for this repo. Cross-cutting tracking lives in
[`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md);
items here are jllama-specific or are this repo's slice of a
cross-cutting initiative.

## Open — jllama-specific

### NativeServer — reuse an already-loaded `LlamaModel` (DONE — attach mode via patch 0007)

**Shipped 2026-07-05** as `NativeServer(LlamaModel, String...)` *attach mode*:
`patches/0007-server-attach-http-frontend.patch` extracts the upstream route table into a shared
`llama_server_register_common_routes(...)` and adds `llama_server_attach(argc, argv,
server_context&)`, which starts only the HTTP frontend (+ stream-session GC) against the
LlamaModel's own `server_context` — exactly the design sketched below: the queue is the
synchronization point, no second model load, no second `start_loop`, no `common_init()` (the JNI
log callback survives), and shutdown goes through the shared `llama_server_request_shutdown()`
path (`ctx_http.stop()` only — never `ctx_server.terminate()`; the caller owns model + backend).
JNI: `native_server.cpp` `startAttachedNativeServer` (resolves the model's `ctx` handle itself).
Contract: close the server before the model. Validated by `NativeServerAttachIntegrationTest`
(HTTP health/props/completion/chat + concurrent direct JNI calls on the same model). The original
feasibility notes are kept below for context.

**Original notes (historical):**

`net.ladenthin.llama.server.NativeServer` (the native-transport server mode that runs the full
upstream `llama_server` — WebUI included — inside `libjllama` over JNI) currently loads its **own**
model from the forwarded argv, exactly like running `llama-server.exe`. This is the "independent
lifecycle" v1: simple, and every llama-server flag is forwarded verbatim.

**Enhancement:** let `NativeServer` optionally attach to an **already-loaded** `LlamaModel`'s
`server_context` instead of loading a second copy of the weights (saves the RAM/VRAM and load time
of a duplicate model when a caller already has a `LlamaModel` open). Feasibility notes from the
initial investigation:

- The upstream HTTP transport (`server_http_context`) and the route bundle
  (`server_routes routes(params, ctx_server)`) only need a reference to a `server_context`. A
  `LlamaModel` already owns and drives one (`jllama_context` in `jni_helpers.hpp`), and its JNI
  methods already post tasks to that context's queue — so a second driver (the HTTP routes) posting
  to the same queue is plausible; the queue is the synchronization point.
- The real work is **lifecycle/ownership**: today `llama_server()` owns the whole flow (parse →
  backend init → `ctx_server.load_model` → `start_loop` on its own thread → cleanup). Reuse would
  need a *different* entry that skips model loading and the `start_loop`/backend ownership (the
  existing `LlamaModel` worker already runs the loop), registers the HTTP routes against the shared
  `server_context`, and starts only `server_http_context`. That is a separate, smaller C++ entry
  point (not `llama_server`), plus reconciling params (the loaded model's params vs. server params)
  and ensuring only one thread drives `update_slots`.
- Logging: `llama_server` calls `common_init()` which routes llama.cpp logging to stderr/file; a
  reuse path must not clobber the JNI log callback a `LlamaModel` consumer may rely on.

Until then, run `NativeServer` standalone (it owns the process's llama backend + logging while
running), or use the Java-transport `OpenAiCompatServer` when sharing a `LlamaModel`.

### GGUF metadata inspector (DONE — GgufInspector)

**DONE (2026-07-05).** `GgufInspector` (root) + `value.GgufMetadata`: pure-Java GGUF v2/v3
header + key/value reader (LE + BE auto-detect, fail-loud on v1/corrupt/truncated, sanity
caps, stops before tensor data) with typed accessors (architecture, name, parameter count,
context length via `<arch>.context_length`, `general.file_type`, chat template) — inspects a
model WITHOUT loading it (complements the loaded-model `getModelMeta()`). 21 model-free tests
against in-memory generated fixtures + a gated real-file check; GgufMetadata in the PIT 100%
gate.

### Session fork/rewind (DONE — SessionCheckpoint)

**DONE (2026-07-05).** `Session.checkpoint(filepath)` → `value.SessionCheckpoint` (slot
KV-save file + transcript-turn snapshot, caller-managed file), `Session.rewind(checkpoint)`
(atomic restore of KV state + transcript under the session lock), `Session.fork(newSlotId,
filepath)` (independent branch on another slot, same system message/customizer; needs
`setParallel(2)+`). All three rejected mid-stream (same guard as save/restore). Plumbing:
`ChatTranscript.turnsSnapshot()/resetTurns(...)`, `SessionState.turnsSnapshot()/
restoreTurns(...)/getSystemMessage()`. Model-free tests for the bookkeeping + streaming
guards; `SessionForkRewindIntegrationTest` (gated) covers rewind-continue, independent fork,
and the own-slot guard against a real model.

### Typed router API (DONE — RouterClient)

**DONE (2026-07-05).** `server.RouterClient` + `value.RouterModel` (+ nested `Status` enum) +
`json.RouterModelsResponseParser` wrap the router-mode model-management endpoints
(`GET /models`, `POST /models/load`, `POST /models/unload`) with typed list/find/load/unload and
`awaitModelLoaded(id, timeout)` (poll-until-LOADED with fail-fast on the router's
`status.failed`/`exit_code` worker-death marker and on unknown ids). 25 model-free unit tests
(`RouterModelTest`, `RouterModelsResponseParserTest`, `RouterClientTest` against a stub HTTP
server); `RouterModeIntegrationTest` now drives discovery/load/readiness through the client
against a real router. Layered-architecture rule updated (Server may access Json); RouterModel is
inside the PIT 100% gate (274/274).

### PIT gate not hermetic — `value.ContentPart.audioFile(Path)` (RESOLVED — hermetic since the reactor move)

**Resolved.** `ContentPartTest` carries hermetic `@TempDir` tests for `audioFile(Path)`
(`audioFileDetectsWavFromExtension` incl. case-insensitive `.WAV`, `audioFileDetectsMp3FromExtension`,
`audioFileRejectsUnknownExtension`) — the exact fix this entry proposed, mirroring the `imageFile(Path)`
temp-file tests. Verified 2026-07-05 in a fixture-less, network-restricted sandbox:
`mvn -f llama/pom.xml test-compile org.pitest:pitest-maven:mutationCoverage` → **295/295 killed (100%),
0 NO_COVERAGE**. No committed audio fixture is needed for the PIT gate. (Unrelated and still true: the
model-backed `AudioInputIntegrationTest` now has a committed default prompt clip
(`src/test/resources/audios/sample.wav`, MIT-granted by the project author, REUSE-annotated) and
self-skips only when the audio model/mmproj are not staged — a test-coverage improvement, not a PIT
concern.)

### Code audit — pre-existing correctness / safety findings (RESOLVED — PRs #258 + #260)

A multi-area audit (2026-06-20) of the **existing** codebase surfaced 18 correctness/safety findings,
intentionally **split into tiers so each could land as its own small, focused PR**. **All 18 are now
fixed and merged** — Tiers 1–3 in **#258**, the deferred `LlamaLoader` extraction race in **#260** —
with regression tests added in **#261 / #262**. The full per-finding rationale lives in those PRs and
their commits; the concise record below is kept for traceability. Nothing in this section is open
except the optional follow-up noted at the end.

**`LlamaLoader` native-lib extraction temp-path race — DONE (atomic write + content-reuse).**
`extractFile` now (1) reuses a byte-identical existing copy instead of rewriting it — so it never
replaces a file another JVM has already loaded (which fails on Windows) — and (2) otherwise extracts
to a per-attempt unique temp file and **atomically moves** it into place, so a concurrent loader can
never observe a half-written library. `jllama` is statically linked (`BUILD_SHARED_LIBS OFF`), so the
extracted file is self-contained — no multi-DLL co-location to coordinate. Verified by the
`NativeLibraryLoadSmokeTest` (real extract+load on macOS) + a `resourceMatchesFile` unit test; the
Windows locked-replace path is exercised by CI's Windows jobs.

**Tier 1 — high impact (#258, `a4325ff`)**

- **N1** — unhandled C++ exceptions crossing the JNI boundary → JVM abort; every entry point (incl. the
  public `LlamaModel.jsonSchemaToGrammar`, plus encode/tokenize/embeddings/rerank/infill/applyTemplate)
  now converts the failure to a `LlamaException` instead of crashing the process.
- **N2** — `parse_string_array` null-deref (null element / OOM) + per-iteration JNI local-ref leak.
- **J1** — `close()` / native `delete()` double-free under concurrent close → `synchronized` close.
- **P1** — `ServerMetrics` cumulative token totals truncated `int` → negative → `Timings.promptN` /
  `predictedN` widened to `long`.

**Tier 2 — medium (#258, `a4325ff` + `3e500aa`)**

- **S1** — unbounded request body → OOM DoS → 16 MiB cap + `Content-Length` pre-check; oversized → HTTP 413.
- **N3** — streaming-reader use-after-free → reader held as a `shared_ptr` and copied out under the lock
  before `next()`.
- **J5** — `Session` permanently wedged on an abandoned stream → `Session.cancelStream()` clears the guard
  and rolls back the pending user turn.
- **J3** — `LlamaIterator.hasNext` made `volatile` (observed across a cross-thread `cancel()`).
- **N4** — log callback made `noexcept` + non-throwing env lookup (no exception unwinds through llama.cpp
  C frames from an unattached thread).

**Tier 3 — hardening (#258, `ac3ad6d`)**

- **S3** — constant-time bearer-key comparison (`MessageDigest.isEqual`).
- **S2** — SSE heartbeat pool sized to the core count (one stalled client can't starve other streams).
- **P3** — `ChatMessage.toolCalls` defensively copied + wrapped unmodifiable.
- **NaN/Inf** — non-finite `float`/`double` rejected at `JsonParameters.withScalar` (they would serialize
  to the invalid JSON tokens `NaN`/`Infinity`).
- **OSInfo** — armhf-detection `exec()` routed through a drain-and-close helper (no fd leak / pipe-full hang).
- **completeBatch** — `completeBatch` / `completeBatchWithStats` / `chatBatch` join every future before
  propagating the first failure (no abandoned in-flight requests).
- **Docs** — `/props` + Ollama discovery routes documented as intentionally-unauthenticated metadata;
  `parseProbabilities` documented as last-wins on duplicate token text (use `parseLogprobs` for lossless data).

**Still open — optional follow-up (lower priority):** full per-process extraction **directory** isolation
+ a `cleanup()` that recursively removes dead-process dirs. Now that writes are atomic and content-checked
this is a tidiness improvement (stops the shared-tmpdir `cleanup()` racing a live peer's flat file), not a
correctness fix — and it still needs the Windows locked-file co-design noted above.

### OpenAI-compatible HTTP endpoint (shipped; follow-ups open)

`net.ladenthin.llama.server.OpenAiCompatServer` is the single OpenAI-compatible server (JDK
`com.sun.net.httpserver`, no new dependency, fat-jar `Main-Class`). It exposes the OpenAI routes
`POST /v1/chat/completions` (streaming SSE + non-streaming), `/v1/completions`, `/v1/embeddings`,
`/v1/rerank`, `/infill`, `GET /v1/models`, `GET /health` and `GET /props`, **plus three alternative
protocol surfaces** — Ollama-native (`/api/version`, `/api/tags`, `/api/show`, `/api/chat`,
`/api/generate`), Anthropic Messages (`POST /v1/messages`) and OpenAI Responses (`POST /v1/responses`).
Every route is also reachable without the `/v1` prefix and sits behind a CORS filter. The CLI is parsed
by the testable `OpenAiServerCli`. (Consolidated from PR #240's JDK + streaming server and #242's
NanoHTTPD server; NanoHTTPD + its dependency deleted.)

**IDE/agent backend hardening — DONE** (from the deep-research investigation
[`docs/feature-investigation-ide-agent-backend.md`](docs/feature-investigation-ide-agent-backend.md);
primary goal: agentic tool-calling with Qwen):

- Agentic tool-calling verified wire-correct: C++ guard pins `tool_calls.function.arguments` as a JSON
  **string** (not object) at b9739 (llama.cpp #20198), plus the existing `finish_reason:"tool_calls"`
  test.
- `stream_options.include_usage` forwarded (new `InferenceParameters.withStreamOptions`) so the trailing
  usage chunk is emitted, and `OpenAiSseFormatter.ensureUsageCachedTokens` guarantees
  `usage.prompt_tokens_details.cached_tokens` (fixes the Copilot custom-endpoint crash, vscode #273482).
- `response_format` (`json_object`/`json_schema`) forwarded for structured outputs.
- `POST /infill` (FIM autocomplete for llama.vscode/Twinny/Tabby/Continue) → native `handleInfill`.
- `POST /v1/rerank` (RAG) → `handleRerank` reshaped to `results`/`data` (`OaiRerankSupport`).
- CORS preflight + `Access-Control-Allow-Origin`; bare-path (no `/v1`) aliases; `cache_prompt=true`
  default; `--mmproj` (vision), `--embedding`, `--reranking` CLI flags.
- **Alternative protocol surfaces** (pure translation over the OpenAI core; tool calls reconstructed by
  `ToolCallDeltaAccumulator`): **Ollama-native** (`/api/version`, `/api/tags`, `/api/show`, `/api/chat`
  with NDJSON streaming, `/api/generate` prompt-completion/FIM — `OllamaApiSupport`; `/api/show`
  advertises tools/insert/vision + context length); **Anthropic Messages** (`POST /v1/messages`, SSE
  events — `AnthropicApiSupport` + `AnthropicStreamTranslator`); **OpenAI Responses** (`POST
  /v1/responses`, SSE events — `ResponsesApiSupport` + `ResponsesStreamTranslator`).
- **`GET /props`** (llama.cpp-native): `default_generation_settings.n_ctx` + `modalities` so autocomplete
  clients (llama.vscode) size their context window (`OpenAiSseFormatter.propsJson`).
- Gated **integration round-trips** over a real socket, run in CI's `test-java-linux-x86_64` job,
  self-skipping when the model is absent — structural assertions only:
  - `OpenAiCompatServerIntegrationTest` (Qwen3-0.6B, chat mode): OpenAI chat (non-stream/stream/tools/
    models) plus Ollama `/api/chat` + discovery, Anthropic `/v1/messages`, OpenAI `/v1/responses`
    (non-stream + stream) and `/props`.
  - `OpenAiServerEmbeddingsIntegrationTest` (CodeLlama-7B + `enableEmbedding`): `/v1/embeddings` (+ bare
    alias).
  - `OpenAiServerRerankIntegrationTest` (jina-reranker + `enableReranking`): `/v1/rerank` (sorted
    `results`/`data`, `top_n` cap).
  - `OpenAiServerCompletionIntegrationTest` (CodeLlama-7B): `/v1/completions`, `/infill`, and Ollama
    `/api/generate` (plain + FIM via `suffix`).

**Open follow-ups (deferred):**

- **Streaming raw-completion path — IN PROGRESS (no new native method needed).** The earlier premise was
  wrong: a streaming raw-completion JNI path **already exists** (`requestCompletion`/`receiveCompletionJson`,
  exposed as `LlamaModel.generate(InferenceParameters) → LlamaIterable`), so this is **Java-only server
  wiring**, not JNI/C++. Progress: **(a) streaming `POST /v1/completions` — DONE** (`OpenAiRequestMapper`
  `toCompletionParameters` + `OpenAiBackend.streamCompletions` driving `generate()` + an
  `OpenAiSseFormatter.completionChunk` `text_completion` chunk + the `streamCompletions` SSE handler;
  HTTP test green). **Remaining:** (b) **token-streaming Ollama `/api/generate`** (translate the
  `text_completion` chunks to NDJSON, mirroring the chat→Ollama translator) and (c) **Continue's native
  `POST /completion`** route in the llama.cpp-native streaming shape (`{"content":…,"stop":…}` per chunk).
- **Future *output* modalities (audio / image) — design note, not yet actionable.** llama.cpp's server
  produces **text** (plus embeddings/rerank); it does **not** generate images or audio output, so there is
  no engine behind a TTS/image-gen response today and building that API surface now would be dead code.
  When/if it becomes real, the integration points are already isolated: a new `OpenAiBackend.stream*`
  primitive + an `OpenAiSseFormatter.*Chunk` formatter per modality, wired into a per-route handler — the
  exact shape the text `streamCompletions` path now establishes. Two concrete future hooks: (1) llama.cpp's
  **OuteTTS** audio path (if it lands in the embedded server) → an `/v1/audio/speech`-style route emitting
  audio chunks; (2) routing image/audio generation to an **external** model behind the same server (the
  binding would proxy, not generate). Keep `LlamaOutput`/chunk formatters modality-neutral so neither
  requires reworking the streaming core.
- **Incremental tool-call streaming on the alternative surfaces.** Ollama/Anthropic/Responses emit each
  tool call *whole* at end-of-stream (reconstructed by `ToolCallDeltaAccumulator`) rather than streaming
  argument fragments. Fine for clients that apply tool calls after generation; revisit if a client needs
  incremental `input_json_delta` / `function_call_arguments.delta` fidelity.
- **Per-model FIM template registry** (Qwen/CodeLlama/DeepSeek v1&V2/StarCoder2/Codestral) — only needed
  if we also expose `/v1/completions`-with-`suffix` FIM; `/infill` (and Ollama `/api/generate` with a
  `suffix`) applies the model's FIM tokens server-side, so this is lower value.
- **Multi-model registry.** Only one model id is advertised/served today; serving several would need
  multi-model load + lifecycle management.
- **Manual real-client validation.** Gated server-side round-trips now exist for every surface (above).
  What remains is manual validation against the actual editor clients — point Copilot's Ollama provider /
  a Custom Endpoint, Claude Code, and a Responses client at the running server — since a server-side
  round-trip confirms the wire shapes but not each client's own parser.
- **Gemma 4 tool-calling validation.** Confirm the pinned llama.cpp (`b9789`) includes the Gemma 4
  tool-call parser fixes; if not, bump per the upgrade procedure.
- **NativeServer — wire upstream `server.cpp` routes to JNI (in progress; scaffold landed `dd264b2`).**
  The upstream HTTP transport (`tools/server/server-http.cpp` + the cpp-httplib backend) is already
  compiled into `libjllama`, and a `server.NativeServer` Java scaffold + `NativeServerSmokeTest` landed
  in `dd264b2`. **Remaining:** wire the upstream `server.cpp` route table (the one upstream TU still
  excluded from the build — it carries `main()` + route wiring) to JNI so the native HTTP server (and the
  embedded WebUI) can be started/stopped from Java. This is the **native-transport alternative** to the
  JDK-based `OpenAiCompatServer` (which is complete and the primary surface); value is shipping the full
  llama.cpp server + WebUI in-process without a separate `llama-server` binary. JNI + C++ work.

### Windows native classifiers — default flip (Ninja default + MSVC classifier) + CUDA/Vulkan/OpenCL GPU

**Design decision UPDATED by the owner (supersedes the earlier "MSVC is the permanent default"
note): the default Windows CPU JAR is now the Ninja Multi-Config build, and the MSVC / Visual
Studio build ships as the `msvc-windows` classifier.** Rationale: both generators use the same MSVC
toolchain (`cl.exe`, static `/MT` CRT) on the same runner, so the produced DLLs are functionally
equivalent with identical runtime dependencies — the only difference is build-system plumbing +
sccache caching. Making Ninja the default gives the most-pulled JAR the cache; MSVC stays available
as a classifier. Three Windows GPU classifiers were added at the same time (x86_64 only, all Ninja):
`cuda13-windows-x86-64`, `vulkan-windows-x86-64`, `opencl-windows-x86-64`.

**Why the cache needs Ninja.** The cache mechanism is the CMake *compiler launcher*
(`-DCMAKE_C_COMPILER_LAUNCHER=sccache`); the Visual Studio generator ignores it entirely, only
Ninja/Makefile generators honor it. Upstream llama.cpp also builds its Windows artifacts with Ninja
Multi-Config + MSVC.

**What shipped (this branch — pending first CI validation):**
- **CPU build jobs:** `build-windows-x86_64` / `build-windows-x86` are now **Ninja** (default,
  artifacts `Windows-{arch}-libraries`); `build-windows-x86_64-msvc` / `build-windows-x86-msvc` are
  **MSVC** (artifacts `Windows-{arch}-msvc`). `test-java-windows-x86_64` (default/Ninja) and
  `test-java-windows-x86_64-msvc` both load the DLL via JNI and run the full model-backed suite.
- **GPU build jobs (x86_64, Ninja, build the artifact only — runners have no GPU, and a
  GPU-linked jllama_test can't be enumerated there; C++ suite runs on the CPU jobs):**
  `build-windows-x86_64-cuda` (`Jimver/cuda-toolkit@v0.2.35` CUDA `13.2.0` + `-DGGML_CUDA=ON`),
  `build-windows-x86_64-vulkan` (`jakoch/install-vulkan-sdk-action` + `-DGGML_VULKAN=ON`),
  `build-windows-x86_64-opencl` (`build_opencl_windows.bat` stages the ICD loader + `-DGGML_OPENCL=ON`).
- **`CMakeLists.txt`** — OS-aware backend routing (CUDA/OpenCL → Windows trees, new Vulkan branch).
- **`.github/build.bat`** — also wraps nvcc with sccache for CUDA builds.
- **`.github/build_opencl_windows.bat`** — new, Windows analogue of `build_opencl_android.sh`.
- **`pom.xml`** — profiles `windows-msvc` / `cuda-windows` / `vulkan-windows` / `opencl-windows`
  (classifiers `msvc-windows` / `cuda13-windows-x86-64` / `vulkan-windows-x86-64` / `opencl-windows-x86-64`).
- **`publish.yml`** — the `package` / `publish-snapshot` / `publish-release` jobs download each
  non-default artifact into `src/main/resources_windows_{msvc,cuda,vulkan,opencl}/` and activate the
  four profiles; all five Windows build jobs are in the `package` `needs:` graph.
- Docs: `README.md` classifier table + `CLAUDE.md` "Windows native classifiers" section.

**Verification — first CI run done (PR #276, run 28327740376).** Green on the first try: default Ninja
CPU flip (x64+x86), MSVC classifier (x64+x86), and the **OpenCL** GPU job (`build_opencl_windows.bat`
ICD staging works). Two GPU jobs were fixed after the first run: **CUDA** (`Version not available:
13.0.0` → bumped `Jimver/cuda-toolkit` `v0.2.24`→`v0.2.35` + `13.2.0`) and **Vulkan**
(`find_package(Vulkan)` couldn't read the `humbletim` SDK layout → switched to
`jakoch/install-vulkan-sdk-action`). Re-run pending to confirm both fixes.

**Optional follow-up:** smoke-test that each *published* classifier JAR loads its DLL on a clean
Windows host with the matching GPU driver/toolkit installed.

**Reference notes:**
- Cache backend is **sccache + Depot WebDAV** (consistent with the other 8 jobs — one token, shared
  cross-branch) rather than upstream's per-branch ccache. sccache supports MSVC `cl.exe`; the
  Release config emits no debug info, so the `/Zi`→`/Z7` PDB caveat doesn't apply.
- It is **"Ninja Multi-Config"**, not plain Ninja — it keeps multi-config semantics, so
  `cmake --build … --config Release` and the config-specific `RUNTIME_OUTPUT_DIRECTORY_RELEASE`
  properties behave exactly as under the VS generator; `/MT` runtime and x64-vs-x86 gating unchanged.
- The arch (`x64`/`x86`) comes from `ilammy/msvc-dev-cmd@v1`, not a `-A` flag (Ninja takes no `-A`).

### Known regression (b9739) — Windows JNI: `common_params_parse` ignores caller argv

**Status: FIXED via local source patch (`patches/0001-win32-arg-parse-embed-guard.patch`).** Surfaced
while bringing PR #248 green (the b9739 build fixes let the Windows Java jobs run to completion and
exposed this). Applied through the generic `patches/` mechanism (see CLAUDE.md "Local llama.cpp source
patches"), so it covers every C++ build and re-applies on each clean build.

**Note on the fix shape (count-guard → deterministic removal).** The first patch used fix option 1
below — the count-guard (override only when the re-derived arg count equals `argc`). It fixed 21/25
Windows Java tests, but **collided** on the 4 server-integration setups (`OpenAiServerRerank*`,
`OpenAiServerToolCalling*`, `MultimodalIntegrationTest`, `OpenAiCompatServerIntegrationTest`) whose
argv length happened to equal `java.exe`'s, so they kept failing with the same parse error. The patch
was changed to **fix option 2** (drop the override entirely for our build — a JNI library is never the
process, so the override is pure liability), which is deterministic. **As of the b9789 bump the patch
was reshaped into the clean opt-in form intended for upstreaming (fix option 3's core):**
`common_params_parse` now parses exactly the argv it is given, and a new `common_params_parse_main()`
wrapper carries the `GetCommandLineW` UTF-8 recovery that the standalone tools' `main()` opt into.
**The patch now carries the full upstream change (37 files):** the ~34 `common_params_parse(argc, argv,
…)` call sites across `tools/*`, `examples/*` and the `tests/*` programs flip to
`common_params_parse_main()`, plus a `tests/test-arg-parser.cpp` regression case. Embedded callers stay
on `common_params_parse`. Our subproject build compiles only the `arg.{cpp,h}` core
(`LLAMA_BUILD_TOOLS`/`TESTS` OFF), so the flips + test are validated via a one-off tools+tests build
(the new test's asserts pass; `test-arg-parser`'s only red is the live `ggml.ai` download check, which
is sandbox-network). The 37-file patch must be re-verified on each llama.cpp bump (the applier fails
loud). Submit it to llama.cpp and drop the local copy once merged.

**Symptom.** On **Windows x86_64 only**, every Java test that loads a real model fails in
`LlamaModel.loadModel` (native) with `LlamaException: "Failed to parse model parameters"`
(25 errors in `Java Tests Windows 2025 x86_64`, both the VS *and* Ninja DLLs). macOS and Linux Java
tests pass. The argv we build is platform-neutral (`--model models/<file>.gguf`, relative, forward
slashes — `TestConstants.MODEL_PATH`), so it is **not** the Windows-Ninja build, **not** our argv,
and **not** a path/escaping issue.

**Root cause (upstream llama.cpp, new in b9739).** `jllama.cpp` (`load_model_impl`, ~line 606) builds
a CLI argv from `ModelParameters` and calls upstream
`common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)`. In b9739, `common/arg.cpp`'s
`common_params_parse` gained a **Windows-only** prologue (arg.cpp:924-931):

```cpp
bool common_params_parse(int argc, char ** argv, ...) {
#ifdef _WIN32
    auto utf8 = make_utf8_argv();          // = CommandLineToArgvW(GetCommandLineW())
    if (!utf8.ptrs.empty()) {              // always non-empty under a JVM
        argc = (int) utf8.buf.size();
        argv = utf8.ptrs.data();           // DISCARDS the caller-supplied argv
    }
#endif
    ... common_params_parse_ex(argc, argv, ctx_arg) ...
}
```

It unconditionally replaces the caller's argv with the host **process** command line
(`GetCommandLineW()`). For the standalone `llama-server.exe` this is correct (fixes UTF-8 CLI args).
For an **embedded/JNI** caller the process is **`java.exe`**, whose command line has no `--model`, so
`common_params_parse_ex` fails and `common_params_parse` returns `false` → our "Failed to parse model
parameters". `common_params_parse_ex` is `static`, so we cannot bypass the block by calling the inner
parser. Our JNI already passes correct UTF-8 argv (`GetStringUTFChars`), so the re-derivation is
unnecessary for us. **This is an upstream bug affecting every embedded Windows consumer of
`common_params_parse`.**

**Fix options (history — option 2 chosen).** (1) guard the block by arg-count — *tried first, it
collided* (see the count-guard note above); (2) **remove the `_WIN32` override for our build — CHOSEN**
(deterministic; our JNI always passes correct UTF-8 argv); (3) file an upstream PR and wait. The patch
re-applies on every llama.cpp bump and the applier fails loud if it stops applying — it is part of the
upgrade checklist. Pre-existing on `main` since #247 (b9682→b9739); independent of the Windows-Ninja
classifier work. **Remaining open item: the upstream PR** (see "Upstream llama.cpp PR" below) so the
local patch can eventually be dropped.

### SonarCloud "Security Rating on New Code" gate — PR #248 (open)

The PR's **only** red is SonarCloud's "Security Rating on New Code" gate (every build/test job is
green; SonarCloud is **not** a merge-blocking build job). The findings are GitHub-Actions/Java
analyzer issues from the Maven scanner — **"C" is the rating *grade* (A–E), not the C language**;
there is no CFamily/C-C++ scan configured. Addressed:

- **`clang-format.yml`** — `pip install` without `--only-binary :all:` can run a package's `setup.py`;
  forced wheels-only (`84297e0`, block scalar so `:all:` doesn't break YAML). *If Sonar still flags it,
  try the `--only-binary=:all:` equals form.*
- **`osv-scanner.yml` / `scorecard.yml`** — top-level `permissions: read-all` → `contents: read`
  (`84297e0`); safe because every job in both files already declares its own exact permissions.
- **`publish.yml`** — workflow-level `permissions: contents: read` (Sonar wants it per-job); **owner
  marked it Accept/"Won't fix" on the dashboard** rather than spreading perms across ~25 release jobs.
  Alternative if ever desired: add `permissions: contents: read` to the ~19 read-only jobs (the 5
  publish/report jobs already declare `contents: write`) and drop the top-level block.
- **`PairTest.java`** — 3 Critical *Reliability* bugs (`assertNotNull` on the primitive `hashCode()`)
  replaced with a determinism check (`9f0d377`). Reliability rating, **not** the Security gate.

**Still open:** the gate was still red as of `9f0d377`. SonarCloud's issues API is auth-gated (403 from
CI), so the exact remaining new-code Vulnerability must be read off the dashboard. Resolve the last
finding, accept it on the dashboard, or merge on the green build/test checks.

### License Compliance (FOSSA-style dependency-license gate) — PR #248 (open)

Separate from the FSFE **REUSE** check (which is green — `reuse lint` reports 266/266 files compliant)
and from SonarCloud: the PR's combined commit status shows a **"License Compliance" check failing with
"17 issues found"** (an error-state commit status posted by a license-scanner GitHub App, not a
workflow in `.github/workflows/`). It contributes to the `mergeable_state: blocked` on #248.

- **Almost certainly pre-existing**, not introduced by this PR: #248 changes **no dependencies** (the
  `pom.xml` edit only adds the `windows-ninja` build profile), so the 17 are dependency-license policy
  findings already present on `main` (e.g. GPL-2.0 carried by the llama.cpp sources).
- **Not yet inspected** — the scanner's dashboard/host is outside this sandbox's egress allowlist, same
  as `sonarcloud.io`. To triage: open the check's details link from the PR (or allowlist the host), read
  the 17 findings, then accept policy-OK licenses on the dashboard or adjust the policy. Confirm whether
  it is a *required* status (if so it blocks merge; if advisory it does not).

### Upstream llama.cpp PR — drop the local Windows arg-parse patch (open)

`patches/0001-win32-arg-parse-embed-guard.patch` is a **local** fix re-applied on every build. To drop
it, PR upstream (against #24779): add a `common_params_parse_argv` companion (or a
`common_params_parse` opt-out flag) that trusts the caller's argv — preserving the standalone tools'
UTF-8 fix while letting embedders (JNI, and any FFI binding) pass their own argv. Ship with the
standalone-safe repro (a plain exe that passes a synthetic argv and shows it gets discarded on Windows
because `GetCommandLineW()` returns the host process line). Once merged and the pin is bumped past it,
delete the patch.

### Branch protection — aarch64 job renamed (open, owner action)

The native aarch64 switch renamed the check **`Cross-Compile Linux aarch64 (LTS)` → `Build and Test
Linux aarch64`**. If a required status check pinned the old name, repoint it or it will sit pending
forever.

### llama.cpp upstream feature exposure (queued, deferred by policy)

These are JNI plumbing items for upstream API additions. Policy: add only after a real user request — they are mostly relevant to specific model families or specialized workflows.

- **Expose `--spec-draft-backend-sampling` toggle via `ModelParameters.setSpecDraftBackendSampling(boolean)`.** Added in b9437 (env `LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING`). Backend sampling for the speculative draft is enabled by default upstream but auto-disabled on `LLAMA_SPLIT_MODE_TENSOR` setups; an explicit Java-side setter lets callers force-disable it for benchmarking or for backends with sampler bugs. Speculative-decoding power users.

- **Expose runtime reasoning control via `InferenceParameters.setReasoningControl(boolean)` + `LlamaModel.endReasoning(...)`.** Added in b9444–b9490: new `common_params_sampling::reasoning_control` flag arms the budget sampler so reasoning can be ended at runtime, and new `common_sampler_reasoning_budget_force(common_sampler *)` triggers the end-of-thinking token injection on the next sample. Upstream also adds a `POST /v1/chat/completions/control` server endpoint accepting `{"id": "...", "action": "reasoning_end"}`. Java mapping would be: (a) `InferenceParameters.setReasoningControl(boolean)` arms the sampler on the inference run, (b) a new `LlamaModel.endReasoning(int slotId)` (or per-streaming-task-id) JNI method calls the upstream `common_sampler_reasoning_budget_force` against the slot's sampler. Useful for interactive UIs that want a "skip thinking and answer now" button. Relevant only for reasoning-trained models (DeepSeek-R1, Qwen3-Thinking, GPT-OSS-Reasoner, etc.).

- **Expose `llama_context_params::n_outputs_max` via `ModelParameters.setMaxOutputs(int)`.** Added in b9444–b9490 (default `-1` = derived from `n_batch`). Caps the number of output slots allocated per context; relevant for memory-constrained setups that always run with `logits_all=false` and want to prevent over-allocation when `n_batch` is large. Trivial JNI plumbing (one `cparams` field passthrough); add when a user reports OOM on context creation tied to output slot pre-allocation.

- **Expose Multi-Token Prediction toggle via `ModelParameters.setMtp(boolean)`.** Existed since the Qwen3.5 MTP work; b9444–b9490 extends it to Step-3.5. CLI flags `--mtp`/`--no-mtp` (env `LLAMA_ARG_MTP`) control whether the draft head runs alongside the main model for accelerated decoding. Java setter would route to `common_params_speculative::type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP`. Relevant only for MTP-trained models.

- **Expose `llama_vocab::get_suppress_tokens()` via `LlamaModel.getSuppressTokens()`.** Added in b9490–b9495 alongside the new `tokenizer.ggml.suppress_tokens` GGUF key and the `LLM_KV_TOKENIZER_SUPPRESS_TOKENS` constant. When a GGUF declares this array, upstream stores it on `llama_vocab::impl::suppress_tokens` and exposes it via the new `llama_vocab::get_suppress_tokens()` accessor. The bias is **applied automatically** inside the model forward graph — the Gemma4 Unified graph (`src/models/gemma4.cpp`) reads the list and adds a `-INFINITY` logit bias to those token IDs via a new `llm_graph_input_logits_bias` input so the model cannot emit them (used to block `<image|>` / `<audio|>` placeholders). A Java mirror would be `public int[] getSuppressTokens()` on `LlamaModel`: a read-only inspector returning the suppression list for debugging or for callers running their own sampling who want to replicate the same bias. Value is low (the bias is auto-applied, Java callers cannot change it; java-llama.cpp does not expose custom logit-bias hooks at this level); cost is trivial (one JNI passthrough + a `getSuppressTokens()` Java method).

### JNI safety and server hardening (from PR #251 contributor)

Raised by [@vaiju1981](https://github.com/vaiju1981) in
[PR #251 comment](https://github.com/bernardladenthin/java-llama.cpp/pull/251#issuecomment-4761363838).
Feel free to contribute fixes — PRs welcome.

- **Unhandled C++ exceptions cross the JNI boundary → JVM abort (UB).** Any `std::exception`
  (or worse, an exception of unknown type) that escapes a native method and crosses the JNI
  boundary causes undefined behaviour on most JVMs and typically aborts the process. Each native
  method in `jllama.cpp` should wrap its body in `try { … } catch (const std::exception& e) {
  env->ThrowNew(llamaExceptionClass, e.what()); return <zero>; } catch (...) { env->ThrowNew(…,
  "unknown C++ exception"); return <zero>; }` so that errors surface as `LlamaException` on the
  Java side instead of crashing the JVM.

- **`parse_string_array` — null deref + JNI local-reference leak.** The helper that reads a
  JSON string array from JNI can dereference a null pointer when an array element is absent,
  and leaks JNI local references when an early exit skips the matching `DeleteLocalRef`. Fix:
  guard every `GetObjectArrayElement` result and pair each reference acquisition with a
  `DeleteLocalRef` before the next iteration or return.

- **`close()` / native `delete()` double-free under concurrent close.** If two threads race to
  call `LlamaModel.close()`, both can reach the native `delete` path and free the same
  `jllama_context` pointer twice → heap corruption. Fix: use `AtomicBoolean closed` + a
  `synchronized` guard (or `compareAndSet`) on the Java side so `close()` is idempotent and
  the native pointer is nulled before the second caller can reach it.

- **`ServerMetrics.getCumulativeTimings()` truncates cumulative token totals to `int`.** The
  cumulative token counters are stored as `long` in the JSON but cast to `int` when
  constructing `ServerMetrics`, silently truncating values above `Integer.MAX_VALUE`
  (~2.1 billion tokens). Fix: widen the field and constructor parameter to `long`.

- **Unbounded request-body read → OOM DoS.** The HTTP handler reads the entire request body
  into a `String`/`byte[]` before parsing it, with no size cap. A client that streams a
  multi-gigabyte body can exhaust heap memory and crash the JVM. Fix: add a configurable
  `maxRequestBodyBytes` limit (e.g. default 4 MB) and reject oversized requests with
  `HTTP 413 Content Too Large` before buffering them.

### Feature backlog from similar projects

- **Feature backlog from similar projects.** See [`docs/feature-investigation-similar-projects.md`](docs/feature-investigation-similar-projects.md) for the consolidated investigation across the 5 pure-Java sibling runtimes ([llama3.java](https://github.com/mukel/llama3.java), [gemma4.java](https://github.com/mukel/gemma4.java), [gptoss.java](https://github.com/mukel/gptoss.java), [qwen35.java](https://github.com/mukel/qwen35.java), [nemotron3.java](https://github.com/mukel/nemotron3.java)) plus the dormant alternative JNI binding [llamacpp4j](https://github.com/sebicom/llamacpp4j). The doc captures 18 candidate items grouped into cross-cutting themes (UTF-8 streaming boundary safety, thinking-channel router, operator timing line, jbang single-file example, README system-properties table, etc.) and per-repo unique findings (Harmony channel decoder, Qwen empty-`<think>` injection, llama_state_* save/load, llama_adapter_lora_* hot-apply, etc.), each with effort sizing (XS / S / M / L) and a prioritised backlog.
  - **Recommended first batch** (items 1, 3, 4, 5): UTF-8 boundary-safe streaming decoder + ~~per-run timing line~~ + one jbang-runnable example + ~~a README system-properties table~~; ~1-2 days total, no JNI changes.
  - **DONE so far:**
    - README system-properties table (`e36f631`, with two cleanups in `3ae6c81` + `28dc9e6`).
    - Per-run timing line (`TimingsLogger` class + wire-in to `CompletionResponseParser` and `ChatResponseParser`; format mirrors what `llama.cpp` CLI prints — `prompt: N tok in X ms (Y tok/s) | gen: … | cache: N | draft: …`; dedicated SLF4J logger `net.ladenthin.llama.timings` so users can suppress it independently; 7 unit tests pin format + pipeline behaviour).
  - **DONE (2026-07-05):**
    - **UTF-8 boundary safety** — resolved natively rather than with the proposed Java-side decoder:
      the investigation showed the upstream server core already holds back incomplete UTF-8 at the
      end of the generated text (`server_context::process_token`), so streamed chunks can never
      split a codepoint. The *actual* gaps were in the JNI crossing: `json_to_jstring_impl` used
      `dump()` (throws `json::type_error 316` when the non-stream final content ends mid-codepoint
      at the token limit) + `NewStringUTF` (expects **Modified** UTF-8 — spec-invalid for
      supplementary-plane characters such as 4-byte emoji; Android CheckJNI aborts). Fixed by
      serialising via upstream `safe_json_to_str` (U+FFFD replacement) and building every payload
      string through the cached `String(byte[], "UTF-8")` constructor (`utf8_to_jstring_impl`);
      the `applyTemplate` return and the log-callback message take the same path. Pinned by new
      C++ unit tests (mock-JNI byte capture, emoji preservation, truncated-UTF-8 no-throw) and the
      model-backed `Utf8RoundTripIntegrationTest` (deterministic `applyTemplate` emoji/CJK
      round-trip + well-formedness of every streamed chunk).
    - **Runtime LoRA adapter control** (backlog item 8, `llama_adapter_lora_*` hot-apply) — typed
      `LlamaModel.getLoraAdapters()` / `setLoraAdapters(Map)` / `setLoraAdapter(int, float)` over
      new JNI methods posting `SERVER_TASK_TYPE_GET_LORA` / `SET_LORA` (the upstream
      `GET`/`POST /lora-adapters` contract; `value.LoraAdapter` + `json.LoraAdapterResponseParser`).
      Closes the `setLoraInitWithoutApply()` inconsistency (its Javadoc pointed at an endpoint the
      bindings could not reach). Tested model-free (parser + PIT-complete value tests, C++
      `ParseLoraRequest`/`ServerTaskResultGetLora` tests) and model-backed
      (`RuntimeLoraIntegrationTest`, adapter-less contract).
    - **Typed batch embeddings** — `LlamaModel.embed(Collection<String>)` → `List<float[]>` over the
      OAI array-input path of `handleEmbeddings` (`json.EmbeddingResponseParser`, index-ordered).
      Requested by upstream kherud users and unserved there.
  - **DONE (2026-07-05, second batch):**
    - **In-JVM router mode** (multi-model management through `NativeServer`): the upstream router
      spawns each model worker by re-executing its own binary — inside a JVM that is `java`, so
      embedded router workers could never start.
      `patches/0008-server-models-worker-cmd-override.patch` adds the `LLAMA_SERVER_WORKER_CMD`
      env override (whitespace-split, replaces only the worker-binary token), exposed as
      `NativeServer.setWorkerCommand(String...)`; each worker then runs as a fresh JVM executing
      the classic single-model `NativeServer`. Validated by `RouterModeIntegrationTest`
      (Linux CI: `--models-dir` listing → `POST /models/load` → worker-JVM spawn → proxied chat
      completion). This closes the old "Multi-model registry" follow-up for the native surface.
    - **In-JVM GGUF quantization** (backlog item 15): `LlamaQuantizer.quantize(in, out,
      QuantizationType[, threads, allowRequantize])` over `llama_model_quantize`
      (LLamaSharp `LLamaQuantizer` / llama-cpp-python precedent). PIT-complete
      `args.QuantizationType` (llama_ftype b9870 mapping) + `QuantizerIntegrationTest`
      (re-quantize the 135M draft model → load + complete; refusal without `allowRequantize`;
      missing-input error path).
  - **Remaining first-batch items:** jbang example.

### Android distribution: AAR + Kotlin-friendly API + sample app

- **DONE (2026-07-05): AAR + Kotlin façade shipped.** `net.ladenthin:llama-android` /
  `llama-android-opencl` (AARs from the standalone plain-Gradle build in `llama-android/` —
  hand-rolled AAR layout was chosen over `com.android.library` so no AGP/SDK is needed to
  build and the classes stay byte-identical to the Maven core jar) and the
  `net.ladenthin:llama-kotlin` reactor module (Flow adapters + suspend wrappers with
  CancellationToken-wired cancellation). CI: `package-android-aar` validates structure +
  16 KB alignment and runs an AGP R8 consumer build from mavenLocal
  (`.github/android-consumer-test/`); publish jobs ship snapshots/releases via Gradle.
  See CLAUDE.md "Android AAR + Kotlin façade". **Remaining from this section:** the sample
  app (`examples/android-sample/` — separate follow-up; covers real arm64 hardware +
  Adreno/OpenCL). **Multi-ABI + emulator CI: DONE (2026-07-05)** — `crosscompile-android-x86_64`
  (fail-loud, in the package/publish needs graphs; also feeds the default JAR via the
  `*-libraries` glob), the CPU AAR ships `jni/{arm64-v8a,x86_64}`, and the
  `test-android-emulator` job runs `connectedDebugAndroidTest` on a KVM x86_64 emulator
  (System.loadLibrary + GgufInspector + real inference on the cached draft model). **Promoted to a
  release gate (2026-07-05):** the job is in both publish `needs:` graphs after running flake-free
  through PR #298's validation cycle.

- **Publish a proper Android AAR alongside the existing JAR-with-resources packaging.** Today java-llama.cpp already cross-compiles the Android arm64 native lib in two flavours (CPU-only, bundled into the main JAR; OpenCL/Adreno under classifier `opencl-android-aarch64`), but both ship as plain Maven JARs that bury `libjllama.so` under `net/ladenthin/llama/Linux-Android/aarch64/`. Android/Gradle consumers expect an `.aar` with an `AndroidManifest.xml`, the native lib under `jni/arm64-v8a/`, and Maven coordinates like `net.ladenthin:llama-android:<version>@aar`. This is the format the [LLaMAndroid](https://github.com/Rattlyy/LLaMAndroid) integration referenced elsewhere in this file has to work around manually. Investigate using `com.android.library` via Gradle in a sibling module, or hand-rolling the AAR layout from the Maven build. Coordinate ABI coverage with any future armv7-a / x86_64 work so the AAR can declare multiple `jniLibs/<abi>/` entries when those land.

- **Provide a Kotlin-friendly façade + Android sample app.** The pure-Java `LlamaIterable` / `LlamaModel` API works on Android today (LLaMAndroid wraps it in a Kotlin `flow {}` block), but a small first-party Kotlin module — coroutine `Flow<LlamaOutput>` adapters, `suspend` variants of the blocking calls, idiomatic `use {}` resource handling — would lower the integration cost meaningfully and serve as the canonical reference for downstream consumers. Pair it with a minimal sample app (single `Activity`, model picker, streaming text view) under e.g. `examples/android-sample/` so the AAR has an exercised end-to-end path in CI. Treat LLaMAndroid as the prior-art baseline; reuse patterns that already work there.

### GraalVM Native Image evaluation

- **Evaluate GraalVM Native Image as an alternative distribution target.** Reference: [GraalVM Native Image](https://www.graalvm.org/latest/reference-manual/native-image/). The pure-Java sibling projects in the README's "Similar Projects" list (mukel's `llama3.java` / `gemma4.java` / `gptoss.java` / `qwen35.java` / `nemotron3.java`) demonstrate that single-jar, no-JNI Java inference is viable for individual model architectures. Native Image opens an orthogonal direction for THIS project: AOT-compile the Java layer + JNI bridge to a self-contained binary that bundles the libjllama.so (or per-OS equivalent) and starts in milliseconds without a JVM, which would make jllama usable in CLI tools, serverless functions, and short-lived processes where JVM startup is the dominant cost.

  **What to investigate before committing**:
  - **JNI-loading shape.** Native Image supports JNI but requires `--enable-native-access=ALL-UNNAMED` + reflection/JNI configuration files (`reflect-config.json`, `jni-config.json`, `resource-config.json`) describing every class/method/field reachable across the JNI boundary. The 17 native methods in `jllama.cpp` plus the JNI-side `FindClass` / `GetFieldID` / `GetMethodID` calls at `JNI_OnLoad` need to be mapped. The GraalVM tracing agent (`-agentlib:native-image-agent=config-output-dir=...`) can auto-generate the config during a representative test run, but the `LlamaLoader` JAR-extraction path needs at least one resource-config rule for `net/ladenthin/llama/{OS}/{ARCH}/lib*.so`.
  - **Native-library packaging.** The current `LlamaLoader` extracts the OS-specific `.so`/`.dll`/`.dylib` from the JAR to a tmp dir at first use. Native Image needs the same file at AOT-execution time, so either (a) ship the native lib alongside the produced binary as a sidecar file and adjust `LlamaLoader` to find it on the same directory, or (b) embed the native lib as a resource and keep the existing extract-to-tmpdir flow (which Native Image supports via `resource-config.json`).
  - **CUDA / Metal / OpenCL backend selection.** Today the choice between CPU-only / `cuda13-linux-x86-64` / `opencl-android-aarch64` JARs is at Maven-classifier time. Native Image would need either one binary per backend (multiplying the release matrix) or a runtime selector inside `LlamaLoader` that picks among bundled backend libs. The latter is a bigger refactor.
  - **Startup-time benchmark to justify the work.** Measure cold-start of a current java-llama.cpp `LlamaModel(new ModelParameters().setModel("...").setNPredict(1))` invocation: how much is JVM startup + class load vs JNI load + model parse + tokenize + 1 token? If JVM startup is < 10 % of cold-start, Native Image yields little. If JVM startup is > 50 %, it's a clear win for CLI / serverless use cases.
  - **Maintenance cost.** Native Image adds a second build matrix (per OS × per backend × per JDK) and a new failure surface (Native Image config drift when a llama.cpp version bump adds new JNI-reachable types). Should ship only with a CI job that exercises the Native Image build on at least one OS, otherwise the config files will rot silently.

  **Out of scope until evidence supports it**: actually implementing any of the above. This entry exists so that when someone asks "can I ship java-llama.cpp as a single 30 MB binary?" the answer points to a concrete investigation plan rather than restarting from zero.

## Open — cross-cutting (slice for this repo)

- **jqwik pin policy** — see [`../workspace/policies/jqwik-prompt-injection.md`](../workspace/policies/jqwik-prompt-injection.md). `jqwik.version ≤ 1.9.3` is mandatory.

- **`@VisibleForTesting` audit.** No usages currently. Walk the production tree for package-private/protected methods or fields that exist purely so tests can reach them, and either annotate (`com.google.common.annotations.VisibleForTesting`) or move into the test source tree.

- **Null-safety refinement.** JSpecify + NullAway are now enforced at compile time in **strict JSpecify mode** with the extra options `CheckOptionalEmptiness`, `AcknowledgeRestrictiveAnnotations`, `AcknowledgeAndroidRecent`, `AssertsEnabled` (see `pom.xml`); `@NullMarked` on the three packages via `package-info.java`; JDK module exports in `.mvn/jvm.config`. The legacy `org.jetbrains.annotations` dep has been removed; all nullability annotations are JSpecify. Public-API methods that may legitimately have no value use `Optional<T>` rather than `@Nullable T` (`ChatResponse.getFirstMessage`, `ChatMessage.getParts`, `ChatRequest.buildToolsJson`). Open follow-up: review remaining unannotated public API surfaces for places where `@Nullable` would be more precise than the implicit non-null default.

- **SpotBugs `effort=Max` + `threshold=Low`** — **DONE (already enabled in `pom.xml`)**, with fb-contrib +
  findsecbugs, bound to `verify`. The legacy "flip the pom / ~65 findings" note is stale: only a handful
  of unexcluded findings remain at any time, and `spotbugs:check` is kept green. Most recent pass fixed the
  6 introduced by the audit Tier-1–3 fixes — `withScalar` uses a single `instanceof Number` (no
  `ITC_INHERITANCE_TYPE_CHECKING`); `ChatMessage.getToolCalls` returns a fresh unmodifiable view (no
  `EI_EXPOSE_REP`); the `LlamaModel` batch methods' deliberate re-throw and the `ChatMessage` public
  constructor's `List` param carry narrow `<Match>` rationale suppressions.
  **Note:** `spotbugs:check` is bound to the `verify` phase, which the model-backed CI test jobs
  (`mvn test` / `mvn package`) do not reach — run `mvn verify` (or a dedicated job) to gate it in CI.

- **Drop the project-wide `OPM_OVERLY_PERMISSIVE_METHOD` suppression in
  `spotbugs-exclude.xml`** once the package-architecture refactor lands
  (see [`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md)
  under "Affects BAF + jllama (multi-package repos)"). The single-root
  package today makes every "method called only by same-package callers
  → could be package-private" finding correct-but-unstable; once layers
  split, cross-layer calls will need public. Snapshot at suppression
  (`07109cc`): 25 sites. The same rule is suppressed in BAF
  (`52c8c95`) for identical reasons.

- **Additional ArchUnit rules to consider** — the full **`layeredArchitecture()`** rule and a **per-module banned-import** rule (`jacksonBannedFromContractsAndLoader` — Jackson kept out of `args`/`callback`/`exception`/`loader`) are now DONE. Still open: more per-module banned-imports if useful, public-API-surface constraints (no public mutable static state, etc.). Partial progress: `7b6667d` covers the "no public field that is not final" sub-rule.

- **Cross-repo code-quality TODOs** — see [`../workspace/policies/code-quality-todos.md`](../workspace/policies/code-quality-todos.md) for the canonical `@VisibleForTesting` design-fit review, package hierarchy review, and class/method naming review. This repo has no `@VisibleForTesting` usages today; package and naming reviews remain open.

## Done (kept for history)

### b9739 upgrade + PR #248 (Windows Ninja, native aarch64, patches mechanism)

- **llama.cpp b9682 → b9739** (#247, merged) + build fixes: `server-schema.cpp` added to the
  `jllama_test` sources (b9739 link fix, `38be6db`); `test_server.cpp` `ParamsFromJsonCmpl`
  expectations updated to b9739 schema behavior (`aaba886`).
- **Windows Ninja artifact** — `ninja-windows` classifier JAR built with Ninja Multi-Config + sccache,
  shipped alongside the permanent MSVC default; both build + Java-test jobs green (`e113ed3`,
  `48f0863`). (See the open section above for the design rationale; verification is done.)
- **Linux aarch64 → native `ubuntu-24.04-arm` build** (`ed9ecbb`). The dockcross `linux-arm64-lts`
  image (GCC 8.5 / glibc 2.17) could no longer compile b9739's C++17 CTAD-in-`new`; now builds natively
  with GCC 14 (mirroring upstream), runs `ctest` on real ARM (446 tests green), and warms sccache
  (99.66% hits). Trade-off: glibc floor 2.17 → ~2.39 (same envelope as upstream's ARM binaries);
  documented in the README classifier table. `build.sh` sccache auto-fetch generalized to aarch64.
- **Generic `patches/` mechanism** — drop `*.patch`/`*.diff` in repo-root `patches/`, applied to the
  FetchContent'd llama.cpp source by `cmake/apply-llama-patches.cmake` via the llama.cpp
  `PATCH_COMMAND` (cross-platform, idempotent, fail-loud). Covers every C++ build from one place.
  First patch fixes the Windows JNI arg-parse regression (`1d875b1` → deterministic form `f651b53`).
  REUSE annotated via `patches/**` glob (`0cffac1`).
- **CUDA sccache verified** — the `manylinux_2_28 (CUDA)` job caches all gcc C/C++ TUs (247/248 hits,
  99.60%); the nvcc `.cu` kernels remain uncached (sccache limitation), and `CUDA_FAST_BUILD` keeps
  PR/validation runs single-arch. (Doc/observation; no code change.)

### Layered package restructure (flat root package → layered hierarchy)

The flat `net.ladenthin.llama` root package was split (via `git mv`, history
preserved) into layered packages so boundaries align with the layers, enforced
by a new `layeredArchitecture()` ArchUnit rule (Api → Loader → Marshalling →
Foundation):

- **Foundation**: `value` (18 DTOs: ChatMessage, ContentPart, Pair, LlamaOutput,
  …), `callback` (CancellationToken, LoadProgressCallback, ToolHandler),
  `exception` (LlamaException, ModelUnavailableException), `args` (existing leaf).
- **Marshalling**: `json` (response parsers + `TimingsLogger`, its only consumer),
  `parameters` (Inference/Model/Json/Cli parameters + `ParameterJsonSerializer` +
  `ChatRequest`).
- **Loader** (internal, NOT exported): `loader` (LlamaLoader, OSInfo,
  ProcessRunner, NativeLibraryPermissionSetter, Java8CompatibilityHelper,
  OfflineModelGuard, LlamaSystemProperties).
- **Api** (root): LlamaModel, Session, LlamaIterable, LlamaIterator.

Cycle-breaking moves: `TimingsLogger` root→`json`, `ParameterJsonSerializer`
`json`→`parameters`, `ChatRequest` root→`parameters` (it carries an
`InferenceParameters` customizer). Test classes mirrored into their subjects'
packages; cross-layer members promoted to `public`. Cross-package Javadoc
`{@link}` references fully-qualified (palantir's `removeUnusedImports` strips
javadoc-only imports). `module-info` exports the new public-API packages and
keeps `loader` internal. All 11 ArchUnit rules green; `javadoc:jar` clean.

**Breaking change**: public-API FQNs changed (e.g. `net.ladenthin.llama.ChatMessage`
→ `net.ladenthin.llama.value.ChatMessage`) — ship under a major version bump.

- **Reactive `LlamaPublisher` removed in favour of consumer-side adapters.**
  The hand-rolled `LlamaPublisher` + `LlamaModel.streamPublisher` /
  `streamChatPublisher` (shipped in PR #188 as §2.3 of the Kotlin SDK
  feature comparison) had zero non-test callers. `LlamaIterable` is
  already `Iterable<LlamaOutput> & AutoCloseable`, and every mainstream
  reactive library wraps it in a few lines via its own resource-management
  primitive (`Flux.using`, `Flowable.using`, Kotlin `use {}`). The real-world
  Android consumer [LLaMAndroid](https://github.com/Rattlyy/LLaMAndroid)
  already uses `LlamaIterable` inside a Kotlin `flow {}` block — bypassing
  the publisher entirely. README "Reactive integration" section documents
  the Reactor / RxJava 3 / Kotlin Flow / Akka patterns; correctness is
  pinned end-to-end by a new `ReactorIntegrationTest` using
  test-scope `reactor-core` (zero runtime deps added; `org.reactivestreams`
  runtime dep dropped). Cleared 6 fb-contrib Max+Low findings on
  `LlamaPublisher$LlamaSubscription` as a side effect.

- **Error Prone bug-pattern promotions to `ERROR`** — `855f447` (12 patterns promoted; `-Xlint:all` enabled).
- **`javac -Werror` + `-Xlint:all,-serial,-options,-classfile,-processing`** — `3e2efbb`. ~20 EP warnings addressed first (EqualsGetClass on `Pair` via instanceof; MissingOverride on `PoolingType` / `RopeScalingType`; JdkObsolete `LinkedList` → `ArrayList` in `LlamaLoader`; StringSplitter inline-suppressed; 3× StringCaseLocaleUsage `Locale.ROOT` in `OSInfo`; EmptyCatch in `OSInfo.isAlpineLinux`; FutureReturnValueIgnored in `LlamaModel.completeAsync`; Finalize on `LlamaModel.finalize`; MixedMutabilityReturnType in 4 parser methods; EnumOrdinal in `InferenceParameters.setMiroStat`; EscapedEntity in `InferenceParameters` javadoc; 4× TypeParameterUnusedInFormals; AnnotateFormatMethod on `Java8CompatibilityHelper.formatted`; SafeVarargs + varargs on `Java8CompatibilityHelper.listOf`).
- **`-parameters` javac arg** — `4350cf2`.
- **`--release N`** — `4350cf2` (`<release>8</release>`).
- **Mutation-testing threshold enforcement (PIT)** — `62f8a00` + `bb93a8f` (docs) + `3bfa51f` (README badge). Runs every CI build with `<mutationThreshold>100</mutationThreshold>`. **Scope expanded 2026-06-07** from the original single `Pair` target (which was stale after the restructure — `llama.Pair`→`value.Pair` matched nothing) to `value.*` + `exception.*` + `args.*` + `json.TimingsLogger` = 27 classes / 163 mutations, all killed. Still open (optional): `json.ChatResponseParser` / `CompletionResponseParser` private-helper survivors (`RerankResponseParser` is excluded — equivalent empty-list mutant).
- **Checker Framework as a second static-nullness pass** — `c63870b`. The original
  `@PolyNull` on `JsonParameters.toJsonString` was simplified to plain `@Nullable`
  (the only `@PolyNull` site in production; eliminated in a later cleanup).
  Native-method constructor calls in `LlamaModel` carry
  `@SuppressWarnings("method.invocation")` (Checker's `@UnderInitialization`
  cannot see that the native callee does not dereference `this`); `Pair.equals`
  and `Usage.equals` declare `@Nullable Object`; `LlamaSystemProperties` getters
  return `@Nullable String`; `getPackage()` and resource-stream null derefs are
  guarded.
- **JPMS `module-info.java` with module-level `@NullMarked`** — `0fd066a` + `9528e79`. The module `net.ladenthin.llama` exports the three hand-written public packages (`net.ladenthin.llama`, `.args`, `.json`). Two-execution `maven-compiler-plugin` pattern; module-level `@NullMarked` lives on the module descriptor.
- **Banned-API enforcement** — Maven Enforcer (`8baae0c`), ArchUnit `System.exit` / `new Random` / `Thread.sleep` (`329d764`), `sun.*` / `com.sun.*` / `jdk.internal.*` (`e6069da`).
- **ArchUnit public-fields-final** — `7b6667d`.
- **LogCaptor smoke test** — `LoggingSmokeTest` (`3cedc6e`).
- **Offline / air-gapped model loading** — `ModelFlag.OFFLINE` + `ModelParameters.setOffline(boolean)` + `hasFlag` helper + public `ModelUnavailableException` (extends now-public `LlamaException`) + deterministic pre-check `OfflineModelGuard`. Unit tests in `LlamaModelOfflineTest`. No JNI rebuild required. *(Originally shipped as `SKIP_DOWNLOAD`/`setSkipDownload` over a parse-failure heuristic; reworked when llama.cpp b9803 removed `common_params::skip_download` and `common_skip_download_exception` — `--skip-download` was never a registered upstream arg, so it never actually skipped a download. `--offline` is the real upstream flag with the intended load-from-cache semantics.)*
- **`LlamaSystemProperties` registry cleanup** — `getLibName()` deleted (`6bb63e1` upstream forensic trace); `OSInfo.getArchName()` now routes through `LlamaSystemProperties.getOsinfoArchitecture()` (`3ae6c81`).
- **Abstract the Java and test writing guidelines to a workspace-level shared layer.** Workspace version chain at [`../workspace/guides/src/CODE_WRITING_GUIDE-8.md`](../workspace/guides/src/CODE_WRITING_GUIDE-8.md) and [`../workspace/guides/test/TEST_WRITING_GUIDE-8.md`](../workspace/guides/test/TEST_WRITING_GUIDE-8.md); canonical TDD skill at [`../workspace/.claude/skills/java-tdd-guide/SKILL.md`](../workspace/.claude/skills/java-tdd-guide/SKILL.md).
- **Standardised CLAUDE.md template** — [`../workspace/templates/CLAUDE.md.template`](../workspace/templates/CLAUDE.md.template).
