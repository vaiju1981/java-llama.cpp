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

### ⚠️ OpenAI server: TWO implementations to consolidate ("best of both")

Two independent, Claude-generated OpenAI-compatible servers now coexist in
`net.ladenthin.llama.server` after PR #240 was merged on top of the NanoHTTPD server that landed
via #242. **This is a temporary state**; one unified implementation must be chosen. Until then both
compile and are tested side by side.

| | **Option A — `OpenAiCompatServer`** (from PR #240) | **Option B — `LlamaServer`** (from #242) |
|---|---|---|
| HTTP layer | JDK `com.sun.net.httpserver` (the supported `jdk.httpserver` module — **no dependency**) | NanoHTTPD (`<optional>` dep, bundled only in fat jar) |
| Streaming | **Yes** — SSE with `delta.tool_calls`, heartbeats during prefill | No — blocking, full JSON per request |
| Routes | `POST /v1/chat/completions`, `GET /v1/models` | `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `GET /v1/models`, `GET /health` |
| Entry point | CLI launcher + embeddable; `OpenAiServerConfig` builder; optional bearer auth; binds `127.0.0.1` | fat-jar `Main-Class`; `LlamaServerArgs` CLI (`--host/--port/--ctx-size/--threads/…`) |
| Native path | `requestChatCompletionStream` / `receiveChatCompletionChunk` (+ `wrap_stream_chunk` C++ helper) | `LlamaModel.handle*` (blocking) |
| Tests | mapper/SSE/parser unit tests + model-free HTTP test over a socket (`ChatBackend` seam) | `OaiRouterTest`, `LlamaServerArgsTest`, `OaiHttpServerIntegrationTest` |

**Important cross-insight:** Option B's own follow-up TODO below ("OpenAI-compatible server: token
streaming (SSE) + Java-8 HTTP layer") lists SSE as *the main functional gap* and says to **avoid**
`com.sun.net.httpserver` because it is "ArchUnit-banned". Option A **already implements that SSE
streaming** with `com.sun.net.httpserver`, and the ban was lifted correctly: `com.sun.net.httpserver`
is a *supported, exported* JDK API (the `jdk.httpserver` module), not an internal `com.sun..` package —
the `noInternalJdkImports` ArchUnit rule now carries an explicit exception for it. So the premise that
blocked the JDK approach on Option B's side does not hold.

**Consolidation task (separate session — a kickoff prompt accompanies this change):** go through both
implementations, take the best of each, settle on ONE server, delete the other, reconcile the
dependency (`pom.xml` NanoHTTPD + assembly), the ArchUnit `layeredArchitecture` `Server` layer, the
`spotbugs-exclude.xml` entries, `package-info.java`, the README "OpenAI-compatible HTTP server"
section, and this TODO (including the now-partly-moot SSE section below).

### OpenAI-compatible HTTP endpoint (shipped; follow-ups open)

`net.ladenthin.llama.server.OpenAiCompatServer` exposes `POST /v1/chat/completions` (streaming via
SSE + non-streaming) and `GET /v1/models` over the JDK's built-in `com.sun.net.httpserver` (no new
dependency), so editors that speak the OpenAI protocol (e.g. VS Code Copilot "Custom Endpoint") can
drive a local model. Streaming uses the native OAI chunk path (`requestChatCompletionStream` /
`receiveChatCompletionChunk`), preserving `delta.tool_calls` for agent mode. Follow-ups, deferred
until requested:

- **Multi-model registry.** Only one model id is advertised/served today; support several models
  chosen by the request `model` field (and listed in `/v1/models`).
- **`stream_options.include_usage` passthrough** so the final streamed `usage` chunk is emitted
  (needs a generic raw-param passthrough on `InferenceParameters`, or explicit mapping).
- **Additional `apiType`s.** VS Code "Custom Endpoint" also offers Anthropic `messages` and OpenAI
  `responses`; only `chat-completions` is implemented. Also consider `/v1/completions` and
  `/v1/embeddings` routes.
- **Gemma 4 tool-calling validation.** Confirm the pinned llama.cpp (`b9682`) includes the Gemma 4
  tool-call parser fixes (landed upstream ~Apr 2026); if not, bump per the upgrade procedure so
  streamed/blocking `tool_calls` come through for Gemma 4 GGUFs.

### llama.cpp upstream feature exposure (queued, deferred by policy)

These are JNI plumbing items for upstream API additions. Policy: add only after a real user request — they are mostly relevant to specific model families or specialized workflows.

- **Expose `--spec-draft-backend-sampling` toggle via `ModelParameters.setSpecDraftBackendSampling(boolean)`.** Added in b9437 (env `LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING`). Backend sampling for the speculative draft is enabled by default upstream but auto-disabled on `LLAMA_SPLIT_MODE_TENSOR` setups; an explicit Java-side setter lets callers force-disable it for benchmarking or for backends with sampler bugs. Speculative-decoding power users.

- **Expose runtime reasoning control via `InferenceParameters.setReasoningControl(boolean)` + `LlamaModel.endReasoning(...)`.** Added in b9444–b9490: new `common_params_sampling::reasoning_control` flag arms the budget sampler so reasoning can be ended at runtime, and new `common_sampler_reasoning_budget_force(common_sampler *)` triggers the end-of-thinking token injection on the next sample. Upstream also adds a `POST /v1/chat/completions/control` server endpoint accepting `{"id": "...", "action": "reasoning_end"}`. Java mapping would be: (a) `InferenceParameters.setReasoningControl(boolean)` arms the sampler on the inference run, (b) a new `LlamaModel.endReasoning(int slotId)` (or per-streaming-task-id) JNI method calls the upstream `common_sampler_reasoning_budget_force` against the slot's sampler. Useful for interactive UIs that want a "skip thinking and answer now" button. Relevant only for reasoning-trained models (DeepSeek-R1, Qwen3-Thinking, GPT-OSS-Reasoner, etc.).

- **Expose `llama_context_params::n_outputs_max` via `ModelParameters.setMaxOutputs(int)`.** Added in b9444–b9490 (default `-1` = derived from `n_batch`). Caps the number of output slots allocated per context; relevant for memory-constrained setups that always run with `logits_all=false` and want to prevent over-allocation when `n_batch` is large. Trivial JNI plumbing (one `cparams` field passthrough); add when a user reports OOM on context creation tied to output slot pre-allocation.

- **Expose Multi-Token Prediction toggle via `ModelParameters.setMtp(boolean)`.** Existed since the Qwen3.5 MTP work; b9444–b9490 extends it to Step-3.5. CLI flags `--mtp`/`--no-mtp` (env `LLAMA_ARG_MTP`) control whether the draft head runs alongside the main model for accelerated decoding. Java setter would route to `common_params_speculative::type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP`. Relevant only for MTP-trained models.

- **Expose `llama_vocab::get_suppress_tokens()` via `LlamaModel.getSuppressTokens()`.** Added in b9490–b9495 alongside the new `tokenizer.ggml.suppress_tokens` GGUF key and the `LLM_KV_TOKENIZER_SUPPRESS_TOKENS` constant. When a GGUF declares this array, upstream stores it on `llama_vocab::impl::suppress_tokens` and exposes it via the new `llama_vocab::get_suppress_tokens()` accessor. The bias is **applied automatically** inside the model forward graph — the Gemma4 Unified graph (`src/models/gemma4.cpp`) reads the list and adds a `-INFINITY` logit bias to those token IDs via a new `llm_graph_input_logits_bias` input so the model cannot emit them (used to block `<image|>` / `<audio|>` placeholders). A Java mirror would be `public int[] getSuppressTokens()` on `LlamaModel`: a read-only inspector returning the suppression list for debugging or for callers running their own sampling who want to replicate the same bias. Value is low (the bias is auto-applied, Java callers cannot change it; java-llama.cpp does not expose custom logit-bias hooks at this level); cost is trivial (one JNI passthrough + a `getSuppressTokens()` Java method).

### Feature backlog from similar projects

- **Feature backlog from similar projects.** See [`docs/feature-investigation-similar-projects.md`](docs/feature-investigation-similar-projects.md) for the consolidated investigation across the 5 pure-Java sibling runtimes ([llama3.java](https://github.com/mukel/llama3.java), [gemma4.java](https://github.com/mukel/gemma4.java), [gptoss.java](https://github.com/mukel/gptoss.java), [qwen35.java](https://github.com/mukel/qwen35.java), [nemotron3.java](https://github.com/mukel/nemotron3.java)) plus the dormant alternative JNI binding [llamacpp4j](https://github.com/sebicom/llamacpp4j). The doc captures 18 candidate items grouped into cross-cutting themes (UTF-8 streaming boundary safety, thinking-channel router, operator timing line, jbang single-file example, README system-properties table, etc.) and per-repo unique findings (Harmony channel decoder, Qwen empty-`<think>` injection, llama_state_* save/load, llama_adapter_lora_* hot-apply, etc.), each with effort sizing (XS / S / M / L) and a prioritised backlog.
  - **Recommended first batch** (items 1, 3, 4, 5): UTF-8 boundary-safe streaming decoder + ~~per-run timing line~~ + one jbang-runnable example + ~~a README system-properties table~~; ~1-2 days total, no JNI changes.
  - **DONE so far:**
    - README system-properties table (`e36f631`, with two cleanups in `3ae6c81` + `28dc9e6`).
    - Per-run timing line (`TimingsLogger` class + wire-in to `CompletionResponseParser` and `ChatResponseParser`; format mirrors what `llama.cpp` CLI prints — `prompt: N tok in X ms (Y tok/s) | gen: … | cache: N | draft: …`; dedicated SLF4J logger `net.ladenthin.llama.timings` so users can suppress it independently; 7 unit tests pin format + pipeline behaviour).
  - **Remaining first-batch items:** UTF-8 boundary-safe streaming decoder + jbang example.

### Android distribution: AAR + Kotlin-friendly API + sample app

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

### OpenAI-compatible server: token streaming (SSE) + Java-8 HTTP layer

The `net.ladenthin.llama.server.LlamaServer` MVP is **non-streaming**: every request calls
the blocking `LlamaModel.handle*` method and returns the full JSON response in one shot. A
client that sends `"stream": true` still receives a single response, not the incremental
`text/event-stream` (SSE) `data: {chunk}\n\n` events the OpenAI API emits for streaming
chat/completions. This is the main functional gap of the server today.

The token source already exists — `LlamaModel.generateChat(InferenceParameters)` /
`generate(...)` yield tokens incrementally through a Java `Iterator` (`LlamaIterable`). What
is missing is an HTTP layer that emits SSE.

**Find a Java-8-compatible HTTP layer with good SSE support (alternative to Javalin), or
implement SSE on NanoHTTPD.** Javalin has a first-class `ctx.sse(...)` API but is **not
usable here**: Javalin 5 requires Java 11 and Javalin 6 requires Java 17, while this repo
targets Java 8; Javalin 4 (the last Java-8 release) is EOL. Options, in rough order of
preference:
- **Implement SSE on the existing NanoHTTPD** via `NanoHTTPD.newChunkedResponse(status,
  "text/event-stream", InputStream)`, bridging a `LlamaIterable` to an `InputStream` that
  writes `data: {chunk}\n\n` frames. No new dependency, stays Java-8 clean; likely the right
  answer. Cost: the iterator→SSE bridge plus closing the `LlamaIterable` on client
  disconnect.
- **Undertow** — Java-8 compatible, has a server-sent-events handler, but a heavier
  dependency tree.
- **Spark Java** (Jetty 9) — Java-8 compatible; SSE support is limited/manual.
- Avoid: Javalin 5/6 (Java 11/17), Javalin 4 (EOL), and the JDK `com.sun.net.httpserver`
  (ArchUnit-banned `com.sun..`).

Scope when implemented: honour `"stream": true` on `POST /v1/chat/completions` and
`POST /v1/completions`, emit OpenAI-style SSE chunks terminated by `data: [DONE]`, close the
underlying `LlamaIterable` on disconnect, and keep the non-streaming path as the default. Add
a model-free routing test plus a real-socket SSE integration test (mirroring
`OaiHttpServerIntegrationTest`).

## Open — cross-cutting (slice for this repo)

- **jqwik pin policy** — see [`../workspace/policies/jqwik-prompt-injection.md`](../workspace/policies/jqwik-prompt-injection.md). `jqwik.version ≤ 1.9.3` is mandatory.

- **`@VisibleForTesting` audit.** No usages currently. Walk the production tree for package-private/protected methods or fields that exist purely so tests can reach them, and either annotate (`com.google.common.annotations.VisibleForTesting`) or move into the test source tree.

- **Null-safety refinement.** JSpecify + NullAway are now enforced at compile time in **strict JSpecify mode** with the extra options `CheckOptionalEmptiness`, `AcknowledgeRestrictiveAnnotations`, `AcknowledgeAndroidRecent`, `AssertsEnabled` (see `pom.xml`); `@NullMarked` on the three packages via `package-info.java`; JDK module exports in `.mvn/jvm.config`. The legacy `org.jetbrains.annotations` dep has been removed; all nullability annotations are JSpecify. Public-API methods that may legitimately have no value use `Optional<T>` rather than `@Nullable T` (`ChatResponse.getFirstMessage`, `ChatMessage.getParts`, `ChatRequest.buildToolsJson`). Open follow-up: review remaining unannotated public API surfaces for places where `@Nullable` would be more precise than the implicit non-null default.

- **SpotBugs `effort=Max` + `threshold=Low`** — currently default effort/threshold. Raising both surfaces ~65 remaining findings (was 90; the cross-repo `OPM_OVERLY_PERMISSIVE_METHOD` suppression in `07109cc` silenced 25 of them pending the package refactor — see below). Top remaining patterns: `DRE_DECLARED_RUNTIME_EXCEPTION` 20, `WEM_WEAK_EXCEPTION_MESSAGING` 14. The BAF/sb/plugin playbook applies: flip pom, run `spotbugs:check`, fix at source where reasonable + narrow `<Match>` with rationale for structural false positives. Cross-cutting (tracked in [`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md)).

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
  SkipDownloadFailureTranslator, LlamaSystemProperties).
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
- **Expose `common_params::skip_download`** — `ModelFlag.SKIP_DOWNLOAD` + `ModelParameters.setSkipDownload(boolean)` + `hasFlag` helper + new public `ModelUnavailableException` (extends now-public `LlamaException`) + Java-side heuristic translator. 7 unit tests in `LlamaModelSkipDownloadTest`. No JNI rebuild required.
- **`LlamaSystemProperties` registry cleanup** — `getLibName()` deleted (`6bb63e1` upstream forensic trace); `OSInfo.getArchName()` now routes through `LlamaSystemProperties.getOsinfoArchitecture()` (`3ae6c81`).
- **Abstract the Java and test writing guidelines to a workspace-level shared layer.** Workspace version chain at [`../workspace/guides/src/CODE_WRITING_GUIDE-8.md`](../workspace/guides/src/CODE_WRITING_GUIDE-8.md) and [`../workspace/guides/test/TEST_WRITING_GUIDE-8.md`](../workspace/guides/test/TEST_WRITING_GUIDE-8.md); canonical TDD skill at [`../workspace/.claude/skills/java-tdd-guide/SKILL.md`](../workspace/.claude/skills/java-tdd-guide/SKILL.md).
- **Standardised CLAUDE.md template** — [`../workspace/templates/CLAUDE.md.template`](../workspace/templates/CLAUDE.md.template).
