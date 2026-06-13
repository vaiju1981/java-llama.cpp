<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# Feature Investigation — ideas from pure-Java sibling runtimes and `llamacpp4j`

Comparison sources (all surveyed in one pass for this document):

| Repo | Shape | License | Survey notes |
|------|-------|---------|--------------|
| [mukel/llama3.java](https://github.com/mukel/llama3.java) | Pure Java, single-file (~3.4k LOC), Vector API + GraalVM Native Image | MIT | Llama 3 / 3.1 / 3.2 |
| [mukel/gemma4.java](https://github.com/mukel/gemma4.java) | Pure Java, single-file (~3.9k LOC) | Apache 2.0 | Gemma 4 + earlier Gemma 2/3 |
| [mukel/gptoss.java](https://github.com/mukel/gptoss.java) | Pure Java, single-file | Apache 2.0 | OpenAI GPT-OSS (Harmony chat format) |
| [mukel/qwen35.java](https://github.com/mukel/qwen35.java) | Pure Java, single-file | Apache 2.0 | Qwen 3.5 dense + MoE |
| [mukel/nemotron3.java](https://github.com/mukel/nemotron3.java) | Pure Java, single-file | Apache 2.0 | NVIDIA Nemotron-3 (dense + MoE + recurrent SSM) |
| [sebicom/llamacpp4j](https://github.com/sebicom/llamacpp4j) | Alternative JNI binding (SWIG-generated facade over `llama.h`) | unspecified | **Dormant** — 1 commit (2023-07-04), pre-GGUF (llama.cpp build 491), no LICENSE, no tests, no CI |

The 5 `mukel` projects are written by the same author (Alfonso² Peterssen), share a single-file template, and re-implement GGUF parsing + tensor kernels in pure Java. They are NOT direct competitors to `java-llama.cpp` (which delegates inference to llama.cpp via JNI); they are interesting because they have **better operator-facing ergonomics** at the CLI and example layers.

`llamacpp4j` is the only other Java-side JNI binding to llama.cpp; the survey looked specifically for API-shape ideas and capabilities not currently exposed here.

Effort sizing (mirrors [`feature-investigation-llama-stack-client-kotlin.md`](feature-investigation-llama-stack-client-kotlin.md)):

| Size | Calendar effort (1 engineer) | Description |
|------|------------------------------|-------------|
| XS   | < 0.5 day                    | Trivial Java-side change, no JNI |
| S    | 0.5 – 2 days                 | Java surface + minor JNI/JSON wiring |
| M    | 2 – 5 days                   | New JNI methods, native plumbing, tests |
| L    | 1 – 2 weeks                  | New native subsystem or large API surface |

---

## 1. What this project already covers

The following are confirmed present in `java-llama.cpp` as of this survey — flagged so we do not re-investigate them:

| Capability | Status |
|---|---|
| `setSkipDownload(boolean)` + typed `ModelUnavailableException` | ✅ (commit `37754d4`) |
| Reasoning-format toggle, reasoning-budget tokens | ✅ (`InferenceParameters#setReasoningFormat` etc.) |
| Tool calls + custom chat templates | ✅ |
| Speculative draft model | ✅ |
| Multimodal vision (mmproj) | ✅ |
| Infill (fill-in-the-middle) | ✅ |
| Streaming via `LlamaIterator` / Reactive Streams `Publisher` | ✅ |
| `CompletableFuture` async + `CancellationToken` | ✅ |
| `LoadProgressCallback` model-load progress | ✅ |

---

## 2. Cross-cutting themes — universal across the 5 `mukel` projects

These ideas appear in every (or nearly every) `mukel` runtime; portability across reasoning-model families makes them the **highest-leverage** items.

### 2.1 Streaming UTF-8 decoder for multi-byte boundary safety  *(S, medium-high priority)*

Sources: `qwen35.java` (`StreamingDecoder`, L2929–2987), `nemotron3.java`, `gemma4.java`.

GGUF byte-fallback tokenisation can split a single Unicode codepoint across two consecutive token pieces. `LlamaIterator` callers today can receive a `LlamaOutput.text` value containing a partial UTF-8 sequence and either render mojibake (CJK, emoji) or hand-roll their own buffering. The `mukel` runtimes wrap the token stream in a small decoder that holds back trailing bytes until a complete codepoint is available, then flushes.

- **Why**: silent correctness bug for non-ASCII users; ~50-LOC fix.
- **Shape**: `Utf8BoundaryStreamingDecoder` helper in the Java layer (no JNI change); optional `setUtf8BoundarySafe(true)` opt-in on `InferenceParameters`, or always-on inside `LlamaIterator`.
- **Test**: use any of the existing CJK / emoji prompts; assert no partial codepoint ever crosses the iterator boundary.

### 2.2 Tri-state thinking-channel router for reasoning models  *(S, medium priority)*

Sources: `gemma4.java`, `gptoss.java` (Harmony channels), `qwen35.java`, `nemotron3.java`.

A `--think off|on|inline` flag with three semantics: **`off`** strips reasoning tokens from the visible stream (and from chat history), **`on`** (default) routes them to a separate sink (e.g. stderr in CLI examples), **`inline`** interleaves them in the main output. Pairs cleanly with this project's existing `setReasoningFormat`/`setReasoningBudgetTokens`.

- **Why**: every reasoning model in this project's test matrix (Qwen3-0.6B, plus any GPT-OSS / Gemma / Nemotron load) exposes thought tokens, but operators currently hand-roll the routing.
- **Shape**: helper class `ThinkingChannelRouter` (or analogous) that consumes a `LlamaIterator` and produces two streams (visible / reasoning), plus an enum knob on `InferenceParameters`.
- **gptoss specifically**: needs a Harmony-channel state machine that recognises `<|start|>`, `<|channel|>`, `<|message|>`, `<|end|>` and exposes `analysis` / `commentary` / `final` channels separately. Worth shipping as a separate `HarmonyChannelDecoder` if GPT-OSS users materialise. *(M for the Harmony variant; S for the generic `<think>` variant.)*

### 2.3 Interactive chat REPL with slash commands  *(XS, low-medium priority)*

Sources: `llama3.java`, `gemma4.java`, `gptoss.java`, `qwen35.java`, `nemotron3.java`.

`/quit`, `/exit`, `/context` (the latter prints `used / max / remaining` tokens for the current chat session). Users currently Ctrl-C out of `ChatExample`.

- **Shape**: a `ChatRepl` example under `src/test/java/examples/`. No new production API surface — it composes existing `LlamaModel` calls.
- **Effort**: 1 new file, ~150 LOC.

### 2.4 ANSI colour auto-detection honouring `NO_COLOR` + `TERM=dumb`  *(XS, low priority)*

Sources: `gemma4.java`, `gptoss.java`, `qwen35.java`, `nemotron3.java`.

Tri-state `--color on|off|auto` helper that honours the [`NO_COLOR`](https://no-color.org) informal standard, detects `TERM=dumb`, and falls back to no-colour when `System.console()` is `null`. ~15 LOC; useful in every example CLI that prints reasoning tokens or perf summaries in a different style.

### 2.5 Operator-grade timing line on stderr  *(XS, medium priority)*

Sources: `qwen35.java`, `nemotron3.java`.

After every generation: a one-line `prompt: X tok/s (P tokens) | generation: Y tok/s (G tokens) | context: U/M` summary to stderr. `LlamaModel.getTimings()` already has all the inputs; no example formats them.

### 2.6 `AutoCloseable Timer.log("label")` idiom  *(XS, low priority)*

Sources: `gemma4.java` (`Timer` class, L320–333), `qwen35.java`.

`try (var t = Timer.log("Load tensors")) { ... }` prints `Load tensors: 312 ms` to stderr on close. 12-line helper. The project already times model load + JNI init + first-token latency in ad-hoc places; one helper would unify them. Friendly to `LogCaptor` (already wired in tests).

### 2.7 `jbang`-runnable single-file example  *(XS, medium priority)*

Sources: all 5 `mukel` runtimes.

Ship a self-contained `Example.java` with the `///usr/bin/env jbang` shebang and `//DEPS net.ladenthin:llama:5.0.0`. Lowers the "try it once" barrier from `mvn dependency:get + classpath wrangling` to one curl-and-run line. Pairs naturally with publishing on Maven Central.

### 2.8 Documented system-properties table in the README  *(XS, medium priority)*

Sources: all `mukel` runtimes (each documents its own `-D…` knobs alongside `--flag` parameters).

Currently the `LlamaSystemProperties` setters (`net.ladenthin.llama.lib.path`, `.tmpdir`, `.osinfo.architecture`, `.test.ngl`, the per-test `.vision.*` and `.nomic.path` properties) are scattered across `CLAUDE.md`, source javadoc, and test setup. A single README table listing every supported property + default + meaning improves discoverability.

---

## 3. Per-repo unique ideas

### 3.1 `llama3.java`

- **`--echo` debug mode** *(XS, low)* — dump every token to stderr separately from `--stream`. Useful for teaching / first-time-user debugging.
- **`-Dllama.VectorBitSize=0|128|256|512`** *(XS, low)* — runtime knob to pin SIMD width / benchmark when multiple ISA variants are co-located. Equivalent for this project: a system property selecting GGML CPU backend variant when multiple are on the library path.

### 3.2 `gemma4.java`

- **README note about `llama-quantize --pure`** *(XS, low)* — mixed-quant GGUF files (e.g. `Q4_0` with embedded `F16` tensors) cause subtle issues that users discover only by trawling the upstream issue tracker. Surface the workaround in the troubleshooting section.

### 3.3 `gptoss.java`

- **`Reasoning: low|medium|high` system-message injection** *(S, high if GPT-OSS users present)* — add `InferenceParameters.setReasoningEffort(LOW|MEDIUM|HIGH)` that synthesises the Harmony `Reasoning: X` line. Encodes a contract operators currently discover only by reading the Jinja template.
- See also Harmony channel decoder under §2.2.

### 3.4 `qwen35.java`

- **"Empty `<think></think>` injection" to *disable* thinking on Qwen models** *(S, medium)* — prefill the assistant header with `<think>\n\n</think>\n\n` so the model produces only the visible answer with zero reasoning tokens, regardless of whether llama.cpp's `reasoning_format` understands the family. Complements existing `setReasoningFormat` / `setReasoningBudgetTokens`. Should land as a `ChatRequest` option or a thin Qwen-aware preset.

### 3.5 `nemotron3.java`

- All unique-value findings overlap with §2 themes; no Nemotron-specific item warranted its own row beyond what §2.1 / §2.2 already cover.

### 3.6 `llamacpp4j`

`llamacpp4j` is dormant (single commit, July 2023, pre-GGUF era) and its design is largely uninteresting (SWIG-generated facade with opaque `SWIGTYPE_p_*` pointers leaking through). The *useful* ideas come from the underlying `llama.h` API surface that SWIG happens to expose, not from anything Sebicom designed:

- **`llama_state_*` save/load API** *(M, medium)* — `llama_copy_state_data`, `llama_set_state_data`, `llama_save_session_file` / `llama_load_session_file`. Useful for prompt-warm-start, multi-tenant resumption, and benchmarking. `ModelParameters` doesn't surface KV-cache snapshotting as first-class Java API.
- **`llama_apply_lora_*` hot-apply at runtime** *(M, medium)* — adapter hot-swap without reloading the base model (common multi-tenant pattern). Use the modern `llama_adapter_lora_*` API, not the deprecated file-based one Sebicom exposes.
- **`llama_model_quantize` exposure** *(S, low)* — one-line wrapper that converts FP16 → Q4/Q5/Q8 GGUF in-process. Lets Java apps build a "download FP16 → quantize for this device" path without shelling out.
- **`llama_print_system_info()` wrapper** *(XS, low)* — trivial diagnostic that prints `AVX = 1 | AVX2 = 1 | …` etc. Useful for bug reports.

**Explicitly skip from `llamacpp4j`**: the SWIG-generated facade itself (brittle, opaque pointer types leak), the `mainn(argv)` shortcut that forwards to `llama.cpp`'s reference CLI, the single-OS prebuilt `.so` checked into git, the README-documented "install JAR into local Maven repo" workflow. `java-llama.cpp`'s JSON-over-JNI + classifier-based packaging is strictly better.

---

## 4. Explicitly out of scope

Recurring "don't port" themes across all 6 sources:

- **Pure-Java tensor kernels / GGUF parser / quantization classes** — redundant with llama.cpp; the entire raison d'être of this project is to *delegate* these to the upstream C++.
- **GraalVM Native Image AOT model preloading** — already captured as its own design-investigation TODO in `CLAUDE.md`; not duplicated here.
- **Reimplementations of samplers** (`ToppSampler`, `CategoricalSampler`) — llama.cpp's sampler chain already covers TOP_P, TYP_P, MIN_P, XTC, DRY, etc.
- **Single-file `jbang` distribution of the whole library** — wrong shape for a JNI library that ships per-OS classifier JARs. *(A single-file `jbang` *example* per §2.7 is fine; the library itself stays multi-module.)*
- **Hard-coded per-model chat-template token strings** (e.g. Gemma's `<|turn>` / `<|think|>`) — llama.cpp's chat-template engine handles these generically.

---

## 5. Prioritised backlog (top picks across all 6 sources)

Sorted by `priority × (1 / effort)`. Items in **bold** are the recommended first batch.

| # | Item | Source(s) | Effort | Priority |
|---|------|-----------|:--:|:--:|
| 1 | **UTF-8 boundary-safe streaming decoder** | §2.1 | S | medium-high |
| 2 | **Tri-state thinking-channel router** (generic `<think>`) | §2.2 | S | medium |
| 3 | **Operator-grade per-run timing line on stderr** | §2.5 | XS | medium |
| 4 | **`jbang`-runnable single-file example** | §2.7 | XS | medium |
| 5 | **System-properties table in README** | §2.8 | XS | medium |
| 6 | Empty `<think></think>` injection (Qwen) | §3.4 | S | medium |
| 7 | `llama_state_*` save/load Java API | §3.6 | M | medium |
| 8 | `llama_adapter_lora_*` hot-apply API | §3.6 | M | medium |
| 9 | Chat REPL with `/quit /exit /context` | §2.3 | XS | low-medium |
| 10 | Harmony channel decoder for GPT-OSS | §2.2 | M | conditional (ship when GPT-OSS users ask) |
| 11 | `Reasoning: X` system-message injection | §3.3 | S | conditional |
| 12 | ANSI colour auto-detection helper | §2.4 | XS | low |
| 13 | `AutoCloseable Timer.log()` idiom | §2.6 | XS | low |
| 14 | `llama_print_system_info()` wrapper | §3.6 | XS | low |
| 15 | `llama_model_quantize` Java surface | §3.6 | S | low |
| 16 | README note on `llama-quantize --pure` | §3.2 | XS | low |
| 17 | `--echo` debug knob in example | §3.1 | XS | low |
| 18 | `-Dllama.VectorBitSize`-style ISA knob | §3.1 | XS | low |

Items 1–5 are the recommended first batch — none requires JNI changes and each closes a documented operator pain point.

---

## 6. Recommended next action

Implement items 1, 3, 4, 5 in one focused "operator-facing ergonomics" commit:

- UTF-8 boundary-safe streaming decoder (genuine correctness fix)
- Per-run timing line (cheap operator signal)
- One `jbang`-runnable example file
- README system-properties table

Estimated total: ~1–2 days of work, zero JNI changes, all backed by Java-only tests. Items 2 and 6–8 are good follow-ups once a real user asks.
