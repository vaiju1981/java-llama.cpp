# Suggested upstream-issue comments

Each section below corresponds to one open issue on
[`kherud/java-llama.cpp`](https://github.com/kherud/java-llama.cpp) that has
been resolved (or documented away) in the downstream fork at
[`bernardladenthin/java-llama.cpp`](https://github.com/bernardladenthin/java-llama.cpp).

The fenced block under each issue is the exact comment body to paste on the
upstream tracker. Evidence pointers come from `docs/history/49be664_open_issues.md`.

Issues with verdict `NEEDS INVESTIGATION` or `STILL POSSIBLE` are intentionally
omitted: #77, #79, #81, #82, #83, #85, #91, #117.

---

## #124 — Request an update

URL: https://github.com/kherud/java-llama.cpp/issues/124

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
The fork now pins llama.cpp to `b9284` with a per-version upgrade workflow
(see `CLAUDE.md` and `.github/workflows/release.yaml`); releases are published
continuously rather than ad-hoc.
```

---

## #123 — Supporting latest llama.cpp version

URL: https://github.com/kherud/java-llama.cpp/issues/123

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
llama.cpp is pinned to `b9284`, which natively supports Qwen3 and Qwen3-VL;
the fork also fetches `tools/mtmd` and links it into `jllama`
(`CMakeLists.txt:125-145,255`), and exposes a typed multimodal Java surface
(`ContentPart`, `ChatMessage(role, List<ContentPart>)`,
`InferenceParameters.setMessages(List<ChatMessage>)`) — see PR #189.
```

---

## #121 — Error on Android apk (aarch64 vs arm64-v8a)

URL: https://github.com/kherud/java-llama.cpp/issues/121

```markdown
Resolved for 64-bit Android in the downstream fork at
https://github.com/bernardladenthin/java-llama.cpp.
`OSInfo` now detects Android runtimes and returns `Linux-Android` as the OS
folder; the dockcross-android-arm64 CI publishes
`src/main/resources/net/ladenthin/llama/Linux-Android/aarch64/libjllama.so`,
matching what the loader resolves at runtime
(`.github/workflows/publish.yml:133`, `OSInfo.java:256-259,350`).
32-bit Android (`armeabi-v7a`) remains an open enhancement, not a regression.
```

---

## #120 — What models are supported?

URL: https://github.com/kherud/java-llama.cpp/issues/120

```markdown
In the downstream fork at https://github.com/bernardladenthin/java-llama.cpp
model architecture support is delegated to the pinned llama.cpp `b9284`
(`CLAUDE.md:11`), which includes Qwen3, Qwen3-VL, SmolLM2 and many other
recent architectures.
```

---

## #119 — Update questions

URL: https://github.com/kherud/java-llama.cpp/issues/119

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
The fork has an automated per-build version-bump cadence; the upgrade
procedure is documented in `CLAUDE.md`, and `git log --oneline` shows
continuous bumps up to llama.cpp `b9284`.
```

---

## #116 — Non-daemon thread inhibits JVM termination

URL: https://github.com/kherud/java-llama.cpp/issues/116

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
The native worker is a C++ `std::thread` (so JVM daemon semantics do not apply)
and is explicitly joined on close: `Java_net_ladenthin_llama_LlamaModel_delete`
calls `jctx->server.terminate()` twice and then `jctx->worker.join()`
(`src/main/cpp/jllama.cpp:937-940`). Streaming readers are drained before
shutdown.
```

---

## #113 — Allow tracking the progress of loading a LlamaModel

URL: https://github.com/kherud/java-llama.cpp/issues/113

```markdown
Implemented in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp
via PR #188 (`70df324`), with a follow-up symbol-export fix in PR #189
(`36d8862`). A new `LoadProgressCallback` functional interface (single method
`boolean onProgress(float)`; return `false` to abort) and a constructor
overload `LlamaModel(ModelParameters, LoadProgressCallback)` plumb the
callback through a new JNI entry point. The upstream
`llama_model_params.progress_callback` only emits the float, so the richer
payload (file name, bytes loaded, etc.) from the original request is not
exposed — that would need an upstream API change.
```

---

## #112 — Qwen 3 model does not load

URL: https://github.com/kherud/java-llama.cpp/issues/112

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp
by bumping the pinned llama.cpp to `b9284` (`CLAUDE.md:11`), which includes
the Qwen3 architecture.
```

---

## #111 — Running with a predefined number of cores

URL: https://github.com/kherud/java-llama.cpp/issues/111

```markdown
The API for this already exists and is exercised in the downstream fork at
https://github.com/bernardladenthin/java-llama.cpp:
`ModelParameters.setThreads(int)` and `setThreadsBatch(int)` map to the
`--threads` / `--threads-batch` CLI flags (`ModelParameters.java:51,61`).
```

---

## #110 — Running embedding in batch

URL: https://github.com/kherud/java-llama.cpp/issues/110

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
In addition to the single-prompt `LlamaModel.embed(String)`, the fork exposes
`handleEmbeddings(String paramsJson, boolean oaiCompat)`
(`LlamaModel.java:316`); the native helper `extract_embedding_prompt`
(`src/main/cpp/json_helpers.hpp:137`) accepts either a string or an array of
strings, enabling batched embeddings.
```

---

## #107 — `OS_NAME` for Mac in CMakeLists.txt

URL: https://github.com/kherud/java-llama.cpp/issues/107

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
`CMakeLists.txt:196` now matches both names:
`if(OS_NAME MATCHES "^Linux" OR OS_NAME STREQUAL "Mac" OR OS_NAME STREQUAL "Darwin")`.
The Java side normalises both `Mac` and `Darwin` to the `Mac` folder
(`OSInfo.java:345-346`).
```

---

## #104 — How to enable `offload_kqv`?

URL: https://github.com/kherud/java-llama.cpp/issues/104

```markdown
KQV offload is on by default upstream. The downstream fork at
https://github.com/bernardladenthin/java-llama.cpp exposes the inverse —
`ModelFlag.NO_KV_OFFLOAD` mapped to `--no-kv-offload`
(`args/ModelFlag.java:50`, used via `ModelParameters.java:739`). Leaving the
flag unset preserves `offload_kqv=true`.
```

---

## #103 — VLM support / image input for multimodal models

URL: https://github.com/kherud/java-llama.cpp/issues/103

```markdown
Implemented in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp
in PR #189. The build links the upstream `mtmd` library into `jllama`
(`CMakeLists.txt:125-145,253-255`); `ModelParameters` exposes `setMmproj`,
`setMmprojUrl`, `enableMmprojAuto`, `enableMmprojOffload`
(`ModelParameters.java:1250-1281`); and the new typed image API ships as
`ContentPart.text/imageUrl/imageBytes/imageFile`, `ChatMessage(role, List<ContentPart>)`,
`ChatMessage.userMultimodal(...)`, and `InferenceParameters.setMessages(List<ChatMessage>)`.
The serializer emits the OAI array-form `content` that the upstream chat path
already routes through `mtmd` — no new JNI was required.
```

---

## #102 — Native call to `delete` doesn't free memory

URL: https://github.com/kherud/java-llama.cpp/issues/102

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
The native destructor (`src/main/cpp/jllama.cpp:917-948`) now clears the field
pointer first, drains `readers` under lock, calls `server.terminate()` twice
to close the half-init race, joins the worker, frees `vocab_only_model`, and
deletes `jctx` (destroying the embedded `server_context`). The reporter's
open/close loop is covered by `MemoryManagementTest#testOpenCloseLoopDoesNotLeak`
(PR #185, commit `cba693c`), which asserts `VmRSS` growth under 200 MB over
20 iterations on Linux.
```

---

## #101 — Log messages are not redirected to a provided consumer

URL: https://github.com/kherud/java-llama.cpp/issues/101

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
`Java_net_ladenthin_llama_LlamaModel_setLogger` (`src/main/cpp/jllama.cpp:954-977`)
now registers the JNI callback as a global ref and installs a trampoline via
`llama_log_set(...)`, which invokes the user's `BiConsumer<LogLevel,String>`
on every log line. JSON vs text formatting is gated by `LogFormat`.
```

---

## #98 — Error loading `nomic-embed-text-v1.5.f16.gguf`

URL: https://github.com/kherud/java-llama.cpp/issues/98

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
The original failure came from the bindings not forwarding `--embedding` at
all; calling `ModelParameters.enableEmbedding()` (sets `ModelFlag.EMBEDDING`,
`ModelParameters.java:1040`) plus `setPoolingType(...)` now initialises the
embedding pipeline correctly. Regression-tested by
`LlamaEmbeddingsTest#testNomicEmbedLoads` (PR #185, commit `cba693c`), which
reproduces the reporter's `setBatchSize(8192).setUbatchSize(8192)` config and
asserts the returned vector length is 768; CI downloads
`nomic-embed-text-v1.5.f16.gguf` via `NOMIC_EMBED_MODEL_URL` in
`.github/workflows/publish.yml`.
```

---

## #95 — Inference content repetition that can't be ended

URL: https://github.com/kherud/java-llama.cpp/issues/95

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
`LlamaIterator` now toggles `hasNext = !output.stop` and exposes explicit
cancellation via `AutoCloseable` (`LlamaIterator.java:51-77`,
`LlamaIterable.java:44`). Repetition tuning remains a sampler concern handled
via `InferenceParameters` (`setRepeatPenalty`, `setMinP`, `setStopStrings`).
Regression-tested by `LlamaModelTest#testIteratorTerminatesOnRepetitivePrompt`
(PR #185, commit `cba693c`).
```

---

## #90 — Process to rebuild after Java-only changes

URL: https://github.com/kherud/java-llama.cpp/issues/90

```markdown
Documented in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
The "Build Commands" section in `CLAUDE.md` separates `mvn compile` /
`mvn package` (Java-only — no native rebuild) from the `cmake -B build &&
cmake --build build` native step. Running `mvn package` alone reuses the
pre-built shared libraries under `src/main/resources/net/ladenthin/llama/{OS}/{ARCH}/`.
```

---

## #89 — What are slots?

URL: https://github.com/kherud/java-llama.cpp/issues/89

```markdown
For reference: in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp
the hand-ported `src/main/cpp/server.hpp` has been removed; the project now
compiles the upstream `server-context.cpp` / `server-queue.cpp` /
`server-task.cpp` / `server-models.cpp` directly. `server_slot` is therefore
an upstream-owned concept; callers see it indirectly via slot-related task
params (`InferenceParameters.setIdSlot`, etc.).
```

---

## #88 — JSON request like role system or role user isn't supported?

URL: https://github.com/kherud/java-llama.cpp/issues/88

```markdown
Supported in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
`LlamaModel.chatComplete(InferenceParameters)` routes to native
`handleChatCompletions` which accepts the standard OpenAI
`messages: [{role, content}, ...]` JSON format
(`LlamaModel.java:215-238`); the chat template is auto-applied via the
compiled upstream chat pipeline.
```

---

## #87 — How to reset the context on every turn?

URL: https://github.com/kherud/java-llama.cpp/issues/87

```markdown
In the downstream fork at https://github.com/bernardladenthin/java-llama.cpp
the upstream server's per-slot KV state is exposed via
`InferenceParameters.setCachePrompt(boolean)` (line 116) and
`setIdSlot(int)`. To "reset" between turns, pass a fresh `id_slot` or set
`cache_prompt=false`; the upstream server also supports an explicit
`/slots/{id}:erase` task type that the chat path can dispatch.
```

---

## #86 — Does the CUDA jar handle both CPU and GPU or only GPU?

URL: https://github.com/kherud/java-llama.cpp/issues/86

```markdown
Documented in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
The CUDA-classified jar is **CUDA-only at runtime** — the built
`libjllama.so` dynamically links against `libcudart.so.13` / `libcublas.so.13`
and does not auto-fall back to CPU. Users on CPU-only hosts must pick the
default (CPU) classifier instead of `cuda13-linux-x86-64`. This is now
explicit in the README's "Choosing the right classifier" section.
```

---

## #84 — rerank

URL: https://github.com/kherud/java-llama.cpp/issues/84

```markdown
Supported in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
`ModelParameters.enableReranking()` (`ModelParameters.java:1049`) enables the
rerank endpoint; `LlamaModel.rerank(boolean, String, String...)`
(`LlamaModel.java:170`) and `rerank(query, documents)` (line 187) return
score pairs. End-to-end coverage exists in `RerankingModelTest` against
`jina-reranker-v1-tiny-en-Q4_0.gguf`.
```

---

## #80 — Segfault on model open and close

URL: https://github.com/kherud/java-llama.cpp/issues/80

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
The native destructor (`src/main/cpp/jllama.cpp:917-948`) now waits on
`worker_ready` before terminating, drains `readers` under lock, and calls
`server.terminate()` twice with a 1 ms sleep to close the
"signalled ready but `start_loop()` not yet running" race — the exact
half-init case that produced the original `std::_Rb_tree::_M_erase` crash.
Regression-tested by `MemoryManagementTest#testOpenCloseWithoutGeneration`
(PR #185, commit `cba693c`): 20 try-with-resources open+immediate-close
iterations with no generation.
```

---

## #78 — Add support for `params.lora_adapters` (llama.cpp ≥ b3534)

URL: https://github.com/kherud/java-llama.cpp/issues/78

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
`ModelParameters.addLoraAdapter(String)` and
`addLoraScaledAdapter(String, float)` (`ModelParameters.java:906,918`) map to
`--lora` and `--lora-scaled`, which the upstream parser already converts into
the vector-of-`lora_adapters` form; `setLoraInitWithoutApply()` covers the
matching flag. llama.cpp pin is `b9284`, well past `b3534`.
```

---

## #72 — JNI signature exception (`@Nullable BiConsumer`)

URL: https://github.com/kherud/java-llama.cpp/issues/72

```markdown
Resolved in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
The `setLogger` signature no longer carries `@Nullable`
(`LlamaModel.java:128`), so `javac -h` accepts it. The build now uses JDK 21
targeting bytecode 1.8.
```

---

## #70 — Add `vocab_only`

URL: https://github.com/kherud/java-llama.cpp/issues/70

```markdown
Implemented in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
`ModelParameters.setVocabOnly()` (`ModelParameters.java:1336`) toggles
`ModelFlag.VOCAB_ONLY` → `--vocab-only` (`args/ModelFlag.java:92`); the
native layer branches on `jctx->vocab_only` to skip server/worker init while
keeping the vocab available for tokenize/decode
(`src/main/cpp/jllama.cpp:710-718, 923, 944`).
```

---

## #50 — Android build error: `libllama.so is incompatible with aarch64linux`

URL: https://github.com/kherud/java-llama.cpp/issues/50

```markdown
Addressed in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp.
When `ANDROID_ABI` is set or `OS_NAME` matches `Android`, the build routes
output to `Linux-Android/<ABI>/` instead of the host-arch folder
(`CMakeLists.txt:151-153,243`); CI cross-builds via dockcross-android-arm64
on Linux runners (`.github/workflows/publish.yml:133`), so the
macOS-host artifact can no longer collide with the Android target.
A formal end-to-end test on a macOS-M2 NDK host is still pending.
```

---

## #34 — Support multimodal inputs

URL: https://github.com/kherud/java-llama.cpp/issues/34

```markdown
Implemented in the downstream fork at https://github.com/bernardladenthin/java-llama.cpp
in PR #189. The upstream `mtmd` library is linked into `jllama`
(`CMakeLists.txt:125-145,253-255`); `ModelParameters` exposes the mmproj
flags (`ModelParameters.java:1250-1281`); and the typed image API ships as
`ContentPart` + `ChatMessage(role, List<ContentPart>)` +
`InferenceParameters.setMessages(List<ChatMessage>)`, which emit the OAI
multipart `content` the upstream chat path already routes through `mtmd`.
See #103 for the parallel write-up.
```
