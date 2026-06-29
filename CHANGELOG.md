# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from version 5.0.0 onward. Pre-fork releases (`1.x`–`4.2.0`) were authored by
[`kherud/java-llama.cpp`](https://github.com/kherud/java-llama.cpp).

## [Unreleased]

### Added
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.0).
- `docs/RELEASE.md` capturing the maintainer-facing release procedure (moved out of CHANGELOG).
- OpenSSF Best Practices badge (project 12862) on README.
- OpenAI-compatible `parallel_tool_calls` support: `ChatRequest.withParallelToolCalls(Boolean)` / `getParallelToolCalls()`, `InferenceParameters.withParallelToolCalls(boolean)`, and pass-through in the `/v1/chat/completions` server mapper.
- Real-model tool-calling integration tests for blocking and streaming required tool calls (`ToolCallingIntegrationTest`, Qwen2.5-1.5B-Instruct), wired into CI and `validate-models`.
- End-to-end vision input across blocking, typed `ChatRequest`, streaming, and OpenAI-compatible request mapping; real-model tests verify that distinct red and blue images produce the correct semantic answers.
- Explicit `setMmprojAuto(boolean)` and `setMmprojOffload(boolean)` controls, including the upstream `--no-mmproj-auto` and `--no-mmproj-offload` flags.
- Per-request KV controls: `InferenceParameters.withSlotId(int)` and `withCacheReuse(int)`.
- Per-request DRY sampling to `InferenceParameters` (`dry_multiplier`/`dry_base`/`dry_allowed_length`/`dry_penalty_last_n`/`dry_sequence_breakers`).
- `ModelParameters.enableSwaFull()` (`--swa-full`): keep full-size SWA KV cache to enable cross-request prompt-prefix reuse.
- Typed cache observability through `Usage.getCachedTokens()`, `Usage.getProcessedPromptTokens()`, `SlotMetrics`, and `ServerMetrics.getSlotMetrics()`.
- Authenticated JSON `GET /metrics` and `GET /slots` endpoints on the embedded server.

### Changed
- Unified `CONTRIBUTING.md` and `SECURITY.md` structure with sibling repositories in the project family.
- Reconciled Java baseline to **11+** across `pom.xml`, README badge, `CLAUDE.md`, and `CONTRIBUTING.md`.
- README license badge corrected from "Apache 2.0" to "MIT" (matches `LICENSE` file and `pom.xml`).
- `pom.xml` SCM URL: `tree/master` → `tree/main` (default branch renamed).
- Upgraded llama.cpp from b9151 to b9172.
- Upgraded llama.cpp from b9803 to b9829. Compiles the new upstream `server-stream.cpp` (resumable-streaming SSE replay buffer) into `libjllama`, required because `server-context`/`server-http`/`server-models` now reference its symbols; refreshed `patches/0001` for the `tests/test-export-graph-ops.cpp` rename and the `server.cpp` GC-init context shift.
- Upgraded llama.cpp from b9829 to b9839. Pure version bump — no project source changes: all four patches (`0001`–`0004`) apply unchanged against b9839, and every upstream change in the range is absorbed inside upstream-compiled translation units. Brings DFlash block-diffusion speculative decoding (`--spec-type draft-dflash`), the MiniCPM5 XML tool-call chat template, a server `--reasoning-preserve` flag (preserve reasoning trace across the full history when the template supports it), and Jinja `min`/`max` array filters; removes the now-unused `common/regex-partial.{cpp,h}` (partial-regex matching is fully inside the PEG parser), which the project never referenced.
- Upgraded llama.cpp from b9839 to b9840. Pure version bump — no project source changes: the range is entirely the new **DeepSeek-V4** architecture (new `deepseek4` arch + dedicated `llama-kv-cache-dsv4` cache, `sqrtsoftplus` MoE gating, hyper-connection/compressor hparams + tensors, conversion scripts and embedded chat template), all absorbed inside upstream-compiled `libllama` and the Python converters. Upstream's `src/CMakeLists.txt` adds the new `llama-kv-cache-dsv4.cpp` itself (built via FetchContent). All four patches (`0001`–`0004`) apply unchanged; the project binds none of the new symbols.
- `configureParallelInference` now applies `slot_prompt_similarity` live via `server_context::set_slot_prompt_similarity()` (upstream PR ggml-org/llama.cpp#22393, carried as `patches/0003` until merged), instead of validating it and discarding the value.
- Extracted the `chatWithTools` agent loop into `ToolCallingAgent`; tool-result errors (unknown tool / handler exception) are now JSON-serialized so tool names containing special characters remain valid JSON.

### Fixed
- Per-request `reasoning_budget_tokens` is now honored (via `patches/0004`, upstream PR ggml-org/llama.cpp#23116): `reasoning_budget_tokens=0` suppresses thinking. `ReasoningBudgetTest` now asserts the suppression directly (the previous test that pinned the unfixed-bug behavior was removed).
- Preserved decoded image buffers across the JNI chat boundary and submitted media requests through llama.cpp's upstream multimodal task path instead of silently tokenizing them as text-only prompts.
- Preserved multipart image content when using the typed `ChatRequest` serializer.
- The standalone OpenAI-compatible server now advertises vision only when the loaded model confirms usable vision support.
- `Session` now pins every inference request to its configured slot, so generation and slot save/restore/erase target the same KV state.
- Cached-token usage is preserved through typed Java responses and OpenAI Responses/Anthropic blocking and streaming adapters.

### Added
- Reasoning-budget tests (Qwen3-0.6B).

## [5.0.1] - 2026-05-14

### Added
- `InferenceParameters.setContinueFinalMessage(boolean)` for the vLLM/transformers-compatible prefill-assistant heuristic (llama.cpp b9134+).
- Tests for `setContinueFinalMessage`.
- Comprehensive Javadoc on public APIs (PR #129).
- Maven Central badge on README (PR #130).

### Changed
- Bumped project version to 5.0.1-SNAPSHOT (PR #127), then released as 5.0.1 (PR #135).
- Refactored GitHub release workflow to parallelise snapshot and release jobs (PR #128).
- Removed snapshot build documentation and badge (PR #131).
- Upgraded Windows CI to `windows-2025` with Visual Studio 2026 (PR #132).
- Switched Windows MSVC runtime from dynamic (`/MD`) to static (`/MT`) to eliminate the `msvcp140.dll` runtime dependency (PR #133).
- Upgraded llama.cpp from b9106 to b9134 (PR #134), then to b9150 (PR #136), then to b9151 (PR #139).
- Refactored CI workflow with explicit snapshot/tag check gates (PR #137).
- Removed `setCtxSizeDraft()` — the underlying CLI flag was deleted upstream in llama.cpp b9106.

### Fixed
- `fix(publish):` quoted gate job names to avoid YAML colon-in-scalar parse errors (PR #138).
- Release routing in the publish workflow now correctly distinguishes snapshot vs. tag pushes.

## [5.0.0] - 2026-05-11

First release of the fork under the `net.ladenthin:llama` Maven coordinates. ~100 merged pull requests since baseline `49be664` (the last pre-fork upstream commit).

### Added
- First publish to Maven Central under `net.ladenthin:llama`.
- Pre-built native libraries for Linux (x86-64, aarch64), macOS (x86-64, arm64), and Windows (x86-64, x86).
- Java API surface: `LlamaModel`, `ModelParameters`, `InferenceParameters`, `LlamaIterator` / `LlamaIterable` for streaming, chat completion (`chatComplete`, `generateChat`, `chatCompleteText`), embeddings, reranking, infilling, raw JSON endpoint handlers, slot management (`saveSlot`, `restoreSlot`, `eraseSlot`), and `getModelMeta()`.
- `chatComplete()` for OpenAI-compatible chat completions, re-implemented from scratch based on a patch by @vaiju1981 (PR #61; see `docs/history/CHAT_INTEGRATION_SUMMARY.md`).
- `mmproj`, reasoning-budget, sigma, and sleep-idle parameters added to `ModelParameters`.
- JaCoCo code-coverage reporting integrated with Coveralls and Codecov (PR #124).
- CodeQL static-analysis workflow on push, PR, and a weekly schedule.
- Automated Claude Code review workflow on pull requests.
- Dependabot for Maven and GitHub Actions dependency updates.
- Automatic snapshot release workflow on `main` push (PR #105) publishing to the Sonatype Central snapshot repository.
- CUDA, Metal, and Vulkan build support via local CMake build.
- Android integration documented in README.
- All system properties (`net.ladenthin.llama.*`) and `LogLevel` values documented.
- `CLAUDE.md` maintainer guide covering upstream upgrade procedure and the b5022→b9172 breaking-change table.

### Changed
- Migrated Maven group and artifact from `de.kherud:java-llama.cpp` to `net.ladenthin:llama` (PR #101).
- Migrated Maven Central publishing from OSSRH (Legacy) to the Sonatype Central Publisher Portal.
- Deleted the hand-ported `server.hpp` fork (~3,780 lines) and linked the upstream `llama.cpp` server source files directly into `jllama`. ~4,100 C++ lines removed in total; future upstream upgrades become a CMake version bump. **The Java API is unchanged.** See `docs/history/REFACTORING.md`.
- Compiled upstream server-context / queue / task / models directly into jllama (PR #96).
- Unified CI into a single `publish.yml` workflow with cross-compilation, testing, coverage, and release stages.
- Upgraded CUDA from 12.1 to 13.2 (PR #50).
- Upgraded llama.cpp from b8913 through b9106 across multiple incremental upgrades.
- `setDraftMax` / `setDraftMin` now emit the canonical `--spec-draft-n-max` / `--spec-draft-n-min` flags (llama.cpp b9016 removed the old aliases).
- Bumped CI GitHub Actions: `actions/checkout` v4 → v6, `actions/upload-artifact` v6 → v7, `actions/download-artifact` v6 → v8, `codeql-action` v3 → v4.

### Fixed
- Javadoc warnings resolved across the public API by adding missing comments.
- `cache_idle_slots` slot-parameter handling aligned with the upstream rename (b8841 → b8854).

## Pre-fork history (kherud/java-llama.cpp 1.x–4.2.0)

Releases `1.1.1` through `4.2.0` were authored by [@kherud](https://github.com/kherud) on the upstream repository. The full upstream release notes are at
<https://github.com/kherud/java-llama.cpp/releases>. The fork's baseline is upstream commit `49be664` (tagged `v4.2.0`, 2025-06-20).

For an architecture-level diff between the pre-fork baseline (`49be664`) and the first 5.0.0 candidate (`24918e4`), see [`docs/history/49be664_24918e4.md`](docs/history/49be664_24918e4.md). For the server-fork-deletion refactor that culminated in 5.0.0, see [`docs/history/REFACTORING.md`](docs/history/REFACTORING.md). For the chat-completion integration that landed in 5.0.0, see [`docs/history/CHAT_INTEGRATION_SUMMARY.md`](docs/history/CHAT_INTEGRATION_SUMMARY.md).

[Unreleased]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.1...HEAD
[5.0.1]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.0...v5.0.1
[5.0.0]: https://github.com/bernardladenthin/java-llama.cpp/releases/tag/v5.0.0
