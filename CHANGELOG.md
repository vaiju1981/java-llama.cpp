# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Release Process

> Paste this prompt into a new Claude Code session, fill in the three placeholders, and send it to perform a release.

```
Release `{PROJECT}` to Maven Central.

**Step 1 — Prepare the release (do immediately):**
1. Read the current version from `pom.xml` on `main` — it will be `{VERSION}-SNAPSHOT`
2. Strip `-SNAPSHOT` from `pom.xml` (→ `{VERSION}`)
3. In `README.md`, update **both**:
   - The release dependency example to `{VERSION}`
   - The snapshot dependency example to `{VERSION}-SNAPSHOT` (it should already match, but verify)
4. Commit both files directly to `main` (no pull request)

**Step 2 — Wait for manual confirmation:**
I will create the `v{VERSION}` tag and GitHub release manually — wait for me to confirm
the release is published on Maven Central before proceeding.

**Step 3 — Post-release snapshot bump (after my confirmation):**
Bump **both** files on `main`:
- `pom.xml` → `{NEXT_VERSION}-SNAPSHOT`
- `README.md` snapshot dependency example → `{NEXT_VERSION}-SNAPSHOT`

Commit both changes together directly to `main`.

**Placeholders:**

| Placeholder      | Value                                        |
|------------------|----------------------------------------------|
| `{PROJECT}`      | *(project name)*                             |
| `{VERSION}`      | *(release version, e.g. `1.3.0`)*           |
| `{NEXT_VERSION}` | *(next snapshot base, e.g. `1.3.1`)*        |
```

---

## [Unreleased]

### Added
- OpenSSF Best Practices badge (project 12862) added to README.
- CONTRIBUTING.md, SECURITY.md, and CHANGELOG.md to satisfy OpenSSF passing-level criteria.

### Changed
- Upgraded llama.cpp from b9151 to b9172.
- Added reasoning-budget tests (Qwen3-0.6B).

---

## [5.0.1] - 2026-05-14

### Added
- `InferenceParameters.setContinueFinalMessage(boolean)` for vLLM-compatible prefill-assistant heuristic (llama.cpp b9134+).
- Tests for `setContinueFinalMessage`.

### Changed
- Upgraded llama.cpp from b9106 to b9145 (b9106 → b9134 → b9145 in increments).
- Switched Windows MSVC runtime from dynamic (`/MD`) to static (`/MT`) to eliminate `msvcp140.dll` dependency.
- Updated CI Windows runners to `windows-2025-vs2026` (Visual Studio 18 2026).
- CI publish workflow: added check-snapshot/check-tag gates for correct release routing; bumped `softprops/action-gh-release` v2 → v3 (Node 24).
- Removed `setCtxSizeDraft()` (CLI flag removed in llama.cpp b9106).

### Fixed
- CI gate job name quoting to prevent YAML parse errors.
- Release routing in publish workflow to correctly distinguish snapshot vs. tag pushes.

---

## [5.0.0] - 2026-05-11

### Added
- First release under the `net.ladenthin` Maven group ID (`net.ladenthin:llama`), published to Maven Central.
- Pre-built native libraries for Linux (x86-64, aarch64), macOS (x86-64, arm64), and Windows (x86-64, x86).
- Java API surface: `LlamaModel`, `ModelParameters`, `InferenceParameters`, `LlamaIterator`/`LlamaIterable` for streaming, chat completion (`chatComplete`, `generateChat`, `chatCompleteText`), embeddings, reranking, infilling, raw JSON endpoint handlers, slot management (`saveSlot`, `restoreSlot`, `eraseSlot`), and `getModelMeta()`.
- `mmproj`, reasoning-budget, sigma, and sleep-idle parameters added to `ModelParameters`.
- JaCoCo code-coverage reporting integrated with Coveralls and Codecov.
- CodeQL static-analysis workflow running on push, PR, and weekly schedule.
- Claude Code automated code-review workflow on pull requests.
- Dependabot for Maven and GitHub Actions dependency updates.
- Snapshot builds published to Sonatype Central snapshot repository on every `main` push.
- CUDA, Metal, and Vulkan build support via local CMake build.
- Android integration documented in README.
- All system properties (`net.ladenthin.llama.*`) and `LogLevel` values documented.

### Changed
- Migrated Maven group and artifact from `de.kherud:java-llama.cpp` to `net.ladenthin:llama`.
- Migrated Maven Central publishing from OSSRH (Legacy) to Sonatype Central Publisher Portal.
- Unified CI into a single `publish.yml` workflow with cross-compilation, testing, coverage, and release stages.
- CI GitHub Actions bumped: `actions/checkout` v4 → v6, `actions/upload-artifact` v6 → v7, `actions/download-artifact` v6 → v8, `codeql-action` v3 → v4.
- Upgraded llama.cpp from b8913 through b9106 (multiple incremental upgrades).
- `setDraftMax`/`setDraftMin` fixed to emit canonical `--spec-draft-n-max`/`--spec-draft-n-min` flags (b9016+ removed old aliases).

### Fixed
- Javadoc: resolved all 69 warnings by adding missing comments.
- Fixed `--cache-idle-slots` bug in slot management parameters.

---

[Unreleased]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.1...HEAD
[5.0.1]: https://github.com/bernardladenthin/java-llama.cpp/compare/v5.0.0...v5.0.1
[5.0.0]: https://github.com/bernardladenthin/java-llama.cpp/releases/tag/v5.0.0
