# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Java bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) via JNI, providing a high-level API for LLM inference in Java. The Java layer communicates with a native C++ library through JNI.

Current llama.cpp pinned version: **b9739**

## Upgrading CUDA Version

Current CUDA version: **13.2**

To change the CUDA version, update the following **three** places:

1. **`.github/build_cuda_linux.sh`** — Line 10: `sudo dnf install -y cuda-toolkit-13-2`
2. **`.github/build_cuda_linux.sh`** — Line 12: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc`
3. **`pom.xml`** — The `<classifier>` tag in the `cuda` jar execution: `cuda13-linux-x86-64`

Also update the header comment in `build_cuda_linux.sh` and the job name in `.github/workflows/release.yaml` for clarity.

Available CUDA versions for RHEL8/Manylinux_2_28 can be browsed at:
```
https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/
```

**Note:** Each CUDA version supports only certain GCC versions. If the dockcross container uses a newer GCC than CUDA supports, the build will fail with `unsupported GNU version`. Check NVIDIA's compatibility table before downgrading CUDA.

Example: To upgrade from 13.2 to a hypothetical 13.3:
```bash
# Edit .github/build_cuda_linux.sh:
#   line 10: cuda-toolkit-13-2 -> cuda-toolkit-13-3
#   line 12: /usr/local/cuda-13.2/bin/nvcc -> /usr/local/cuda-13.3/bin/nvcc
# Edit pom.xml classifier: cuda13-linux-x86-64 (major version only, no need to change for minor bumps)
# Edit CLAUDE.md line: Current CUDA version: **13.2** -> **13.3**
git add .github/build_cuda_linux.sh pom.xml CLAUDE.md
git commit -m "Upgrade CUDA from 13.2 to 13.3"
```

### Fast local CUDA builds (`CUDA_FAST_BUILD`) — single-arch speed knob

The CUDA artifact must ship kernels for **every supported GPU generation**, so the default
build — and every CI/release build — compiles the **full `CMAKE_CUDA_ARCHITECTURES` set** that
ggml/llama.cpp selects. nvcc recompiles each `.cu` kernel once per architecture, which is the
dominant cost of the ~70 min CUDA job. **`sccache` does not help here:** it caches the gcc
C/C++ TUs but not the nvcc `.cu` kernels (sccache's nvcc support is limited/experimental), so
the per-arch nvcc passes remain even with the cache on. The one reliable lever to cut that time
is to build **fewer architectures**.

`build_cuda_linux.sh` therefore honors an **opt-in** env knob — default **off** (full arch set,
release-safe):

```bash
# Full release build (default): all archs — slow, runs on every GPU generation.
.github/build_cuda_linux.sh "-DOS_NAME=Linux -DOS_ARCH=x86_64"

# Fast local dev build: one arch only. Defaults to `native` (the build machine's own GPU;
# needs a GPU present at configure time). Override with CUDA_ARCH=<cc>, e.g. CUDA_ARCH=90.
CUDA_FAST_BUILD=1 .github/build_cuda_linux.sh "-DOS_NAME=Linux -DOS_ARCH=x86_64"
CUDA_FAST_BUILD=1 CUDA_ARCH=90 .github/build_cuda_linux.sh "-DOS_NAME=Linux -DOS_ARCH=x86_64"
# Direct-cmake equivalent: cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
```

**Default + CI policy (release-safety is the invariant).** An artifact built with `CUDA_FAST_BUILD`
runs on only the single GPU generation it was compiled for, so the **distributed jar must always be
the full arch set**. The script default is **off** (full) so any *local/manual* build is
release-safe. In CI (`publish.yml`, the `crosscompile-linux-x86_64-cuda` job) the flag is **on for
validation runs** (PR / push / non-publish dispatch) to cut nvcc time, and **off only when actually
publishing to Central** — it is wired as `CUDA_FAST_BUILD: ${{ inputs.publish_to_central && '0' || '1' }}`
(`'0'`=full, `'1'`=fast). Because the `publish-snapshot`/`publish-release` jobs require
`publish_to_central`, **every artifact that reaches Central is built with the full arch set** while
ordinary PR/push CI stays fast. CI has no GPU, so the fast path pins a fixed `CUDA_ARCH` (default
`120` — the newest CUDA 13.2 arch, sm_120 / consumer Blackwell — in the job env) — `native`
would fail at configure. Both `CUDA_FAST_BUILD` and `CUDA_ARCH` are
forwarded into the dockcross container via `DOCKCROSS_ARGS` `-e`. To cache the nvcc kernels too you
would add `-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache` (gated behind the same probe), but sccache's nvcc
caching is unreliable — the arch knob is the better lever and is what this repo ships.

## Android minimum API level

Current Android minimum API level: **28** (Android 9.0 Pie)

This is enforced through bionic's **weak-symbol** mechanism, *not* by bumping
`__ANDROID_API__` or passing `-DANDROID_PLATFORM`. See "How the API gate is
satisfied" below for why. To change anything here, update:

1. **`CMakeLists.txt`** — the `add_compile_definitions(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__)`
   block and its Android-detection guard (`OS_NAME MATCHES "Android"` etc.).
2. **`CLAUDE.md`** (this file) — the "Current Android minimum API level" line above.
3. **`README.md`** — the minimum-API note (the `[!NOTE]` block near the Android
   classifier entries and the "Importing in Android" section).

**Why API 28?** `mtmd-helper.cpp` (part of the upstream llama.cpp `mtmd`
multimodal library) includes `vendor/sheredom/subprocess.h`, which calls
`posix_spawn`, `posix_spawnp`, and `posix_spawn_file_actions_*`. Bionic only
exposes those `<spawn.h>` declarations once the minimum SDK is ≥ 28 (and
`getifaddrs`/`freeifaddrs` in `<ifaddrs.h>`, used by cpp-httplib, at ≥ 24). The
symbols exist in `libc.so` at all API levels; bionic only hides the
*declarations* below the introducing API.

**How the API gate is satisfied (important — the obvious fixes do not work).**
The CI cross-compiler is the `dockcross-android-arm64` image, which is **not**
the Google NDK CMake toolchain — it is a Debian-style cross-clang at
`/usr/aarch64-linux-android/bin/clang`. Consequently:

- It never sets the `ANDROID` / `ANDROID_ABI` CMake variables, so any
  `if(ANDROID_ABI)`-guarded logic silently does nothing.
- It **ignores** `-DANDROID_PLATFORM=android-28` (CMake prints it as a
  "Manually-specified variables were not used by the project" warning).
- `clang` predefines `__ANDROID_API__` from its baked-in target triple, so
  `-D__ANDROID_API__=28` would only clash with the builtin (`-Wmacro-redefined`)
  and would *not* move `__ANDROID_MIN_SDK_VERSION__`, which is what bionic's
  `__BIONIC_AVAILABILITY_GUARD(api)` actually tests.

The working fix is `add_compile_definitions(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__)`
for the Android build. That macro forces `__BIONIC_AVAILABILITY_GUARD(api)` to
`1` for every API level (declarations always visible) and makes any symbol newer
than the toolchain's baked-in min-SDK a **weak** reference resolved by the
dynamic linker at load time — present on every API-28+ device the artifact
targets. It is never compiler-predefined, so defining it is clean. The guard
detects Android via `OS_NAME MATCHES "Android"` (CI passes
`-DOS_NAME=Linux-Android`) and the compiler path, not `ANDROID_ABI`.

## OpenCL / Adreno backend on Android

A second Android arm64 artifact is built with the OpenCL backend enabled and
Adreno-tuned kernels embedded. It ships under the Maven classifier
`opencl-android-aarch64` and is consumed only when callers explicitly request it.
The default Android arm64 JAR remains CPU-only.

Three places wire it together (mirrors the CUDA classifier pattern):

1. **`CMakeLists.txt`** — `elseif(GGML_OPENCL)` branch routes artifacts to
   `src/main/resources_android_opencl/net/ladenthin/llama/${OS_NAME}/${OS_ARCH}/`.
2. **`.github/workflows/publish.yml`** — `crosscompile-android-aarch64-opencl`
   job runs the dockcross-android-arm64 build with
   `-DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON`
   and uploads as artifact `android-libraries-opencl`. The `package`,
   `publish-snapshot`, and `publish-release` jobs download it into
   `resources_android_opencl/` and activate the `opencl-android` Maven profile.
3. **`pom.xml`** — the `opencl-android` profile produces a second JAR with
   `<classifier>opencl-android-aarch64</classifier>` from the
   `${project.build.outputDirectory}_opencl_android` tree.

Local sanity build:
```bash
.github/dockcross/dockcross-android-arm64 .github/build_opencl_android.sh \
  "-DOS_NAME=Linux-Android -DOS_ARCH=aarch64 \
   -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
   -DGGML_OPENCL_USE_ADRENO_KERNELS=ON"
```
Artifacts land in `src/main/resources_android_opencl/net/ladenthin/llama/Linux-Android/aarch64/`.

The dockcross image does not ship OpenCL headers or a stub `libOpenCL.so`, so
`build_opencl_android.sh` first stages Khronos `OpenCL-Headers` and
cross-builds `OpenCL-ICD-Loader` into `/tmp/opencl-stage/` before invoking the
main project cmake with `-DOpenCL_INCLUDE_DIR=...` and `-DOpenCL_LIBRARY=...`.
At runtime the device must provide its own OpenCL ICD (`libOpenCL.so`);
Qualcomm Adreno drivers do. Devices without an ICD should use the default
CPU-only Android JAR.

## Windows Ninja artifact (sccache-cached, parallel to the MSVC build)

The Visual Studio generator ignores `CMAKE_{C,CXX}_COMPILER_LAUNCHER`, so the two MSVC Windows
jobs (`build-windows-x86_64`, `build-windows-x86`) **cannot** use the sccache/Depot cache. Rather
than switch the trusted MSVC build, the repo builds the **same CPU natives a second time** with the
**`Ninja Multi-Config`** generator (which *does* honor the launcher) and ships them as a separate
**`ninja-windows`** Maven classifier JAR. **The MSVC build is the default JAR and is kept
permanently** — the Ninja artifact is an additional, cache-accelerated, independently
end-to-end-tested option, not a replacement. (Upstream llama.cpp ships its `windows-cuda` artifact
with Ninja Multi-Config + MSVC, proving the combination works on the same tree.)

Unlike the CUDA / OpenCL classifiers — which differ by a **GGML backend flag** and route their
output in `CMakeLists.txt` — the Ninja Windows build differs only by **generator/toolchain**, so
there is **no `CMakeLists.txt` change**: both generators emit to the canonical
`src/main/resources/.../Windows/{x86_64,x86}/`. Routing to the classifier tree happens purely at the
CI-download + pom-profile level. Four places wire it together:

1. **`.github/build.bat`** — sccache probe guard mirroring `build.sh`'s `sccache_can_wrap_compiler()`:
   when `USE_CACHE=true` and `sccache` is on PATH, it compiles a trivial TU through `sccache cl.exe`;
   only on success does it pass `-DCMAKE_{C,CXX}_COMPILER_LAUNCHER=sccache` and print
   `sccache --show-stats`. A missing/crashing sccache falls back to a green uncached build. The MSVC
   jobs do not set `USE_CACHE`, so the guard is inert for them.
2. **`.github/workflows/publish.yml`** — build jobs `build-windows-x86_64-ninja` /
   `build-windows-x86-ninja` (`windows-2025-vs2026`, `ilammy/msvc-dev-cmd@v1` for the arch env,
   sccache v0.16.0 from the GitHub release **zip** + Depot WebDAV, `build.bat -G "Ninja Multi-Config"`),
   uploading artifacts `Windows-{x86_64,x86}-ninja` (**not** `*-libraries`, so the `package` job's
   `pattern: "*-libraries"` ignores them). `test-java-windows-x86_64-ninja` loads the Ninja DLL via
   JNI and runs the full model-backed suite. The `package`, `publish-snapshot`, and `publish-release`
   jobs download `Windows-*-ninja` into `src/main/resources_windows_ninja/` and activate the
   `windows-ninja` Maven profile.
3. **`pom.xml`** — the `windows-ninja` profile produces a second JAR with `<classifier>ninja-windows</classifier>`
   from the `${project.build.outputDirectory}_windows_ninja` tree (separate compile pass + resource
   copy + classified jar; mirrors the `cuda` / `opencl-android` profiles). Activated only in CI.
4. **`README.md`** — the `ninja-windows` row + dependency snippet in "Choosing the right classifier".

`src/main/resources_windows_ninja/` is git-ignored (staged by CI, never committed — same policy as
the native libs and the CUDA/OpenCL trees).

**Local sanity build** (needs MSVC + a Ninja on PATH; sccache optional):
```bat
mvn -q compile
.github\build.bat -G "Ninja Multi-Config" -DOS_NAME=Windows -DOS_ARCH=x86_64 -DBUILD_TESTING=ON
ctest --test-dir build --output-on-failure
```

## WebUI (llama.cpp Svelte UI) embedding

The llama.cpp WebUI is **built once in CI and shared to every native build**, then
compiled into `libjllama` so the embedded server (`server-http.cpp`) can serve it.
This repo commits no build outputs, so the assets are produced per-pipeline, never
checked in (same policy as the native libs).

Pipeline (`.github/workflows/publish.yml`):

1. **`build-webui` job** (ubuntu — the *only* job that runs `npm`): resolves the
   pinned `b<nnnn>` tag from `CMakeLists.txt`'s `GIT_TAG`, sparse-checks-out
   `ggml-org/llama.cpp@<tag>` `tools/ui`, runs the upstream Svelte build
   (`npm ci && npm run build`), gzips `dist/` into `dist/_gzip/` (LLAMA_UI_GZIP
   parity), builds the self-contained `llama-ui-embed` host tool (plain C++17, **no
   npm**) and runs it to produce the platform-independent **`webui-generated/ui.cpp`
   + `ui.h`**, uploaded as the `webui-generated` artifact.
2. **Every native build job** (`needs: [startgate, build-webui]`) downloads that
   artifact into `webui-generated/` before building. npm never runs in the dockcross
   cross-compilers (which have no node) or per-platform.
3. **CMake** (the "WebUI assets" block in `CMakeLists.txt`): if
   `webui-generated/ui.cpp` + `ui.h` exist, compiles `ui.cpp` in and adds its dir to
   the include path — the generated `ui.h` `#define`s `LLAMA_UI_HAS_ASSETS`, which
   activates `server-http.cpp`'s static-asset routes. If absent, it falls back to the
   empty-asset stub `src/main/cpp/webui_stub/ui.h` (no embedded UI) so local builds —
   and any job without the artifact — still build and run.

The WebUI version **auto-follows** the pinned `GIT_TAG`: a llama.cpp version bump
needs no extra step here, `build-webui` re-reads the tag and rebuilds the matching UI.

**Building the WebUI locally** (optional — a plain `cmake` build uses the stub and
ships no UI):
```bash
# needs node/npm + network; embed.cpp is plain C++17 (no npm)
git clone --depth 1 --branch b9739 https://github.com/ggml-org/llama.cpp /tmp/lc
( cd /tmp/lc/tools/ui && npm ci && npm run build \
  && ( cd dist && find . -type f -not -path './_gzip/*' \
       | while read -r f; do mkdir -p "_gzip/$(dirname "$f")"; gzip -9 -c "$f" > "_gzip/$f"; done ) \
  && g++ -O2 -std=c++17 -o /tmp/llama-ui-embed embed.cpp )
mkdir -p webui-generated
/tmp/llama-ui-embed webui-generated/ui.cpp webui-generated/ui.h /tmp/lc/tools/ui/dist
cmake -B build && cmake --build build --target jllama   # now embeds the real UI
```
`webui-generated/` is git-ignored.

## CI build cache & parallelism (sccache + Depot)

The native build dominates CI time (134 llama.cpp model TUs + ggml + the 16.6k-line
`httplib.cpp`, all at `-O3`). Two knobs in **`.github/build.sh`**, both behind the
`use_cache` `workflow_dispatch` input (default **true**), keep it fast and stop the macOS
runners OOM-ing.

**`BUILD_JOBS` — compile parallelism.** `build.sh` builds with `cmake --build -j${BUILD_JOBS}`
(default: all cores, via portable `nproc` → `sysctl -n hw.ncpu` → `4` detection). GitHub's
~7 GB **macOS arm64** runners OOM under full `-j` when `httplib.cpp` co-schedules with the
model TUs; the runner is then killed as **SIGTERM / exit 143** ("received a shutdown
signal"), which *looks* like a timeout but is an out-of-memory kill. The three macOS build
jobs therefore set `BUILD_JOBS: 2` to bound peak memory.

**`sccache` → Depot Cache — shared compiler cache.** When `USE_CACHE=true` **and** `sccache`
plus a cache token are present, `build.sh` adds
`-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache` and prints
`sccache --show-stats`. The cache lives in **Depot Cache** over sccache's **WebDAV** backend:

- `SCCACHE_WEBDAV_ENDPOINT: https://cache.depot.dev`
- `SCCACHE_WEBDAV_TOKEN: ${{ secrets.DEPOT_TOKEN }}` — a Depot **organization** token, stored
  as the repo secret **`DEPOT_TOKEN`**.

Because `sccache` is **content-addressed** and llama.cpp is pinned (`GIT_TAG b9739`), the
~280 upstream object files are byte-identical every run, so a warm cache recompiles only the
*changed* files. Depot's cache is **shared across all branches** (unlike GitHub's
per-branch `actions/cache`), so every branch builds incrementally; a `b<nnnn>` version bump
naturally invalidates the upstream entries (their content changed) with no manual step. It
stays `-O3` and is **bit-identical** to a clean build (release-safe).

**Safety / transparency.** It is **inert** until `DEPOT_TOKEN` is configured and on **fork
PRs** (secrets are hidden there) — those simply compile normally; the `Install sccache` step
is `continue-on-error`; and `use_cache=false` forces a pristine, from-scratch build. Crucially,
`build.sh` runs a **probe-compile health-check** (`sccache_can_wrap_compiler`) before trusting
sccache as the launcher: it compiles a trivial TU *through* sccache, and only sets
`-DCMAKE_{C,CXX}_COMPILER_LAUNCHER=sccache` if that succeeds. So a sccache that is present but
**crashes** (the in-container panic that stalled phase 2) also falls back to an uncached, green
`-O3` build — it logs the Rust panic backtrace (and the detached server's `SCCACHE_ERROR_LOG`,
when a job sets one) for diagnosis but never reds the build. This closes the gap the original
absent-only guard left.

**The fork-PR `.sccache_check` 403 (mac-only symptom) and its two guards.** A fork PR (e.g.
`vaiju1981/java-llama.cpp` → upstream) runs with secrets withheld, so `SCCACHE_WEBDAV_TOKEN`
(`= secrets.DEPOT_TOKEN`) is **empty**. Depot rejects the unauthenticated server-startup
`.sccache_check` with **403 Forbidden** (`PermissionDenied (temporary) … Forbidden`), and
because sccache treats a failed startup check as fatal, *every* TU dies. The symptom looked
**mac-only** purely because of an asymmetry in how sccache reaches `PATH`: the macOS jobs ran
`brew install sccache` **unconditionally** (`if: USE_CACHE == 'true'`), whereas the
Linux/dockcross/aarch64 jobs only **fetch** sccache when a token is present (the `[ -n
"$SCCACHE_WEBDAV_TOKEN…" ]` guard in `build.sh`'s fetch block) — so on a tokenless fork PR
mac was the only platform with sccache on `PATH` to misfire. Two independent guards now prevent
it: **(1)** every `Install sccache` step is gated `if: env.USE_CACHE == 'true' && env.SCCACHE_WEBDAV_TOKEN
!= ''`, so a tokenless fork PR never even installs sccache (mac now matches Linux); and **(2)**
`build.sh`'s build step **retries once without the launcher** when the build fails *and* the
output shows an sccache cache error (`sccache: error` / `Server startup failed` / `cache storage
failed`) — a clean uncached `-O3` rebuild that is content-identical and release-safe. The retry
is gated on that error signature so a genuine compile error still fails fast and is reported
(no wasteful uncached rebuild). Guard (2) also covers an *intermittent* 403 that strikes a
valid-token job mid-build, which the one-shot probe cannot foresee.

**Rollout.** **Phase 1 — DONE & proven: the 3 macOS build jobs** (slowest + OOM-prone) —
`brew install sccache` + the env above + `BUILD_JOBS: 2`. macOS build dropped **~40 min → ~6 min**
with a warm cache. **Phase 2 — DONE: all 5 dockcross cross-compile jobs** now have the same
steady-state env (`USE_CACHE` + `SCCACHE_WEBDAV_*` + `DOCKCROSS_ARGS`). The probe makes it safe
to enable them all at once — any container where sccache crashes falls back to an uncached green
build automatically. (The first attempt enabled all four at once without the probe and was
reverted: the static-musl sccache v0.8.2 panicked in-container and redded the build. With
v0.16.0 + the probe this is no longer a risk.) Job-by-job status:
1. `crosscompile-linux-x86_64` (manylinux2014) — ✅ **verified green** in PR #245: sccache
   **v0.16.0** probe passed in-container (devtoolset-10 gcc), `sccache ON` over Depot WebDAV,
   warm cache 277/278 hits (99.64%), 1m46s build time.
2. `crosscompile-linux-x86_64-cuda` (via `build_cuda_linux.sh`, which execs `build.sh`) —
   🚧 **first run in progress** (diagnostics on). Only the gcc C/C++ TUs cache (134 model files
   + ggml + httplib); the nvcc `.cu` kernels won't (limited sccache nvcc support) — still a
   large partial win on the ~70 min full-arch job; the fast single-arch (sm_120) validation path
   cuts nvcc time independently of sccache.
3. `crosscompile-linux-aarch64` — ✅ **enabled**, now a **native `ubuntu-24.04-arm` build** (not
   dockcross): `build.sh` self-fetches the aarch64 static-musl sccache (the fetch block in
   `build.sh` maps `uname -m` → `x86_64`/`aarch64`) and the probe guards it. See "Linux aarch64:
   native ARM build" below for why it moved off the cross-compiler.
4. `crosscompile-android-aarch64` — ✅ **enabled** (same steady-state env; probe guards it).
5. `crosscompile-android-aarch64-opencl` — ✅ **enabled**. `build_opencl_android.sh` stages the
   OpenCL headers/loader, then delegates the jllama cmake build to `build.sh` via `exec`
   (same pattern as `build_cuda_linux.sh`), so it inherits the probe and launcher automatically.

Per-job recipe: add `env:` { `USE_CACHE`, `SCCACHE_WEBDAV_ENDPOINT`, `SCCACHE_WEBDAV_TOKEN` } and
`DOCKCROSS_ARGS: "-e SCCACHE_WEBDAV_ENDPOINT -e SCCACHE_WEBDAV_TOKEN -e USE_CACHE"` — the
dockcross wrapper only forwards host env it is explicitly told to via `-e`. The fetched sccache
version is the `SCCACHE_DL_VERSION` knob in `build.sh` (default **0.16.0**; overridable per-job
to try a different build against a container that crashed another). **Windows** is handled
separately (the Visual Studio generator ignores `CMAKE_*_COMPILER_LAUNCHER`): see
"Windows Ninja artifact" below — the cached path uses the **Ninja Multi-Config** generator with a
`build.bat` sccache probe and a direct sccache zip download (not `mozilla-actions/sccache-action`),
shipped as a parallel `ninja-windows` classifier JAR while the MSVC default stays the trusted build.

**Cross-repo scope.** This Depot/sccache compiler cache makes sense only for java-llama.cpp —
it is the only sibling repo with a native (C++/JNI) build. It does not apply to the pure-Maven
siblings; why (and why the `DEPOT_TOKEN` org secret and the README "Build cache by Depot" badge
are kept jllama-only) is explained in the cross-repo status under "Deliberate non-parity":
[`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md).

## Local llama.cpp source patches (`patches/`)

The fetched llama.cpp source is patched before it compiles, via a generic mechanism:

- **`patches/`** (repo root) — drop any number of `*.patch` / `*.diff` files here. They are applied
  in **filename order** (use a numeric prefix, e.g. `0001-`, `0002-`), so keep them independent or
  ordered. Each must be a `git apply`-compatible unified diff with paths relative to the llama.cpp
  source root (`a/common/arg.cpp` / `b/common/arg.cpp`, i.e. `-p1`).
- **`cmake/apply-llama-patches.cmake`** — the applier. Cross-platform (`cmake -P`, so identical on
  Linux/macOS/Windows), **idempotent** (`git apply --reverse --check` skips already-applied patches
  so a reconfigure never double-applies) and **fail-loud** (a patch that no longer applies aborts
  the configure — a stale patch can't be silently dropped from a release build).
- **`CMakeLists.txt`** — wired as the llama.cpp `FetchContent_Declare(... PATCH_COMMAND ...)`, so it
  runs for **every** C++ build (all CI jobs *and* local `cmake -B build`) from one place — no
  per-build-step plumbing.

**On a llama.cpp version bump, every patch must still apply** — if a bump shifts the patched code,
the configure fails with an "does not apply cleanly" error; refresh the diff against the new source
and recommit. Treat `patches/` as part of the upgrade checklist below.

Current patches:

| Patch | Fixes |
|-------|-------|
| `0001-win32-arg-parse-embed-guard.patch` | Windows JNI regression from llama.cpp **#24779** (b9739): `common_params_parse` unconditionally replaced the caller's argv with the process command line (`GetCommandLineW`), so an embedded/JNI caller (`java.exe`) lost its `--model …` args → "Failed to parse model parameters". The patch **drops the override for our build** (keeps the `make_utf8_argv()` call referenced so there's no `-Wunused-function`, but never adopts its result), so the caller's already-UTF-8 argv is always used. This is **deterministic** — an earlier count-guard variant (only override when the re-derived arg count equals `argc`) collided on the server-integration tests whose argv length happened to equal `java.exe`'s and kept them failing. The upstream PR can instead expose an opt-out / `common_params_parse_argv` that preserves the standalone tools' UTF-8 fix. |

## Upgrading/Downgrading llama.cpp Version

To change the llama.cpp version, update the following **three** files (and re-verify `patches/`):

1. **CMakeLists.txt** — the `GIT_TAG` line for llama.cpp: `GIT_TAG        b8831`
2. **README.md** — the badge and link line with the version number
3. **CLAUDE.md** — the "Current llama.cpp pinned version" line

Example: To upgrade from b8808 to b8831:
```bash
# Edit CMakeLists.txt: change GIT_TAG b8808 to b8831
# Edit README.md: change b8808 to b8831 (in both badge and link)
# Edit CLAUDE.md: change b8808 to b8831
git add CMakeLists.txt README.md CLAUDE.md
git commit -m "Upgrade llama.cpp from b8808 to b8831"
git push -u origin <your-branch>
```

**Note:** Always test the build with `cmake -B build && cmake --build build --config Release` after version changes to catch compatibility issues early.

### Inspecting API changes between versions

Use the GitHub compare URL to diff any two llama.cpp builds:

```
https://github.com/ggml-org/llama.cpp/compare/b<FROM>...b<TO>
```

Example — what changed between b6721 and b6732:
```
https://github.com/ggml-org/llama.cpp/compare/b6721...b6732
```

The GitHub HTML page may time out for large ranges; fall back to the API:
```
https://api.github.com/repos/ggml-org/llama.cpp/compare/b<FROM>...b<TO>
```

For individual file content at a specific build:
```
https://raw.githubusercontent.com/ggerganov/llama.cpp/b<VERSION>/common/chat.h
```

### Files to check for API compatibility

The three project C++ files (`jllama.cpp`, `server.hpp`, `utils.hpp`) pull in the following
llama.cpp headers. Any of these can introduce breaking changes on upgrade.

**Include dependency graph:**
```
jllama.cpp / server.hpp / utils.hpp
│
├── arg.h ──────────────────────────► common.h ─┐
├── common.h ──────────────────────────────────►├── ggml-opt.h ──► ggml.h
├── chat.h ─────────────► common.h, peg-parser.h └── ggml-backend.h ──► ggml-alloc.h
├── speculative.h ──────► llama.h, common.h
├── sampling.h ─────────► llama.h, common.h
├── download.h ─────────► (stdlib only, no deps)
├── log.h ──────────────► ggml.h
├── llama.h ────────────────────────────────────► ggml.h, ggml-cpu.h, ggml-backend.h, ggml-opt.h
│                                                  └── llama-cpp.h ──► llama.h
├── json-schema-to-grammar.h
├── base64.hpp
├── mtmd.h
└── mtmd-helper.h
```

**Priority-ordered review list for upgrade diffs** (highest break risk first)

The top 8 rows cover all known API-level breaking changes from b5022 → b8831.
For future upgrades, provide diffs for at least these 8 files rather than the full patch.
Also review the project `CMakeLists.txt` for build-system-level breaks (e.g. renamed link targets, new required headers) — those are not visible in header file diffs alone.

| File | What to watch for |
|------|-------------------|
| `common/common.h` | `common_params`/`common_params_speculative` struct fields, `model_alias` container type, `common_init_result` shape, `build_info` symbol (removed in b8831 — now `llama_build_info()` from `build-info.h`) |
| `common/chat.h` | `common_chat_parser_params` (was `common_chat_syntax`), `to_json_oaicompat`, `common_chat_msg_diff_to_json_oaicompat`, `set_tool_call_ids` |
| `common/speculative.h` | `common_speculative_init`, `common_speculative_draft`, `common_speculative_accept` signatures, struct names |
| `tools/mtmd/mtmd.h` | `mtmd_context_params` fields, `image_marker`/`media_marker` API, deprecated symbols (was `common/mtmd.h` before ~b8190) |
| `include/llama-cpp.h` | `common_init_result_ptr` type, access pattern changes (`.get()` vs `->method()`) |
| `common/arg.h` | `n_parallel` sentinel value, what moved to `download.h` across versions |
| `include/llama.h` | Core llama_ function signatures, token types, `llama_model_ptr`, renamed structs |
| `common/download.h` | `common_remote_params` struct, `headers` field format (string vs key-value pair) |
| `common/common.cpp` | Implementation of any inline API used directly |
| `common/speculative.cpp` | Speculative decoding implementation details |
| `common/chat.cpp` | Chat parsing implementation |
| `common/sampling.h` | Sampler API, `common_sampler_*` functions |
| `common/log.h` | Log macro signatures |
| `tools/mtmd/mtmd-helper.h` | Multimodal helper functions |
| `common/json-schema-to-grammar.h` | Grammar API |
| `ggml/include/ggml.h` | `ggml_type` enum values (e.g. `GGML_TYPE_F16`), tensor primitives |
| `ggml/include/ggml-backend.h` | Backend/device abstraction types |
| `ggml/include/ggml-opt.h` | Optimizer params pulled in via `common.h` |

**Safe to skip** (have never caused a break; not used directly by project code):
`common/sampling.h`, `common/log.h`, `tools/mtmd/mtmd-helper.h`, `common/json-schema-to-grammar.h`,
`ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, `ggml/include/ggml-opt.h`,
`ggml-alloc.h`, `ggml-cpu.h`, `peg-parser.h`, `base64.hpp`

For the full record of upstream API breaks across version ranges (b5022 &#x2192; current), including which rows required project source changes vs. which stayed inside upstream-compiled translation units, see [`docs/history/llama-cpp-breaking-changes.md`](docs/history/llama-cpp-breaking-changes.md). When bumping the `llama.cpp` version, append a new row to that file covering the upgrade range.

## Build Commands

### Java (Maven)
```bash
mvn compile          # Compiles Java and generates JNI headers
mvn test             # Run all tests (requires native library and model files)
mvn package          # Build JAR
mvn -P assembly package  # Also build the fat jar-with-dependencies uber JAR (library + Java deps + native libs); CI builds it and uploads it in the `llama-jars` artifact
mvn test -Dtest=LlamaModelTest#testGenerate  # Run a single test method
```

### Native Library (CMake)
Must run `mvn compile` first to generate JNI headers, then:
```bash
# CPU only
cmake -B build
cmake --build build --config Release

# CUDA (Linux)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Metal (macOS)
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release

# Optional: enable model downloading via URL
cmake -B build -DLLAMA_CURL=ON
```

Built libraries are placed in `src/main/resources/net/ladenthin/llama/{OS}/{ARCH}/`.

### Building the native library for local Java tests

`mvn test` does **not** build the native library — Maven only compiles Java
and runs surefire. The shared library must already exist on disk under the
platform-specific resource path that `LlamaLoader` resolves at runtime.
Without it the JVM throws `UnsatisfiedLinkError` and every Java test fails
immediately (it does not auto-skip).

The output path is derived by `CMakeLists.txt` from `OS_NAME` and `OS_ARCH`
detected by the helper script `.github/dockcross/dockcross-resolve-host`
(falls back to `uname` on hosts where the script is absent). The mapping
mirrors `OSInfo.translateOSNameToFolderName` on the Java side, so the same
folder name is produced on both ends.

| Host | Library file | Resource path produced by `cmake --build` |
|------|--------------|-------------------------------------------|
| Linux x86_64 | `libjllama.so` | `src/main/resources/net/ladenthin/llama/Linux/x86_64/` |
| Linux aarch64 | `libjllama.so` | `src/main/resources/net/ladenthin/llama/Linux/aarch64/` |
| macOS Apple Silicon | `libjllama.dylib` | `src/main/resources/net/ladenthin/llama/Mac/aarch64/` |
| macOS Intel | `libjllama.dylib` | `src/main/resources/net/ladenthin/llama/Mac/x86_64/` |
| Windows x86_64 | `jllama.dll` (+ `llama.dll`, `ggml.dll`) | `src/main/resources/net/ladenthin/llama/Windows/x86_64/` |

The Windows `RUNTIME_OUTPUT_DIRECTORY_*` properties (`CMakeLists.txt:266-269`)
deposit `jllama.dll` alongside the upstream `llama.dll` / `ggml.dll`; all
three must remain co-located so the loader can resolve transitive imports.

End-to-end local workflow for running Java tests:

```bash
# 1. Generate JNI headers (one-time per Java API change)
mvn -q compile

# 2. Configure + build the native library for the current host
cmake -B build
cmake --build build --config Release -j$(nproc)
# The shared lib lands directly in src/main/resources/.../{OS}/{ARCH}/ —
# no separate install step is needed.

# 3. Ensure model files referenced by tests are present under models/.
#    The default test models (downloaded by CI in publish.yml) are:
curl -L --fail "$MODEL_URL"          --create-dirs -o models/codellama-7b.Q2_K.gguf
curl -L --fail "$RERANKING_MODEL_URL" --create-dirs -o models/jina-reranker-v1-tiny-en-Q4_0.gguf
curl -L --fail "$DRAFT_MODEL_URL"     --create-dirs -o models/AMD-Llama-135m-code.Q2_K.gguf
curl -L --fail "$REASONING_MODEL_URL" --create-dirs -o models/Qwen3-0.6B-Q4_K_M.gguf

# 4. Run tests. Tests that need a model file self-skip via Assume.assumeTrue()
#    when their GGUF is absent, so partial model availability is OK.
mvn test
# CPU-only host (no GPU): pin GPU layers to 0
mvn test -Dnet.ladenthin.llama.test.ngl=0
# Run a single test class or method
mvn test -Dtest=MemoryManagementTest
mvn test -Dtest=LlamaModelTest#testGenerateAnswer
```

**Optional models** referenced by individual tests are gated on a system
property so CI can skip them cleanly when the GGUF is not downloaded.
The full property → consumer → default table for every `net.ladenthin.llama.*`
property the library understands (runtime + test) is the user-facing
**[System Properties Reference](README.md#system-properties-reference)** in
the README. The summary below covers only the optional-model bindings:

| Property | Default test that uses it | Model |
|----------|---------------------------|-------|
| `net.ladenthin.llama.nomic.path` | `LlamaEmbeddingsTest#testNomicEmbedLoads` | `nomic-embed-text-v1.5.f16.gguf` (issue #98 regression) |
| `net.ladenthin.llama.vision.model` | `MultimodalIntegrationTest` (upstream kherud/java-llama.cpp#103 / #34) | `SmolVLM-500M-Instruct-Q8_0.gguf` (any vision-capable GGUF works) |
| `net.ladenthin.llama.vision.mmproj` | `MultimodalIntegrationTest` | matching mmproj for the vision model, e.g. `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` |
| `net.ladenthin.llama.vision.image` | `MultimodalIntegrationTest` | committed default `src/test/resources/images/test-image.jpg`; override to any png/jpeg/webp/gif on disk |

Run those tests by setting the property:
```bash
mvn test -Dtest=LlamaEmbeddingsTest#testNomicEmbedLoads \
         -Dnet.ladenthin.llama.nomic.path=models/nomic-embed-text-v1.5.f16.gguf
mvn test -Dtest=MultimodalIntegrationTest \
         -Dnet.ladenthin.llama.vision.model=models/SmolVLM-500M-Instruct-Q8_0.gguf \
         -Dnet.ladenthin.llama.vision.mmproj=models/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf
# The vision.image property defaults to src/test/resources/images/test-image.jpg
# (a CC-BY-4.0 / MIT-granted photo of flowers and bees by the project author);
# override only if you want to test a different image.
```

`MultimodalIntegrationTest` self-skips when any of the three vision properties
points at a missing path, so a partial setup (just the vision model + the
committed image, no mmproj) lets the test class load without erroring.

**Restricted-network environments.** Some hosts (e.g. ephemeral remote
execution sandboxes) block outbound traffic to `huggingface.co`. In that
case downloading models for the Java tests is not possible from the host
itself; the native library can still be built and the C++ test suite
(`ctest --test-dir build`) still runs because it depends only on the
upstream sources fetched at CMake configure time. Java tests should then
be exercised either in CI (via `.github/workflows/publish.yml`) or on a
developer machine with HF access; pre-staged models can also be uploaded
into `models/` out-of-band.

**Verifying the native library *loads* without models (model-free smoke).**
Even with HuggingFace blocked you can still do the one piece of *real native*
verification that does not need a GGUF: confirm the library loads and its
`JNI_OnLoad` resolves every Java class it looks up by name. The model-gated
tests cannot do this in a restricted sandbox — they self-skip via
`Assume.assumeTrue(model present)` **before** the lib is ever loaded, so a plain
`mvn test` is silent on load-time breakage. The full local recipe:

```bash
# 1. Build the native lib locally (FetchContent pulls llama.cpp from GitHub,
#    which is reachable even when huggingface.co is not):
mvn -q compile
cmake -B build -DBUILD_TESTING=ON
cmake --build build --config Release -j$(nproc)   # -> src/main/resources/.../<os>/<arch>/libjllama.so
# 2. Force LlamaModel.<clinit> (System.load -> JNI_OnLoad) with no model:
mvn test -Dtest=NativeLibraryLoadSmokeTest
```

`NativeLibraryLoadSmokeTest` (in the `loader` package) calls
`Class.forName("net.ladenthin.llama.LlamaModel")`, which runs
`LlamaLoader.initialize() -> System.load() -> JNI_OnLoad`, which in turn calls
`FindClass(...)` for every JNI-referenced Java class. It **passes** when the lib
loads cleanly, **fails** if the native-resource path in `LlamaLoader` is wrong
(lib not found) or a `FindClass`/field-signature FQN in
`src/main/cpp/jllama.cpp` is stale after a Java package move (lib loads but
`JNI_OnLoad` throws `NoClassDefFoundError: net/ladenthin/llama/...`), and
**self-skips** when `libjllama` is not on the classpath (pure-Java checkout, no
CMake build) so it never breaks a build-less `mvn test`.

Both of those failure modes shipped on a branch once — the layered-package
restructure left (a) `LlamaLoader.getNativeResourcePath()` deriving the resource
root from the loader's own package (which moved to `…loader`) and (b)
`jllama.cpp` still `FindClass`-ing the old flat paths — and neither was visible
to a local `mvn test` (model tests skipped) or to the pure-Java unit tests.
**When you move a Java class the JNI layer references by name** (`LlamaModel`
[root], `exception.LlamaException`, `value.LogLevel`, `args.LogFormat`,
`callback.LoadProgressCallback`), update the matching `FindClass` / `"L…;"`
signature string in `src/main/cpp/jllama.cpp` and keep the native-resource root
anchored at `net/ladenthin/llama/` in `LlamaLoader.NATIVE_RESOURCE_BASE` (it must
not track the loader's own Java package). This is the same
"FQN/path not updated after a package move" class as the stale
`spotbugs-exclude.xml`, PIT `targetClasses`, and `CMakeLists.txt` OSInfo repairs.

### Code Formatting

C++ formatting is **enforced in CI** (`.github/workflows/clang-format.yml`) with a **pinned**
clang-format — currently **22.1.5**, installed via `pip install clang-format==22.1.5`. Format with
that exact version before committing; a different clang-format version reflows code differently and
will fail the check.

```bash
pip install "clang-format==22.1.5"
clang-format -i src/main/cpp/*.cpp src/main/cpp/*.hpp src/test/cpp/*.cpp   # Format C++ code
```

The generated JNI header `src/main/cpp/jllama.h` (produced by `javac -h`) is intentionally excluded.
To bump the enforced version, update the pin in **both** the workflow (`CLANG_FORMAT_VERSION`) and
this line, then reformat the whole tree with the new version in the same commit.

**`.clang-format` sets `SortIncludes: Never` — do not re-enable include sorting.** The project has
order-sensitive includes (see the "Include order rule" above): the upstream `server-*.h` headers and
`utils.hpp` must precede `json_helpers.hpp` / `jni_helpers.hpp`, which use the `json` alias those
headers define. Alphabetical sorting moves the helper headers first and breaks the build with
`'json' does not name a type` (it slips past a local build whose toolchain resolves `json` anyway,
but fails the manylinux/aarch64/Android CI compilers). Keep include order manual.

### Javadoc — must build cleanly before `mvn package`

The release packaging job runs `mvn package` with the `release` profile, which attaches
a javadoc jar via `maven-javadoc-plugin`. The plugin treats Javadoc tool **errors** as
build failures (warnings are tolerated). After changing any public/protected Java API,
verify the javadoc build succeeds locally:

```bash
mvn clean javadoc:jar -DskipTests=true -Dgpg.skip=true
# expected: BUILD SUCCESS
```

Common Javadoc errors that fail the build (not warnings):

- **Unbalanced HTML**: `</p>` without a matching `<p>`, mismatched `<ul>`/`<li>`, stray
  closing tags. Symptom: `error: unexpected end tag: </p>`.
- **Invalid `{@link …}` targets**: typo'd class, method, or parameter name.
- **Self-closing void HTML elements written as `<br>` inside `<pre>` blocks** in HTML5
  mode (rare but seen).

Common Javadoc *warnings* (do not fail the build, but should be cleaned up on new code):

- `no main description` — a doc comment containing only `@param`/`@return`/`@throws`
  tags with no leading prose. Fix: add a one-line description before the tags.
- `no @return` / `no @param` — public method missing the tag. Fix: add it.
- `no comment` — public method/field/enum constant has no doc comment at all.
- `use of default constructor, which does not provide a comment` — public class with
  no explicit constructor (the synthetic default has no Javadoc). Fix: add an explicit
  no-arg constructor with a Javadoc comment.

Preferred doc-comment shapes for getters and small value types:

```java
/**
 * Brief one-line description of the value.
 *
 * @return the value
 */
public T getThing() { ... }
```

A bare `/** @return … */` triggers `no main description`; add a leading sentence.

If the local check passes (`BUILD SUCCESS`), the `mvn package` job in
`.github/workflows/publish.yml` will pass the `attach-javadocs` step.

## Architecture

### Two-Layer Design

**Java layer** (`src/main/java/net/ladenthin/llama/`):
- `LlamaModel` — Main API class (AutoCloseable). Wraps native context for inference, embeddings, re-ranking, and tokenization.
- `ModelParameters` / `InferenceParameters` — Builder-pattern parameter classes that serialize to JSON (extend `JsonParameters`) for passing to native code.
- `LlamaIterator` / `LlamaIterable` — Streaming generation via Java `Iterator`/`Iterable`.
- `LlamaLoader` — Extracts the platform-specific native library from the JAR to a temp directory, or finds it on `java.library.path`.
- `OSInfo` — Detects OS and architecture for library resolution.
- **`server` package — OpenAI-compatible HTTP endpoint (a single implementation).**
  - `server.OpenAiCompatServer` — built only on the JDK's `com.sun.net.httpserver` (no new dependency), both embeddable and the fat-jar `Main-Class`. Serves `POST /v1/chat/completions` (streaming via SSE + non-streaming), `POST /v1/completions`, `POST /v1/embeddings`, `POST /v1/rerank`, `POST /infill`, `GET /v1/models` and `GET /health` (every route is also reachable without the `/v1` prefix), so editors that speak the OpenAI protocol (e.g. VS Code Copilot "Custom Endpoint", Cline, Roo Code, Continue) can drive a local model. Streaming chat uses the native OAI chunk path (`LlamaModel.streamChatCompletion` → `requestChatCompletionStream` / `receiveChatCompletionChunk` + the C++ `wrap_stream_chunk` helper), preserving `delta.tool_calls`; completions/embeddings/infill forward verbatim to the matching `LlamaModel.handle*`; rerank reshapes `handleRerank` into the OAI `results`/`data` shape. The chat mapper forwards `stream_options` and `response_format` and defaults `cache_prompt=true`; a CORS `Filter` answers `OPTIONS` preflights; `OpenAiSseFormatter.ensureUsageCachedTokens` guarantees `usage.prompt_tokens_details.cached_tokens` on the streamed usage chunk (Copilot crash fix, microsoft/vscode #273482). **Agentic tool-calling is the primary target**; a C++ guard (`test_server.cpp`) pins `tool_calls.function.arguments` as a JSON string (llama.cpp #20198).
  - **Alternative protocol surfaces** (pure translation over the OpenAI chat core — no second inference path; each reconstructs streamed tool calls via `ToolCallDeltaAccumulator`): **Ollama-native** (`GET /api/version`, `/api/tags`, `POST /api/show`, `/api/chat` with NDJSON streaming, `/api/generate` prompt-completion/FIM — `OllamaApiSupport`; `/api/show` advertises tools/insert/vision capabilities + context length for Copilot's Ollama provider), **Anthropic Messages** (`POST /v1/messages`, SSE event stream — `AnthropicApiSupport` + `AnthropicStreamTranslator`), and **OpenAI Responses** (`POST /v1/responses`, SSE event stream — `ResponsesApiSupport` + `ResponsesStreamTranslator`). The llama.cpp-native `GET /props` (context length + `modalities`) is served via `OpenAiSseFormatter.propsJson` for autocomplete clients that size their context from it.
  - Supporting classes: `OpenAiServerConfig` (builder; optional bearer auth; binds `127.0.0.1`; `corsAllowOrigin`; `supportsVision`), `OpenAiServerCli` (testable CLI arg parser → `ModelParameters` + `OpenAiServerConfig`; flags incl. `--mmproj`/`--embedding`/`--reranking`), `OpenAiRequestMapper` (OAI chat request → `InferenceParameters`), `OpenAiSseFormatter` (SSE/models/error JSON + usage normalization), `OaiRerankSupport` (pure rerank request/response shaping), and the model-free test seam `OpenAiBackend`/`ChunkSink` + `LlamaModelBackend`. The streaming envelope is parsed by `json.ChatStreamChunkParser`.
  - The `server` package is a dedicated top layer in the ArchUnit `layeredArchitecture` rule (the only layer allowed to access the root `Api`); `noInternalJdkImports` carries an explicit exception for the supported `com.sun.net.httpserver` (the exported `jdk.httpserver` module, which `module-info.java` `requires`). See README "OpenAI-compatible HTTP server".

**Native layer** (`src/main/cpp/`):
- `jllama.cpp` — JNI implementation bridging Java calls to llama.cpp. ~1,215 lines; 17 native methods.
- `utils.hpp` — Helper utilities (format helpers, argv stripping, token-piece serialisation).
- `json_helpers.hpp` — Pure JSON transformation helpers (no JNI, no llama state). Independently unit-testable.
- `jni_helpers.hpp` — JNI bridge helpers (handle management + server orchestration). Includes `json_helpers.hpp`.
- Uses `nlohmann/json` for JSON deserialization of parameters.
- The upstream server library (`server-context.cpp`, `server-queue.cpp`, `server-task.cpp`, `server-models.cpp`) is compiled directly into `jllama` via CMake — there is no hand-ported `server.hpp` fork. **Phase 2:** the upstream HTTP transport (`tools/server/server-http.cpp`) and its `cpp-httplib` backend (`vendor/cpp-httplib/httplib.cpp`) are now compiled into `jllama` too, so the OpenAI-compatible server can be driven natively from JNI *inside* `libjllama` — no separate `llama-server` executable (a JNI shared library loads anywhere a JVM runs, which a standalone binary does not). `server-http.cpp` does `#include "ui.h"` (the WebUI asset table that `tools/ui`/`llama-ui` normally generates); since the Svelte WebUI is not shipped, `src/main/cpp/webui_stub/ui.h` supplies the upstream **empty-asset** interface and leaves `LLAMA_UI_HAS_ASSETS` undefined (all static-asset-serving blocks compile out). `<cpp-httplib/httplib.h>` already resolves via `llama-common`'s `vendor/` include dir (same nlohmann/json 3.12.0 as the FetchContent copy). No SSL: `CPPHTTPLIB_OPENSSL_SUPPORT` is left undefined (plain-HTTP; bind localhost / front with a TLS proxy). Only `server.cpp` (the standalone `main()` + route wiring) remains excluded — wiring the routes to JNI is the next step.

### Native Helper Architecture

The project C++ helpers follow a strict semantic split:

**`json_helpers.hpp`** — Pure data transforms.
- Input: `nlohmann::json`, `server_task_result_ptr`, plain C++ types.
- Output: `json`, `std::vector`, `std::optional`, plain C++ types.
- Zero JNI calls (`JNIEnv*` never appears).
- Zero llama state (`llama_context*`, `llama_vocab*`, `server_context*` never appear).
- Functions are named without `_impl` suffix — they are the canonical implementation.
- Testable with JSON literals and fake result objects; no JVM and no loaded model required.
- Upstream server headers must be included by the translation unit first (they define `server_task_result_ptr`, `json`, etc.).

Functions: `get_result_error_message`, `results_to_json`, `rerank_results_to_json`,
`parse_encoding_format`, `extract_embedding_prompt`, `is_infill_request`,
`parse_slot_prompt_similarity`, `parse_positive_int_config`, `wrap_stream_chunk`.

**`log_helpers.hpp`** — Pure log-formatting transforms.
- Input: `ggml_log_level`, message text (`const char*`), an explicit `std::time_t` timestamp.
- Output: `const char*` level label / `std::string` JSON.
- Zero JNI calls (`JNIEnv*` never appears).
- Zero llama/server state — depends only on the `ggml_log_level` enum (from `ggml.h`) and
  nlohmann/json; no upstream server headers required (more standalone than `json_helpers.hpp`).
- Functions are `[[nodiscard]] inline`, named without an `_impl` suffix — the canonical implementation.
- Testable with literal levels/strings and a fixed timestamp; no JVM and no loaded model required.

Functions: `log_level_name`, `format_log_as_json`.

**`jni_helpers.hpp`** — JNI bridge helpers, split into two layers:

*Layer A* (no server headers required): handle management.
- `jllama_context` struct — owns `server_context` (value member, pimpl inside), background
  worker thread, cached `vocab`, saved `params`, and a `readers` map for streaming tasks.
- `get_jllama_context_impl` — reads Java `ctx` handle, returns the `jllama_context*` wrapper.
  Does NOT throw on zero handle (valid no-op for destructor-style calls).
- `require_json_field_impl` — throws `"<field> is required"` if key is absent.
- `jint_array_to_tokens_impl` — reads a Java `int[]` into `std::vector<int32_t>`.

*Layer B* (requires upstream server headers in the TU before `jni_helpers.hpp`): orchestration.
Includes `json_helpers.hpp` so all bridge helpers can call transforms directly.
- `json_to_jstring_impl` — serialises any `json` value to a JNI string via `dump()`.
- `results_to_jstring_impl` — delegates to `results_to_json` then `json_to_jstring_impl`.
- `vec_to_jarray_impl<JArray,JElem,CppElem>` — generic C++ vector → JNI primitive array.
- `embedding_to_jfloat_array_impl` — converts `std::vector<float>` to `jfloatArray`.
- `tokens_to_jint_array_impl` — converts `std::vector<int32_t>` to `jintArray`.

Functions with `_impl` suffix are called directly from `jllama.cpp`.

**Include order rule:**
```
// In jllama.cpp and any TU that uses Layer B helpers:
#include "server-context.h"   // upstream server headers must come first
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "jni_helpers.hpp"    // includes json_helpers.hpp internally
```

**Adding a new pure transform** (e.g. a new JSON field parser):
- Add it to `json_helpers.hpp`. No JNI, no llama types.
- Add tests to `src/test/cpp/test_json_helpers.cpp`.

**Adding a new JNI bridge helper:**
- Add it to `jni_helpers.hpp` in the appropriate layer.
- If it needs upstream server types, put it in Layer B (after the `json_helpers.hpp` include).
- Add tests to `src/test/cpp/test_jni_helpers.cpp`.

### Parameter Flow
Java parameters are serialized to JSON strings and passed to native code, which deserializes them using nlohmann/json. This avoids complex JNI field mapping for the many llama.cpp parameters.

### Native Library Resolution
`LlamaLoader` tries in order:
1. System property `net.ladenthin.llama.lib.path`
2. `java.library.path`
3. Extracts from JAR resources at `net/ladenthin/llama/{os}/{arch}/`

### Cross-compilation
Docker-based cross-compilation scripts are in `.github/dockcross/` for **Android** targets (and the
x86_64 manylinux jobs). **Linux `aarch64` is no longer cross-compiled** — it builds natively on a
GitHub `ubuntu-24.04-arm` runner (see "Linux aarch64: native ARM build" below). The
`.github/dockcross/dockcross-linux-arm64-lts` wrapper is now unused by CI (left in place; harmless).

### Linux aarch64: native ARM build

The `crosscompile-linux-aarch64` job (id kept for its downstream `needs:` reference; display name is
now **"Build and Test Linux aarch64"**) builds **natively on `ubuntu-24.04-arm`**, mirroring upstream
llama.cpp's own `ubuntu-cpu` aarch64 release job (`ubuntu-24.04-arm` + **GCC 14**).

**Why it moved off dockcross.** The old `dockcross/linux-arm64-lts` image ships **GCC 8.5 / glibc
2.17**; llama.cpp **b9739** uses C++17 CTAD-in-`new`, which needs **GCC ≥ 12**, so the cross build
stopped compiling. Upstream solved the same problem by building natively on `ubuntu-24.04-arm` with
GCC 14 and ships a **glibc ≈ 2.39** ARM binary with no old-glibc compatibility layer. This repo now
does the same: the aarch64 artifact's **glibc floor rises 2.17 → ~2.39** — the same envelope
upstream's own ARM binaries require (the x86_64 artifact stays at manylinux2014 / glibc 2.17).

Wiring (mirrors the macOS native jobs, not the dockcross jobs):
- `runs-on: ubuntu-24.04-arm`; `setup-java` → `mvn compile` (generates the JNI header) → `build.sh`.
- Installs `gcc-14`/`g++-14` and exports `CC`/`CXX` (upstream parity).
- `build.sh` flags: `-DGGML_NATIVE=OFF` (portable across ARMv8 CPU generations — no build-host
  `-march` baked in) `-DBUILD_TESTING=ON`, then **`ctest` runs the C++ unit suite on real ARM
  hardware** (the cross build ran no tests at all).
- sccache: `build.sh`'s Linux auto-fetch now covers `aarch64` as well as `x86_64` (it maps
  `uname -m` to the matching static-musl release); the probe still gates it, so a miss just builds
  uncached.
- Branch protection: if a required check pinned the old name "Cross-Compile Linux aarch64 (LTS)",
  repoint it to "Build and Test Linux aarch64".

## Testing

### Java tests
Require a model file. The CI downloads models from HuggingFace:
- **LlamaModel tests**: CodeLlama-7B-GGUF (`codellama-7b.Q2_K.gguf`)
- **RerankingModel tests**: Jina-Reranker model

Set the model path via system property or environment variable (see test files for exact property names).

Test files are in `src/test/java/net/ladenthin/llama/` and `src/test/java/examples/`.

### C++ unit tests

**No JVM and no model file required.** All tests run on pure data structures using mock
objects. The binary is named `jllama_test` and is built by CMake when `BUILD_TESTING=ON`.

#### Commands

```bash
# 1. Configure (once per fresh clone or after CMakeLists.txt changes)
cmake -B build -DBUILD_TESTING=ON

# 2. Build (incremental; -j$(nproc) uses all CPU cores)
cmake --build build --config Release -j$(nproc)

# 3. Run all tests
ctest --test-dir build --output-on-failure

# Count tests across all files
grep -rn "^TEST\b\|^TEST_F\b\|^TEST_P\b" src/test/cpp/ | wc -l

# Run a single named test (GoogleTest filter syntax)
ctest --test-dir build --output-on-failure -R "ResultsToJson"
```

#### Test files

| File | Tests | Scope |
|------|-------|-------|
| `src/test/cpp/test_utils.cpp` | 156 | Upstream helpers: `server_tokens`, `server_grammar_trigger`, `gen_tool_call_id`, `json_value`, `json_get_nested_values`, UTF-8 helpers, `format_response_rerank`, `format_embeddings_response_oaicompat`, `oaicompat_completion_params_parse`, `oaicompat_chat_params_parse`, `are_lora_equal`, `strip_flag_from_argv`, `token_piece_value`, `json_is_array_and_contains_numbers`, `format_oai_sse`, `format_oai_resp_sse`, `format_anthropic_sse` |
| `src/test/cpp/test_server.cpp` | 188 | Upstream result types: `result_timings`, `task_params::to_json()` (incl. `dry_sequence_breakers`, `preserved_tokens`, `timings_per_token`), `completion_token_output`, `server_task_result_cmpl_partial` (non-oaicompat + `to_json_oaicompat` + logprobs + `to_json_oaicompat_chat` + `to_json_anthropic` + dispatcher), `server_task_result_cmpl_final` (non-oaicompat + `to_json_oaicompat` + `to_json_oaicompat_chat` + `to_json_oaicompat_chat_stream` + `to_json_anthropic` + `to_json_anthropic_stream` + tool_calls + dispatcher), `server_task_result_embd`, `server_task_result_rerank`, `server_task_result_metrics`, `server_task_result_slot_save_load`, `server_task_result_slot_erase`, `server_task_result_apply_lora`, `server_task_result_error`, `format_error_response`, `server_task::need_sampling()`, `server_task::n_tokens()`, `server_schema::eval_llama_cmpl_schema()` (parsing pipeline + grammar routing + error paths), `response_fields` projection |
| `src/test/cpp/test_json_helpers.cpp` | 47 | All functions in `json_helpers.hpp`: `get_result_error_message`, `results_to_json`, `rerank_results_to_json`, `parse_encoding_format`, `extract_embedding_prompt`, `is_infill_request`, `parse_slot_prompt_similarity`, `parse_positive_int_config`, `wrap_stream_chunk` |
| `src/test/cpp/test_log_helpers.cpp` | 13 | All functions in `log_helpers.hpp`: `log_level_name`, `format_log_as_json` |
| `src/test/cpp/test_jni_helpers.cpp` | 41 | All functions in `jni_helpers.hpp` using a zero-filled `JNINativeInterface_` mock |

**Current total: 445 tests (all passing).**

#### Upstream source location (in CMake build tree)

llama.cpp is fetched via CMake FetchContent, pinned to `GIT_TAG b9739`.

**GoogleTest** is a separate `BUILD_TESTING`-only FetchContent (`GIT_TAG v1.17.0`), used solely
by the `jllama_test` C++ unit-test binary — not by the shipped library, and not coupled to the
llama.cpp pin or the bundled nlohmann/json. There is **no constraint behind the exact tag**; it
is just the latest stable at the time it was last touched. Bump it from time to time (nothing
auto-tracks it), pairing the bump with a green `C++ Tests` CI run.

```
build/_deps/llama.cpp-src/tools/server/   ← server-task.h, server-common.h, etc.
build/_deps/llama.cpp-src/include/        ← llama.h, llama-cpp.h
build/_deps/llama.cpp-src/common/         ← common.h, chat.h, arg.h, etc.
```

When reading a `to_json()` implementation to write tests against it, read from:
`build/_deps/llama.cpp-src/tools/server/server-task.cpp`

#### Mock JNI pattern used in test_jni_helpers.cpp

```cpp
// Zero-fill the interface so all unpatched fn pointers are nullptr
JNINativeInterface_ iface = {};
// Patch only the stubs this test needs, e.g.:
iface.GetLongField  = [](JNIEnv*, jobject, jfieldID) -> jlong { return some_handle; };
iface.ThrowNew      = [](JNIEnv*, jclass, const char*) -> jint { return 0; };
// Wire up the env
JNIEnv_ fake_env = {};
fake_env.functions = &iface;
JNIEnv *env = &fake_env;
```

Any stub that is called but not patched will crash (null function pointer) — deliberately,
so missing stubs are caught immediately rather than silently.

#### How to add a new C++ test

1. Open the appropriate `src/test/cpp/test_*.cpp`:
   - Pure JSON transform → `test_json_helpers.cpp`
   - JNI helper → `test_jni_helpers.cpp`
   - Upstream result type `to_json()` → `test_server.cpp`
   - `utils.hpp` function or upstream utility → `test_utils.cpp`
2. Add a `TEST(SuiteName, TestName) { ... }` block using GoogleTest macros.
3. Rebuild: `cmake --build build --config Release -j$(nproc)`
4. Run: `ctest --test-dir build --output-on-failure`
5. Commit with message summarising coverage added and new test total.

#### Finding untested code paths

```bash
# List all functions defined in a header
grep -n "^inline\|^static\|^\[\[nodiscard\]\]" src/main/cpp/utils.hpp

# Check which functions already have tests
grep -n "function_name" src/test/cpp/*.cpp

# Find all fields in an upstream to_json() method
grep -n "\"field_name\"" build/_deps/llama.cpp-src/tools/server/server-task.cpp

# Check which JSON fields Java actually reads (important: must test these)
grep -rn "field_name" src/main/java/net/ladenthin/llama/
```

#### Testing complex scenarios — methodology

Simple tests verify individual field values on a default-constructed struct.
Complex tests verify **control flow**: switch dispatchers, cross-cutting flags, and
multi-step parameter pipelines.  The same build/run/commit loop applies.

**1. Dispatcher (switch) coverage**

Every `to_json()` that is a switch on `res_type` has one test per arm:

```cpp
// Pattern: set is_updated=true, set res_type, call to_json(), check the
// distinguishing field that differs between arms.
server_task_result_cmpl_final f;
f.is_updated = true;
f.stream     = false;
f.res_type   = TASK_RESPONSE_TYPE_OAI_CMPL;
// ... set required fields ...
const json j = f.to_json();
EXPECT_EQ(j.at("object").get<std::string>(), "text_completion");
```

The same pattern handles the `stream` flag fork inside `OAI_CHAT`:
`stream=false` → single object with `"object":"chat.completion"`;
`stream=true`  → JSON array of chunks with `"object":"chat.completion.chunk"`.

**2. Cross-cutting flag interaction**

Some flags (verbose, include_usage, timings.prompt_n) cut across multiple formatters.
Test each flag in one formatter only — they share the same code path:

```cpp
// verbose=true must add __verbose to the first chunk/top-level object
f.verbose = true;
EXPECT_TRUE(j.contains("__verbose"));

// timings absent when prompt_n < 0 (default), present when >= 0
f.timings.prompt_n = 5;
EXPECT_TRUE(j.contains("timings"));
```

**3. Parameter parsing (`eval_llama_cmpl_schema`) without a model**

`server_schema::eval_llama_cmpl_schema(vocab, params_base, n_ctx_slot, logit_bias_eog, data)`
can be called with `nullptr` vocab **if the JSON does not trigger grammar/preserved_tokens
tokenisation** (those are the only vocab-dependent paths).  This lets us test the full
parsing pipeline including error throws:

```cpp
common_params          params_base;
std::vector<llama_logit_bias> no_bias;
const int n_ctx = 512;

// test: repeat_last_n=-1 is expanded to n_ctx_slot
json data = {{"repeat_last_n", -1}};
auto p = server_schema::eval_llama_cmpl_schema(nullptr, params_base, n_ctx, no_bias, data);
EXPECT_EQ(p.sampling.penalty_last_n, n_ctx);

// test: invalid value throws std::runtime_error
json bad = {{"dry_sequence_breakers", json::array()}};  // empty → error
EXPECT_THROW(server_schema::eval_llama_cmpl_schema(nullptr, params_base, n_ctx, no_bias, bad),
             std::runtime_error);
```

**4. Array-returning formatters**

Some methods (e.g. `to_json_oaicompat_chat_stream()`) return a JSON array of event objects,
not a single object.  Check with `is_array()` first, then iterate or index:

```cpp
const json j = f.to_json_oaicompat_chat_stream();
ASSERT_TRUE(j.is_array());
ASSERT_GE(j.size(), 1u);
// Last chunk always has a non-null finish_reason
EXPECT_FALSE(j.back().at("choices")[0].at("finish_reason").is_null());
```

**5. `response_fields` projection**

`to_json_non_oaicompat()` supports a projection list via `response_fields`.
When non-empty, only those dot-separated paths survive:

```cpp
f.response_fields = {"content", "tokens_predicted"};
const json j = f.to_json_non_oaicompat();
EXPECT_TRUE(j.contains("content"));
EXPECT_FALSE(j.contains("stop_type"));  // filtered out
```

## Key Constraints

- **Java 8+** runtime required. Built with JDK 21 targeting bytecode 1.8 for broad compatibility.
- Native memory allocated by llama.cpp is not GC-managed — always use `LlamaModel` in try-with-resources or call `close()` explicitly.
- The `server.hpp` file is adapted from llama.cpp upstream — minimize modifications to ease future upgrades.
- Platform-specific native libraries must be pre-built and placed under `src/main/resources/` before packaging for distribution.

## Javadoc Conventions

See [`../workspace/policies/javadoc-conventions.md`](../workspace/policies/javadoc-conventions.md).

## SpotBugs Suppressions

See [`../workspace/policies/spotbugs-suppressions.md`](../workspace/policies/spotbugs-suppressions.md).

## Spotless Formatting

See [`../workspace/policies/spotless-formatting.md`](../workspace/policies/spotless-formatting.md).
Run `mvn spotless:apply` before every commit that touches `.java` files.

## jqwik Policy

See [`../workspace/policies/jqwik-prompt-injection.md`](../workspace/policies/jqwik-prompt-injection.md).

## Lombok Config

See [`../workspace/policies/lombok-config.md`](../workspace/policies/lombok-config.md).

## CI Test Diagnostics

See [`../workspace/policies/ci-test-diagnostics.md`](../workspace/policies/ci-test-diagnostics.md).

## JPMS Module Descriptor

This repo ships a `module-info.java` compiled in a separate `release 9` execution. Javadoc
currently runs in **classpath mode** (javadoc `<source>` is `1.8`), which is the *only* thing
keeping it clear of the JPMS module-mode javadoc trap that bit BAF. **Before raising the Java /
javadoc source level to ≥ 9, read**
[`../workspace/policies/jpms-module-descriptor.md`](../workspace/policies/jpms-module-descriptor.md).

## Open TODOs

Open TODOs for this repo live in [`TODO.md`](TODO.md). Cross-repo status
tracking lives in [`../workspace/crossrepostatus.md`](../workspace/crossrepostatus.md).
