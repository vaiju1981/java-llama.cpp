#!/bin/bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
# SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
#
# SPDX-License-Identifier: MIT

mkdir -p build

# Build parallelism. Defaults to all cores; RAM-limited CI runners (notably GitHub's
# ~7 GB macOS arm64) export BUILD_JOBS lower (e.g. 2) so the large httplib.cpp + the 134
# llama.cpp model TUs do not exhaust memory and get the runner OOM-killed mid-compile
# (which surfaces as a SIGTERM / "runner received a shutdown signal", not a clean timeout).
JOBS="${BUILD_JOBS:-}"
if [ -z "$JOBS" ]; then
  JOBS="$( { command -v nproc >/dev/null 2>&1 && nproc; } || sysctl -n hw.ncpu 2>/dev/null || echo 4 )"
fi

# Fetch sccache when caching is requested but the runner/container doesn't ship it — the
# dockcross cross-compile containers (manylinux/Android) and Linux hosts have no sccache,
# while macOS installs it via brew in the workflow. Best-effort and inert-safe: any failure
# leaves sccache absent, so the build just proceeds uncached. The static musl binary runs in
# any x86_64 Linux container (the cross-compile host is always x86_64).
#
# SCCACHE_DL_VERSION is overridable per-job, so a container that crashes one sccache build can
# try another without editing this script (the in-container panic that stalled phase 2 was on
# v0.8.2; v0.15.0 is the current stable default). A wrong/unavailable version just fails the
# `curl -f` and falls back to an uncached build, so bumping it can never red a build.
SCCACHE_DL_VERSION="${SCCACHE_DL_VERSION:-0.15.0}"
if [ "${USE_CACHE:-true}" = "true" ] && [ -n "${SCCACHE_WEBDAV_TOKEN:-}${SCCACHE_GHA_ENABLED:-}" ] \
   && ! command -v sccache >/dev/null 2>&1 \
   && [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
  SCCACHE_REL="sccache-v${SCCACHE_DL_VERSION}-x86_64-unknown-linux-musl"
  echo "build.sh: fetching ${SCCACHE_REL} (no sccache on PATH)..."
  if curl -fsSL --proto =https --proto-redir =https \
        "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_DL_VERSION}/${SCCACHE_REL}.tar.gz" \
        -o /tmp/sccache.tgz && tar -xzf /tmp/sccache.tgz -C /tmp; then
    export PATH="/tmp/${SCCACHE_REL}:$PATH"
    echo "build.sh: sccache -> $(command -v sccache || echo 'still missing')"
  else
    echo "build.sh: sccache fetch failed; continuing without cache"
  fi
fi

# Health-check before trusting sccache as the compiler launcher. Because sccache *is* the
# launcher (cmake runs `sccache <compiler> ...` for every TU), a present-but-crashing sccache
# fails every compile and reds the whole build — exactly the in-container panic that stalled
# phase 2 (the static-musl binary panicked while wrapping the cross-compiler, failing ggml.c.o).
# The probe runs the real compiler through sccache on a trivial TU; only if that succeeds is the
# launcher enabled. On any failure it logs the captured output (the Rust panic backtrace, plus
# the detached server's SCCACHE_ERROR_LOG when a job sets one) and the build runs WITHOUT the
# cache — a clean, uncached -O3 build that still goes green. This closes the gap the old
# absent-only guard left: it handled sccache *missing*, not sccache *crashing*.
sccache_can_wrap_compiler() {
  probe_cc="${CC:-}"
  if [ -z "$probe_cc" ]; then
    for c in cc gcc clang; do
      if command -v "$c" >/dev/null 2>&1; then probe_cc="$c"; break; fi
    done
  fi
  if [ -z "$probe_cc" ]; then
    echo "build.sh: sccache probe: no C compiler on PATH to probe; building uncached"
    return 1
  fi
  probe_dir="$(mktemp -d 2>/dev/null || echo "/tmp/sccache-probe.$$")"
  mkdir -p "$probe_dir" || return 1
  printf 'int main(void){return 0;}\n' > "$probe_dir/probe.c"
  probe_out="$(sccache "$probe_cc" -c "$probe_dir/probe.c" -o "$probe_dir/probe.o" 2>&1)"
  probe_rc=$?
  rm -rf "$probe_dir"
  if [ "$probe_rc" -ne 0 ]; then
    echo "build.sh: sccache probe FAILED (rc=${probe_rc}) wrapping '${probe_cc}' — building WITHOUT cache."
    [ -n "$probe_out" ] && printf '%s\n' "$probe_out" | sed 's/^/build.sh:   sccache-probe| /'
    if [ -n "${SCCACHE_ERROR_LOG:-}" ] && [ -f "${SCCACHE_ERROR_LOG}" ]; then
      echo "build.sh:   --- detached server log (${SCCACHE_ERROR_LOG}) ---"
      sed 's/^/build.sh:   sccache-srv| /' "${SCCACHE_ERROR_LOG}" 2>/dev/null || true
    fi
    return 1
  fi
  echo "build.sh: sccache probe OK (wrapped '${probe_cc}')"
  return 0
}

# Optional shared compiler cache: sccache fronting Depot Cache (WebDAV). Enabled only when
# USE_CACHE is true AND sccache + a cache token are present AND the probe confirms sccache can
# wrap the compiler — so it stays inert before the DEPOT_TOKEN secret is configured, on fork PRs
# (secrets hidden), and when sccache would crash; all of those just compile normally. sccache is
# content-addressed, so a cache hit is bit-identical to a fresh -O3 compile (release-safe), and
# it degrades to direct compilation if the cache is unreachable.
LAUNCH=""
if [ "${USE_CACHE:-true}" = "true" ] && command -v sccache >/dev/null 2>&1 \
   && [ -n "${SCCACHE_WEBDAV_TOKEN:-}${SCCACHE_GHA_ENABLED:-}" ] \
   && sccache_can_wrap_compiler; then
  LAUNCH="-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
  echo "build.sh: sccache ON (endpoint=${SCCACHE_WEBDAV_ENDPOINT:-default}), building with -j${JOBS}"
else
  echo "build.sh: sccache OFF, building with -j${JOBS}"
fi

cmake -Bbuild $LAUNCH $@ || exit 1
cmake --build build --config Release -j"${JOBS}" || exit 1

# Only query stats when sccache was actually used as the launcher; if the probe rejected a
# crashing sccache, re-invoking it here would just repeat the crash output (harmless but noisy).
if [ -n "$LAUNCH" ] && command -v sccache >/dev/null 2>&1; then
  sccache --show-stats || true
fi
