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
if [ "${USE_CACHE:-true}" = "true" ] && [ -n "${SCCACHE_WEBDAV_TOKEN:-}${SCCACHE_GHA_ENABLED:-}" ] \
   && ! command -v sccache >/dev/null 2>&1 \
   && [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
  SCCACHE_REL="sccache-v0.8.2-x86_64-unknown-linux-musl"
  echo "build.sh: fetching ${SCCACHE_REL} (no sccache on PATH)..."
  if curl -fsSL --proto =https --proto-redir =https \
        "https://github.com/mozilla/sccache/releases/download/v0.8.2/${SCCACHE_REL}.tar.gz" \
        -o /tmp/sccache.tgz && tar -xzf /tmp/sccache.tgz -C /tmp; then
    export PATH="/tmp/${SCCACHE_REL}:$PATH"
    echo "build.sh: sccache -> $(command -v sccache || echo 'still missing')"
  else
    echo "build.sh: sccache fetch failed; continuing without cache"
  fi
fi

# Optional shared compiler cache: sccache fronting Depot Cache (WebDAV). Enabled only when
# USE_CACHE is true AND sccache + a cache token are present, so it stays inert before the
# DEPOT_TOKEN secret is configured and on fork PRs (secrets hidden) — those just compile
# normally. sccache is content-addressed, so a cache hit is bit-identical to a fresh -O3
# compile (release-safe), and it degrades to direct compilation if the cache is unreachable.
LAUNCH=""
if [ "${USE_CACHE:-true}" = "true" ] && command -v sccache >/dev/null 2>&1 \
   && [ -n "${SCCACHE_WEBDAV_TOKEN:-}${SCCACHE_GHA_ENABLED:-}" ]; then
  LAUNCH="-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
  echo "build.sh: sccache ON (endpoint=${SCCACHE_WEBDAV_ENDPOINT:-default}), building with -j${JOBS}"
else
  echo "build.sh: sccache OFF, building with -j${JOBS}"
fi

cmake -Bbuild $LAUNCH $@ || exit 1
cmake --build build --config Release -j"${JOBS}" || exit 1

if command -v sccache >/dev/null 2>&1; then
  sccache --show-stats || true
fi
