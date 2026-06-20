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
