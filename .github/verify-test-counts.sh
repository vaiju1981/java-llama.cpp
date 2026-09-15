#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# Fails a Java test job when the suite silently stopped running tests.
#
# WHY THIS EXISTS. Surefire's working directory is the module basedir while the shared GGUF cache
# restores to the reactor root, so every model path resolved one directory too deep and EVERY
# model-gated class aborted in its @BeforeAll assumption — on every test-java-* job, for months,
# while the pipeline stayed green. Several stale assertions rode along unnoticed.
#
# The shape is what defeats the obvious guard: a CLASS-LEVEL assumption failure makes Surefire
# record tests="0" errors="0" failures="0" skipped="0" for that class. It contributes NO test
# entries at all, so "did this run skip anything?" is structurally blind to it — a skipped test is
# still a reported test. Two checks catch it:
#
#   1. Any testsuite reporting tests="0". This is the exact signature above, it is precise, and it
#      is platform-independent: a class that runs nowhere is a bug on every OS.
#   2. A floor on the total number of tests executed. The backstop for a whole class file going
#      missing from the run rather than reporting zero.
#
# Check 1 is the sensitive one; check 2 is deliberately slack so it never fails spuriously on a
# platform that legitimately runs fewer tests. Tighten --min-total once real per-platform numbers
# are known from a green run.
#
# Usage: .github/verify-test-counts.sh <surefire-reports-dir> [--min-total N]
# Exit codes: 0 all good, 1 a check failed, 2 nothing to scan.

set -euo pipefail

REPORT_DIR="${1:-}"
MIN_TOTAL=0
shift || true
while [ $# -gt 0 ]; do
    case "$1" in
        --min-total) MIN_TOTAL="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[ -n "$REPORT_DIR" ] || { echo "usage: $0 <surefire-reports-dir> [--min-total N]" >&2; exit 1; }
[ -d "$REPORT_DIR" ] || { echo "ERROR: no such directory: $REPORT_DIR" >&2; exit 2; }

shopt -s nullglob
reports=("$REPORT_DIR"/TEST-*.xml)
shopt -u nullglob

# An empty input is a failure, never a pass — the same rule verify-bytecode-version.sh follows,
# for the same reason: a job that produced no reports at all has not proved anything.
if [ "${#reports[@]}" -eq 0 ]; then
    echo "ERROR: no TEST-*.xml under $REPORT_DIR — the suite did not run" >&2
    exit 2
fi

total=0
empty_suites=()
for f in "${reports[@]}"; do
    # The count lives on the <testsuite> element; take the first match so a nested element
    # carrying the same attribute name cannot shift the number.
    n="$(grep -o 'tests="[0-9]*"' "$f" | head -1 | grep -o '[0-9]*' || true)"
    [ -n "$n" ] || n=0
    total=$((total + n))
    if [ "$n" -eq 0 ]; then
        empty_suites+=("$(basename "$f")")
    fi
done

status=0

if [ "${#empty_suites[@]}" -gt 0 ]; then
    echo "ERROR: ${#empty_suites[@]} test class(es) contributed ZERO test entries:" >&2
    printf '  %s\n' "${empty_suites[@]}" >&2
    echo "  A class-level @BeforeAll assumption that fails looks exactly like this. It is NOT a" >&2
    echo "  skip — the class reports no tests at all, so it is invisible to a skip check. If a" >&2
    echo "  model is missing, validate-models should have failed the job before this point." >&2
    status=1
fi

if [ "$total" -lt "$MIN_TOTAL" ]; then
    echo "ERROR: only $total test(s) executed, below the floor of $MIN_TOTAL" >&2
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo "test counts verified: $total test(s) across ${#reports[@]} class(es), none empty (floor $MIN_TOTAL)"
fi
exit "$status"
