#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
#
# SPDX-License-Identifier: MIT

# Assembles the per-OS "all backends" server fat jars distributed as GitHub Release
# assets (never deployed to Maven Central).
#
# For every OS/arch that has GPU classifier jars, the default jar-with-dependencies
# uber jar (library classes + Java runtime deps + default CPU natives for every
# platform) is copied and each backend's native tree is added under a backend
# subdirectory (net/ladenthin/llama/<OS>/<ARCH>/<backend>/), together with a
# jllama-backends.txt manifest listing the backends in priority order. LlamaLoader
# reads that manifest at runtime and loads the first backend whose library loads,
# falling back to the default CPU library (see LlamaLoader.BACKEND_MANIFEST_FILE).
#
# Fail-loud invariants (a broken invariant must red the pipeline, never skip):
#   * every <classifier> in llama/pom.xml must be parseable/explicitly excluded,
#   * the pom classifier set and the on-disk classifier-jar set must match exactly,
#   * every expected backend library must end up inside the combined jar,
#   * the Main-Class of the combined jar must survive the zip update.
#
# Usage: package-fatjars.sh <jars-dir> <out-dir> [pom]
#   jars-dir  directory holding the `llama-jars` artifact (llama/target/*.jar)
#   out-dir   output directory for the combined fat jars + sha256 files
#   pom       path to llama/pom.xml (default: llama/pom.xml)
set -euo pipefail

JARS_DIR="${1:?usage: package-fatjars.sh <jars-dir> <out-dir> [pom]}"
OUT_DIR="${2:?usage: package-fatjars.sh <jars-dir> <out-dir> [pom]}"
POM="${3:-llama/pom.xml}"

fail() {
    echo "::error::$*" >&2
    exit 1
}

# Resolve to absolute paths: the zip update below runs from the staging directory.
[ -d "$JARS_DIR" ] || fail "jars dir not found: $JARS_DIR"
JARS_DIR="$(cd "$JARS_DIR" && pwd)"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# Classifiers that intentionally get no combined-jar treatment:
#   msvc-windows           CPU-only build variant, redundant with the default Windows natives
#   opencl-android-aarch64 `java -jar` is not applicable on Android (AAR is the delivery path)
EXCLUDED_CLASSIFIERS=("msvc-windows" "opencl-android-aarch64")

# Backend priority order used for the manifest (first loadable backend wins at runtime).
# A classifier whose backend token is not in this list fails the build: a new backend
# must be consciously ranked here, not silently appended.
BACKEND_PRIORITY=("cuda13" "rocm" "sycl-fp16" "sycl-fp32" "sycl" "vulkan" "opencl" "openvino")

MAIN_CLASS="net.ladenthin.llama.server.ServerLauncher"

is_excluded() {
    local classifier="$1" excluded
    for excluded in "${EXCLUDED_CLASSIFIERS[@]}"; do
        [ "$classifier" = "$excluded" ] && return 0
    done
    return 1
}

backend_priority_index() {
    local backend="$1" i
    for i in "${!BACKEND_PRIORITY[@]}"; do
        [ "${BACKEND_PRIORITY[$i]}" = "$backend" ] && { echo "$i"; return 0; }
    done
    return 1
}

# --- 1. Source of truth #1: the <classifier> set declared in the pom -------------------
[ -f "$POM" ] || fail "pom not found: $POM"
mapfile -t POM_CLASSIFIERS < <(grep -oP '<classifier>\K[^<]+(?=</classifier>)' "$POM" | sort -u)
[ "${#POM_CLASSIFIERS[@]}" -gt 0 ] || fail "no <classifier> entries parsed from $POM"
echo "pom classifiers (${#POM_CLASSIFIERS[@]}): ${POM_CLASSIFIERS[*]}"

# --- 2. The base fat jar (exactly one) + the version derived from its file name --------
mapfile -t BASE_FAT_JARS < <(find "$JARS_DIR" -maxdepth 1 -name 'llama-*-jar-with-dependencies.jar' | sort)
[ "${#BASE_FAT_JARS[@]}" -eq 1 ] \
    || fail "expected exactly 1 default jar-with-dependencies in $JARS_DIR, got ${#BASE_FAT_JARS[@]}: ${BASE_FAT_JARS[*]:-none}"
BASE_FAT_JAR="${BASE_FAT_JARS[0]}"
VERSION="$(basename "$BASE_FAT_JAR")"
VERSION="${VERSION#llama-}"
VERSION="${VERSION%-jar-with-dependencies.jar}"
echo "base fat jar: $BASE_FAT_JAR (version $VERSION)"

# --- 3. Source of truth #2: the classifier jars actually built by the package job ------
DISK_CLASSIFIERS=()
for jar in "$JARS_DIR"/llama-"$VERSION"-*.jar; do
    [ -e "$jar" ] || continue
    classifier="$(basename "$jar")"
    classifier="${classifier#llama-"$VERSION"-}"
    classifier="${classifier%.jar}"
    case "$classifier" in
        sources | javadoc | jar-with-dependencies) continue ;;
    esac
    DISK_CLASSIFIERS+=("$classifier")
done
[ "${#DISK_CLASSIFIERS[@]}" -gt 0 ] || fail "no classifier jars found in $JARS_DIR for version $VERSION"
mapfile -t DISK_CLASSIFIERS < <(printf '%s\n' "${DISK_CLASSIFIERS[@]}" | sort -u)

# --- 4. Exact set equality in BOTH directions (no silent gaps, no unknown jars) --------
if ! diff <(printf '%s\n' "${POM_CLASSIFIERS[@]}") <(printf '%s\n' "${DISK_CLASSIFIERS[@]}"); then
    fail "classifier set mismatch between $POM and the jars in $JARS_DIR (see diff above)"
fi
echo "classifier sets match (${#POM_CLASSIFIERS[@]} classifiers)"

# --- 5. Parse every non-excluded classifier as <backend>-<os>-<arch> -------------------
# Associative arrays keyed by "<os>-<arch>" in classifier notation (e.g. linux-x86-64).
declare -A TARGET_BACKENDS   # target -> space-separated backend names
declare -A BACKEND_CLASSIFIER # "<target>/<backend>" -> classifier string
for classifier in "${POM_CLASSIFIERS[@]}"; do
    if is_excluded "$classifier"; then
        echo "excluded from combined jars: $classifier"
        continue
    fi
    case "$classifier" in
        *-linux-x86-64) os="linux" arch="x86-64" backend="${classifier%-linux-x86-64}" ;;
        *-linux-aarch64) os="linux" arch="aarch64" backend="${classifier%-linux-aarch64}" ;;
        *-windows-x86-64) os="windows" arch="x86-64" backend="${classifier%-windows-x86-64}" ;;
        *-windows-aarch64) os="windows" arch="aarch64" backend="${classifier%-windows-aarch64}" ;;
        *) fail "unparseable classifier '$classifier': expected <backend>-<os>-<arch> — add a parse rule or an explicit exclusion" ;;
    esac
    backend_priority_index "$backend" > /dev/null \
        || fail "backend '$backend' (classifier '$classifier') is not in BACKEND_PRIORITY — rank the new backend consciously"
    target="$os-$arch"
    TARGET_BACKENDS[$target]="${TARGET_BACKENDS[$target]:-} $backend"
    BACKEND_CLASSIFIER["$target/$backend"]="$classifier"
done
[ "${#TARGET_BACKENDS[@]}" -gt 0 ] || fail "no combined-jar targets derived from the classifier set"

# --- 6. Assemble one combined jar per target --------------------------------------------
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

for target in $(printf '%s\n' "${!TARGET_BACKENDS[@]}" | sort); do
    case "$target" in
        *-x86-64) os="${target%-x86-64}" arch="x86-64" ;;
        *-aarch64) os="${target%-aarch64}" arch="aarch64" ;;
        *) fail "internal error: unexpected target '$target'" ;;
    esac
    # Classifier notation -> resource-folder notation (OSInfo folder names).
    case "$os" in
        linux) os_folder="Linux" main_lib="libjllama.so" ;;
        windows) os_folder="Windows" main_lib="jllama.dll" ;;
        *) fail "internal error: unexpected os '$os'" ;;
    esac
    case "$arch" in
        x86-64) arch_folder="x86_64" ;;
        aarch64) arch_folder="aarch64" ;;
        *) fail "internal error: unexpected arch '$arch'" ;;
    esac
    resource_dir="net/ladenthin/llama/$os_folder/$arch_folder"

    # Order this target's backends by BACKEND_PRIORITY.
    ordered_backends=()
    for backend in "${BACKEND_PRIORITY[@]}"; do
        case " ${TARGET_BACKENDS[$target]} " in
            *" $backend "*) ordered_backends+=("$backend") ;;
        esac
    done

    staging="$WORK_DIR/$target"
    mkdir -p "$staging/$resource_dir"
    manifest="$staging/$resource_dir/jllama-backends.txt"
    {
        echo "# Native backends in priority order; first loadable backend wins."
        echo "# Extra tokens after a backend name are sibling files loaded before its library."
    } > "$manifest"

    for backend in "${ordered_backends[@]}"; do
        classifier="${BACKEND_CLASSIFIER["$target/$backend"]}"
        classifier_jar="$JARS_DIR/llama-$VERSION-$classifier.jar"
        extract_dir="$WORK_DIR/extract-$target-$backend"
        mkdir -p "$extract_dir"
        unzip -q "$classifier_jar" "$resource_dir/*" -d "$extract_dir" \
            || fail "$classifier_jar contains no native tree at $resource_dir"
        [ -f "$extract_dir/$resource_dir/$main_lib" ] \
            || fail "$classifier_jar: expected $resource_dir/$main_lib not found"
        mkdir -p "$staging/$resource_dir/$backend"
        mv "$extract_dir/$resource_dir/"* "$staging/$resource_dir/$backend/"
        # Manifest line: backend name + every sibling file (sorted, deterministic) except
        # the main library — the loader extracts and loads the siblings first.
        extras="$(cd "$staging/$resource_dir/$backend" && find . -type f ! -name "$main_lib" -printf '%P\n' | sort | tr '\n' ' ')"
        extras="${extras% }"
        if [ -n "$extras" ]; then
            echo "$backend $extras" >> "$manifest"
        else
            echo "$backend" >> "$manifest"
        fi
    done

    out_jar="$OUT_DIR/llama-$VERSION-all-$target-jar-with-dependencies.jar"
    cp "$BASE_FAT_JAR" "$out_jar"
    (cd "$staging" && zip -q -ur "$out_jar" net)

    # --- Verify the combined jar -----------------------------------------------------
    listing="$(unzip -l "$out_jar")"
    for backend in "${ordered_backends[@]}"; do
        echo "$listing" | grep -qF "$resource_dir/$backend/$main_lib" \
            || fail "$out_jar: missing $resource_dir/$backend/$main_lib after zip update"
    done
    echo "$listing" | grep -qF "$resource_dir/jllama-backends.txt" \
        || fail "$out_jar: missing $resource_dir/jllama-backends.txt"
    unzip -p "$out_jar" META-INF/MANIFEST.MF | grep -qF "Main-Class: $MAIN_CLASS" \
        || fail "$out_jar: Main-Class $MAIN_CLASS did not survive the zip update"
    sample_backend="${ordered_backends[0]}"
    unzip -p "$out_jar" "$resource_dir/$sample_backend/$main_lib" \
        | cmp -s - "$staging/$resource_dir/$sample_backend/$main_lib" \
        || fail "$out_jar: $resource_dir/$sample_backend/$main_lib differs from its source"

    (cd "$OUT_DIR" && sha256sum "$(basename "$out_jar")" > "$(basename "$out_jar").sha256")
    echo "OK: $(basename "$out_jar") ($(du -h "$out_jar" | cut -f1); backends: ${ordered_backends[*]})"
done

# --- 7. The default (all-platform CPU) fat jar is a release asset too ------------------
cp "$BASE_FAT_JAR" "$OUT_DIR/"
(cd "$OUT_DIR" && sha256sum "$(basename "$BASE_FAT_JAR")" > "$(basename "$BASE_FAT_JAR").sha256")
echo "OK: $(basename "$BASE_FAT_JAR") (default CPU fat jar, copied as-is)"

ls -lh "$OUT_DIR"
