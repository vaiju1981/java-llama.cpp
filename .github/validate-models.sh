#!/bin/bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
# SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
#
# SPDX-License-Identifier: MIT

# Validate that all required model files exist and are valid GGUF files
# GGUF files start with magic bytes: 0x47 0x47 0x55 0x46 ("GGUF")

set -e

# Every CI Java test job (Linux + all macOS + all Windows) restores the shared model
# cache before validating, and runs the embedding / vision / TTS integration tests
# with their properties set — so all of these are REQUIRED, not optional. A missing
# model is a hard failure here (it would otherwise let an integration test silently
# self-skip). The required set comes from the single source of truth
# .github/models.csv (filename,url per line; # comments ignored) so this gate can
# never drift from what download-models actually fetches.
MODELS_CSV="$(dirname "$0")/models.csv"
[ -f "${MODELS_CSV}" ] || { echo "ERROR: model manifest not found: ${MODELS_CSV}"; exit 1; }
MODELS=()
while IFS=, read -r name _url; do
  case "$name" in ''|\#*) continue ;; esac
  MODELS+=("models/$name")
done < "${MODELS_CSV}"
[ "${#MODELS[@]}" -gt 0 ] || { echo "ERROR: no models parsed from ${MODELS_CSV}"; exit 1; }

# Optional GGUFs validated only when present. The vision test image is committed to
# src/test/resources/images/test-image.jpg and is not validated here — its presence
# is asserted directly by MultimodalIntegrationTest. The audio-input model
# (AudioInputIntegrationTest) has no committed clip and no CI download, so that test
# self-skips and its model is intentionally not listed here.
OPTIONAL_MODELS=()

validate_gguf() {
  local model="$1"
  local required="$2"
  if [[ ! -f "$model" ]]; then
    if [[ "$required" == "required" ]]; then
      echo "ERROR: Model not found: $model"
      exit 1
    else
      echo "- $model (optional, skipped: not present)"
      return
    fi
  fi
  local size
  size=$(stat -f%z "$model" 2>/dev/null || stat -c%s "$model" 2>/dev/null)
  if [[ $size -lt 4 ]]; then
    echo "ERROR: Model file too small (likely corrupted): $model (size: $size bytes)"
    exit 1
  fi
  local magic
  magic=$(xxd -p -l 4 "$model")
  if [[ "$magic" != "47475546" ]]; then
    echo "ERROR: Invalid GGUF magic bytes in $model (got: $magic, expected: 47475546)"
    exit 1
  fi
  echo "✓ $model ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo $size bytes))"
}

echo "Validating model files..."
for model in "${MODELS[@]}"; do
  validate_gguf "$model" required
done
for model in "${OPTIONAL_MODELS[@]}"; do
  validate_gguf "$model" optional
done

echo "All models validated successfully!"
