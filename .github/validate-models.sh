#!/bin/bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
# SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
#
# SPDX-License-Identifier: MIT

# Validate that all required model files exist and are valid GGUF files
# GGUF files start with magic bytes: 0x47 0x47 0x55 0x46 ("GGUF")

set -e

MODELS=(
  "models/codellama-7b.Q2_K.gguf"
  "models/jina-reranker-v1-tiny-en-Q4_0.gguf"
  "models/AMD-Llama-135m-code.Q2_K.gguf"
  "models/Qwen3-0.6B-Q4_K_M.gguf"
)

# Optional GGUFs validated only when present so jobs that do not download
# them (e.g. cross-compile smoke runs) still pass. The vision test image is
# committed to src/test/resources/images/test-image.jpg and is not validated
# here — its presence is asserted directly by MultimodalIntegrationTest.
OPTIONAL_MODELS=(
  "models/nomic-embed-text-v1.5.f16.gguf"
  "models/SmolVLM-500M-Instruct-Q8_0.gguf"
  "models/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"
)

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
