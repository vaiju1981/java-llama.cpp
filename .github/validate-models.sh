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

# Optional GGUFs and image, validated only when present so jobs that do not
# download them (e.g. cross-compile smoke runs) still pass.
OPTIONAL_MODELS=(
  "models/nomic-embed-text-v1.5.f16.gguf"
  "models/SmolVLM-500M-Instruct-Q8_0.gguf"
  "models/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"
)

OPTIONAL_IMAGES=(
  "models/Red_Apple.jpg"
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

validate_image() {
  local img="$1"
  if [[ ! -f "$img" ]]; then
    echo "- $img (optional, skipped: not present)"
    return
  fi
  local size
  size=$(stat -f%z "$img" 2>/dev/null || stat -c%s "$img" 2>/dev/null)
  if [[ $size -lt 100 ]]; then
    echo "ERROR: Image file too small (likely an HTML error page): $img (size: $size bytes)"
    exit 1
  fi
  # Accept JPEG (FF D8 FF), PNG (89 50 4E 47), WebP RIFF (52 49 46 46), GIF (47 49 46 38)
  local magic
  magic=$(xxd -p -l 4 "$img")
  case "$magic" in
    ffd8ff*|89504e47|52494646|47494638)
      echo "✓ $img ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo $size bytes))"
      ;;
    *)
      echo "ERROR: Unrecognised image magic in $img (got: $magic)"
      exit 1
      ;;
  esac
}

echo "Validating model files..."
for model in "${MODELS[@]}"; do
  validate_gguf "$model" required
done
for model in "${OPTIONAL_MODELS[@]}"; do
  validate_gguf "$model" optional
done
for img in "${OPTIONAL_IMAGES[@]}"; do
  validate_image "$img"
done

echo "All models validated successfully!"
