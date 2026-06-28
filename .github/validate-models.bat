REM SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
REM SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
REM
REM SPDX-License-Identifier: MIT

@echo off
REM Validate that all required model files exist and are valid GGUF files
REM GGUF files start with magic bytes: 0x47 0x47 0x55 0x46 ("GGUF")

setlocal enabledelayedexpansion

REM Every CI Java test job (incl. Windows) now downloads the full model set before
REM validating and runs the embedding / vision / TTS integration tests, so all of
REM these are REQUIRED (a missing one is a hard failure, not a silent self-skip).
set "MODELS=models\codellama-7b.Q2_K.gguf" "models\jina-reranker-v1-tiny-en-Q4_0.gguf" "models\AMD-Llama-135m-code.Q2_K.gguf" "models\Qwen3-0.6B-Q4_K_M.gguf" "models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" "models\nomic-embed-text-v1.5.f16.gguf" "models\SmolVLM-500M-Instruct-Q8_0.gguf" "models\mmproj-SmolVLM-500M-Instruct-Q8_0.gguf" "models\OuteTTS-0.2-500M-Q4_K_M.gguf" "models\WavTokenizer-Large-75-F16.gguf"

REM No optional models remain (the audio-input model has no CI download and its
REM test self-skips). Left empty so the optional loop below is a no-op.
set "OPTIONAL_MODELS="

echo Validating required model files...
for %%M in (%MODELS%) do (
  if not exist "%%M" (
    echo ERROR: Model not found: %%M
    exit /b 1
  )

  REM Check file size using PowerShell
  for /f %%S in ('powershell -Command "(Get-Item '%%M').Length"') do set "size=%%S"

  if !size! lss 4 (
    echo ERROR: Model file too small (likely corrupted^): %%M (size: !size! bytes^)
    exit /b 1
  )

  REM Check GGUF magic bytes using PowerShell: 47475546 in hex = GGUF in ASCII
  for /f %%H in ('powershell -Command "[System.BitConverter]::ToString((Get-Content '%%M' -Encoding Byte -ReadCount 4)[0]) -replace '-',''"') do set "magic=%%H"

  if not "!magic!"=="47475546" (
    echo ERROR: Invalid GGUF magic bytes in %%M (got: !magic!, expected: 47475546^)
    exit /b 1
  )

  echo OK: %%M ^(!size! bytes^)
)

echo Validating optional vision model files...
for %%M in (%OPTIONAL_MODELS%) do (
  if not exist "%%M" (
    echo SKIP: %%M not present
  ) else (
    for /f %%S in ('powershell -Command "(Get-Item '%%M').Length"') do set "size=%%S"
    if !size! lss 4 (
      echo ERROR: Model file too small (likely corrupted^): %%M (size: !size! bytes^)
      exit /b 1
    )
    for /f %%H in ('powershell -Command "[System.BitConverter]::ToString((Get-Content '%%M' -Encoding Byte -ReadCount 4)[0]) -replace '-',''"') do set "magic=%%H"
    if not "!magic!"=="47475546" (
      echo ERROR: Invalid GGUF magic bytes in %%M (got: !magic!, expected: 47475546^)
      exit /b 1
    )
    echo OK: %%M ^(!size! bytes^)
  )
)

echo All models validated successfully!
exit /b 0
