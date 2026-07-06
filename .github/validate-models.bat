REM SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
REM SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
REM
REM SPDX-License-Identifier: MIT

@echo off
REM Validate that all required model files exist and are valid GGUF files
REM GGUF files start with magic bytes: 0x47 0x47 0x55 0x46 ("GGUF")

setlocal enabledelayedexpansion

REM Every CI Java test job (incl. Windows) restores the shared model cache before
REM validating and runs the embedding / vision / TTS integration tests, so all of
REM these are REQUIRED (a missing one is a hard failure, not a silent self-skip).
REM
REM The required set comes from the single source of truth .github\models.csv
REM (filename,url per line; # comments ignored) so this gate can never drift from
REM what download-models actually fetches. The list is built UNQUOTED (no filename
REM contains spaces): an earlier hardcoded form embedded literal quotes into the
REM variable, which made the for-loop see the whole list as ONE token — `if not
REM exist` then parsed it as path-plus-command, so only a fragment was ever checked
REM and the exit /b 1 never fired (the ERROR printed but the step passed; observed
REM in run 28805360584, where an empty model cache sailed through this "gate").
if not exist "%~dp0models.csv" (
  echo ERROR: model manifest not found: %~dp0models.csv
  exit /b 1
)
set MODELS=
for /f "usebackq eol=# tokens=1 delims=," %%N in ("%~dp0models.csv") do set "MODELS=!MODELS! models\%%N"
if "!MODELS!"=="" (
  echo ERROR: no models parsed from %~dp0models.csv
  exit /b 1
)

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
