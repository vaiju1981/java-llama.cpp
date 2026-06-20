REM SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
REM SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
REM
REM SPDX-License-Identifier: MIT

@echo off
setlocal enabledelayedexpansion

REM ---------------------------------------------------------------------------
REM Optional shared compiler cache: sccache fronting Depot Cache (WebDAV).
REM Mirrors build.sh's sccache_can_wrap_compiler() probe. Because sccache *is*
REM the compiler launcher (cmake runs `sccache cl.exe ...` for every TU), a
REM present-but-crashing sccache would red every build; so we trust it only after
REM a trivial TU compiles *through* it. Enabled only when USE_CACHE=true AND
REM sccache is on PATH AND the probe succeeds; otherwise the build proceeds
REM uncached and green. The Visual Studio generator jobs do NOT set USE_CACHE, so
REM this stays inert for them (and the VS generator ignores
REM CMAKE_{C,CXX}_COMPILER_LAUNCHER anyway -- only Ninja/Makefiles honor it).
REM ---------------------------------------------------------------------------
set "LAUNCH="
if /I "%USE_CACHE%"=="true" (
    where sccache >nul 2>&1
    if errorlevel 1 (
        echo build.bat: USE_CACHE=true but sccache not on PATH; building WITHOUT cache.
    ) else (
        set "PROBE_DIR=%TEMP%\sccache-probe-%RANDOM%"
        mkdir "!PROBE_DIR!" >nul 2>&1
        (echo int sccache_probe_verify = 0;)> "!PROBE_DIR!\probe.c"
        sccache cl.exe /nologo /c "!PROBE_DIR!\probe.c" /Fo"!PROBE_DIR!\probe.obj" > "!PROBE_DIR!\probe.log" 2>&1
        if errorlevel 1 (
            echo build.bat: sccache probe FAILED wrapping cl.exe -- building WITHOUT cache.
            type "!PROBE_DIR!\probe.log"
        ) else (
            echo build.bat: sccache probe OK ^(wrapped cl.exe^).
            set "LAUNCH=-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
        )
        rmdir /s /q "!PROBE_DIR!" >nul 2>&1
    )
)

mkdir build
cmake -Bbuild %LAUNCH% %*
if errorlevel 1 exit /b %ERRORLEVEL%
cmake --build build --config Release
if errorlevel 1 exit /b %ERRORLEVEL%

REM Only query stats when sccache was actually wired in as the launcher; re-invoking
REM a rejected/crashing sccache here would just repeat its failure output.
if defined LAUNCH (
    echo build.bat: sccache --show-stats
    sccache --show-stats
)
