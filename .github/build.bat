REM SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
REM SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
REM
REM SPDX-License-Identifier: MIT

@echo off
setlocal enabledelayedexpansion

REM The core project (CMakeLists.txt + src\) lives in the `llama\` module of the Maven
REM reactor. Re-root here once (relative to this script's own location) so cmake
REM configures the module regardless of the caller's CWD.
cd /d "%~dp0..\llama" || exit /b 1

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

REM NOTE: nvcc is NOT wrapped with sccache on Windows. Unlike build.sh (Linux) -- where
REM sccache caches the per-arch .cu device passes -- sccache on Windows cannot parse the
REM nvcc command line (it dies with `sccache: error: Could not parse shell line` and
REM fails every .cu compile). So CUDA device code is built by nvcc directly (uncached)
REM here; the cl.exe C/C++ TUs still cache via the C/CXX launcher set above.

mkdir build
cmake -Bbuild %LAUNCH% %*
if errorlevel 1 exit /b 1
cmake --build build --config Release
set "BUILD_RC=!ERRORLEVEL!"

REM Print cache stats (best-effort) regardless of build outcome -- only when sccache
REM was wired in as the launcher.
if defined LAUNCH (
    echo build.bat: sccache --show-stats
    sccache --show-stats
    REM KISS per-job cache summary in the GitHub Actions job summary (like upstream llama.cpp's
    REM ccache-action table). Parse the text stats: the top-level "Compile requests" line is the
    REM total and the top-level "Cache hits" line is the hits (the per-language "Cache hits (C/C++)"
    REM line has "(" after the label, so the digit-anchored findstr regex skips it). Only in CI
    REM (GITHUB_STEP_SUMMARY set); local runs are untouched. Best-effort -- skipped if the two
    REM numbers can't be parsed or there were no requests. Integer math with rounding to one decimal.
    if defined GITHUB_STEP_SUMMARY (
        set "SCC_REQ="
        set "SCC_HITS="
        for /f "tokens=3" %%a in ('sccache --show-stats 2^>nul ^| findstr /r /c:"^Compile requests  *[0-9]"') do set "SCC_REQ=%%a"
        for /f "tokens=3" %%a in ('sccache --show-stats 2^>nul ^| findstr /r /c:"^Cache hits  *[0-9]"') do set "SCC_HITS=%%a"
        if defined SCC_REQ if defined SCC_HITS if !SCC_REQ! gtr 0 (
            set /a SCC_RATE10=^(!SCC_HITS! * 1000 + !SCC_REQ! / 2^) / !SCC_REQ!
            set /a SCC_WHOLE=!SCC_RATE10! / 10
            set /a SCC_DEC=!SCC_RATE10! %% 10
            >>"%GITHUB_STEP_SUMMARY%" echo ### sccache statistics
            >>"%GITHUB_STEP_SUMMARY%" echo.
            >>"%GITHUB_STEP_SUMMARY%" echo ^| Cache hits ^| Requests ^| Hit rate ^|
            >>"%GITHUB_STEP_SUMMARY%" echo ^|------------^|----------^|----------^|
            >>"%GITHUB_STEP_SUMMARY%" echo ^| !SCC_HITS! ^| !SCC_REQ! ^| !SCC_WHOLE!.!SCC_DEC!%% ^|
        )
    )
)

REM Propagate a build failure as a non-zero exit (a prior bug let a failed `cmake
REM --build` reach here and exit 0, masquerading as a green build with no artifacts).
if not "!BUILD_RC!"=="0" (
    echo build.bat: cmake --build failed with exit code !BUILD_RC!.
    exit /b !BUILD_RC!
)
