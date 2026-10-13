@echo off
REM ============================================================================
REM  GodhomeRL -- one-command setup for Windows (cmd.exe)
REM
REM  Usage:
REM      setup.bat                                  build into the current env
REM      setup.bat --venv .venv                      build into a new venv
REM      setup.bat --install "C:\Games\godhome-..."  explicit game location
REM
REM  After it finishes:
REM      python -m godhome_rl doctor
REM      python -m godhome_rl train --name godhome --bosses all --n-envs 3
REM ============================================================================

setlocal EnableDelayedExpansion

REM ---------------------------------------------------------------------------
REM 0. Locate the repository root and a Python interpreter
REM ---------------------------------------------------------------------------
set "ROOT=%~dp0"
pushd "%ROOT%" || exit /b 1

set "PY="
for %%P in (py python python3) do (
    if not defined PY (
        %%P --version >nul 2>&1
        if !errorlevel! equ 0 set "PY=%%P"
    )
)
if not defined PY (
    echo [setup] No Python found.  Install Python 3.10+ from https://python.org
    echo [setup] and make sure "Add python.exe to PATH" is ticked.
    popd
    exit /b 1
)
for /f "tokens=*" %%V in ('%PY% --version 2^>^&1') do set "PYVER=%%V"
echo [setup] using %PY% %PYVER%

REM ---------------------------------------------------------------------------
REM 1. Parse arguments
REM ---------------------------------------------------------------------------
set "VENV="
set "INSTALL="
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--venv" (
    set "VENV=%~2"
    shift
    shift
    goto parse
)
if /i "%~1"=="--install" (
    set "INSTALL=%~2"
    shift
    shift
    goto parse
)
echo [setup] ignoring unknown argument "%~1"
shift
goto parse
:parsed

REM ---------------------------------------------------------------------------
REM 2. Optional virtual environment
REM ---------------------------------------------------------------------------
if defined VENV (
    if not exist "%VENV%\Scripts\python.exe" (
        echo [setup] creating a virtual environment at %VENV%
        %PY% -m venv "%VENV%" || (echo [setup] venv creation failed & popd & exit /b 1)
    )
    set "PY=%VENV%\Scripts\python.exe"
)

REM ---------------------------------------------------------------------------
REM 3. Dependencies
REM ---------------------------------------------------------------------------
echo [setup] installing dependencies
%PY% -m pip install --upgrade pip --quiet
%PY% -m pip install -e . --quiet || (echo [setup] dependency install failed & popd & exit /b 1)

REM ---------------------------------------------------------------------------
REM 4. Locate the Godhome install
REM ---------------------------------------------------------------------------
set "INSTALL_ARG="
if defined INSTALL set "INSTALL_ARG=--install %INSTALL%"
if not defined INSTALL if defined GODHOME_RL_INSTALL set "INSTALL_ARG=--install %GODHOME_RL_INSTALL%"

echo [setup] building the patched game (first run unpacks ~4000 assets)
%PY% -m godhome_rl build %INSTALL_ARG% || (echo [setup] build failed & popd & exit /b 1)

REM ---------------------------------------------------------------------------
REM 5. Verify
REM ---------------------------------------------------------------------------
%PY% -m godhome_rl doctor || (echo [setup] doctor reported problems & popd & exit /b 1)

echo.
echo [setup] done.
if defined VENV echo [setup] activate with:  %VENV%\Scripts\activate
echo [setup] then run:  python -m godhome_rl train --name godhome --bosses all --n-envs 3
echo [setup]           python -m godhome_rl play  --model runs/godhome/checkpoints/latest.pt

popd
endlocal
exit /b 0