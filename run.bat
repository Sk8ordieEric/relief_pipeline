@echo off
REM relief_pipeline launcher. Finds a Python that has numpy/opencv/scipy and runs
REM whatever you pass it, so you don't have to remember which interpreter works.
REM
REM   run.bat check_setup.py
REM   run.bat relief_build.py --depth ... --mask ... --out projects\rex\rex_v1
REM   run.bat inlay_build.py  --mask ...  --out projects\oak_raven\inlay

setlocal
cd /d "%~dp0"

set "PY="
REM ComfyUI's embedded Python already has all three packages - most reliable bet.
if exist "%~dp0..\ComfyUI_windows_portable\python_embeded\python.exe" (
  set "PY=%~dp0..\ComfyUI_windows_portable\python_embeded\python.exe"
)
if not defined PY if exist "%USERPROFILE%\Anaconda3\python.exe" set "PY=%USERPROFILE%\Anaconda3\python.exe"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Could not find a Python interpreter. Install Python 3.10+ or edit this file.
  exit /b 1
)

echo Using: %PY%
"%PY%" %*
endlocal
