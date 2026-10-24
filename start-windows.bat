@echo off
rem ZenTrainer launcher for Windows (on Linux: start-linux.sh).
rem
rem It does two things, then hands over to zt (cli\: one Python program for both systems):
rem   1. makes sure uv is installed (uv is a Python manager: it downloads the right Python itself)
rem   2. runs zt with that Python: first-run setup, update check, the menu
rem
rem Arguments pass through to zt, e.g. "start-windows.bat start --open". Only Git for
rem Windows is needed beforehand. Nothing is installed system-wide: uv goes into your user folder.
setlocal
cd /d "%~dp0"

rem ---- settings ------------------------------------------------------------------------------------
rem uv's official installer (https://docs.astral.sh/uv/), and where it puts uv
set "UV_INSTALLER=https://astral.sh/uv/install.ps1"
set "UV_DIR=%USERPROFILE%\.local\bin"
rem zt, the Python program that does the rest
set "CLI=cli"
rem the Python version comfy.lock pins (the managed ComfyUI uses the same one)
set "PYTHON_VERSION=3.12"
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "(Get-Content comfy.lock -Raw | ConvertFrom-Json).python"`) do set "PYTHON_VERSION=%%p"

rem ---- 1. uv ---------------------------------------------------------------------------------------
set "PATH=%UV_DIR%;%PATH%"
where uv >nul 2>nul && goto have_uv
echo uv isn't installed. ZenTrainer uses it to get Python %PYTHON_VERSION% and its packages.
echo   installer: %UV_INSTALLER%
echo   installs to: %UV_DIR%
set "answer=y"
set /p answer="Install uv now? [Y/n] "
if /i "%answer:~0,1%"=="n" (
  echo Not installed. Install uv yourself, then run this again.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy ByPass -Command "irm %UV_INSTALLER% | iex"
if errorlevel 1 (
  echo The uv installer failed.
  pause
  exit /b 1
)
:have_uv

rem ---- 2. hand over to zt -------------------------------------------------------------------------
rem uv downloads Python %PYTHON_VERSION% and zt's packages (cli\uv.lock) the first time, into cli\.venv,
rem then reuses them. --frozen: use the lock as it is, never rewrite it.
uv run --quiet --frozen --project %CLI% --python %PYTHON_VERSION% zt %*
set "code=%errorlevel%"
rem a double-clicked window would close at once: keep it open to read when something went wrong
if not "%code%"=="0" pause
exit /b %code%
