@echo off
REM ==========================================================
REM US T-Bill Coupon-Equivalent Runner (.bat at project root)
REM Calls: src\fetch_us_tbill.py with CLI args
REM Usage:
REM   .\run_us_tbill_fetcher.bat 2001-01-01 2025-01-01 13 "data/"
REM where 13 is the tenor in weeks (4,6,8,13,17,26,52)
REM ==========================================================

set START=%~1
set END=%~2
set TENOR=%~3
set OUTDIR=%~4

IF "%START%"=="" (
  echo.
  echo Usage: run_us_tbill_fetcher.bat START END TENOR OUTDIR
  echo Example: run_tbill.bat 2001-01-01 2025-01-01 13 "data/"
  echo.
  exit /b 1
)

REM Optional: activate virtualenv if present
IF EXIST ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

python src\fetch_us_tbill.py ^
  --start "%START%" ^
  --end "%END%" ^
  --tenor %TENOR% ^
  --outdir "%OUTDIR%"

IF %ERRORLEVEL% NEQ 0 (
  echo [ERROR] T-Bill fetcher failed with code %ERRORLEVEL%.
  exit /b %ERRORLEVEL%
)

echo [INFO] T-Bill data download complete.
exit /b 0
