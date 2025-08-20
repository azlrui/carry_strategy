@echo off
REM ==========================================================
REM Global Indices Runner (.bat at project root)
REM Calls your Python fetcher with CLI args
REM Usage (minimal):
REM   .\run_index_fetcher.bat 2001-01-01 2025-01-01 1mo "data\"
REM
REM Usage (with .env + exclusions + subset):
REM   .\run_index_fetcher.bat 2001-01-01 2025-01-01 1mo "data\" ".env" valid_indices fundamentally_excluded "USD,EUR,GBP"
REM ==========================================================

REM ---- Script name (change if needed) ----
REM If your file is src\fetch_stock_market_data.py, set that below.
set SCRIPT=src\fetch_stock_market_data.py

set START=%~1
set END=%~2
set FREQ=%~3
set OUTDIR=%~4

REM Optional args
set ENVFILE=%~5
set ENVKEY=%~6
set EXCLUDEKEY=%~7
set CURR=%~8

IF "%START%"=="" (
  echo.
  echo Usage: run_index_fetcher.bat START END FREQ OUTDIR [ENV_FILE] [ENV_KEY] [EXCLUDE_KEY] [CURRENCIES]
  echo Example 1: run_index_fetcher.bat 2001-01-01 2025-01-01 1mo "data\indexes"
  echo Example 2: run_index_fetcher.bat 2001-01-01 2025-01-01 1mo "data\indexes" ".env" valid_indices fundamentally_excluded "USD,EUR,GBP"
  echo.
  exit /b 1
)

REM Optional: activate virtualenv if present
IF EXIST ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

REM Build optional flags
set EXTRA=
IF NOT "%ENVFILE%"=="" (
  set EXTRA=%EXTRA% --env-file "%ENVFILE%"
)
IF NOT "%ENVKEY%"=="" (
  set EXTRA=%EXTRA% --env-key "%ENVKEY%"
)
IF NOT "%EXCLUDEKEY%"=="" (
  set EXTRA=%EXTRA% --exclude-key "%EXCLUDEKEY%"
)
IF NOT "%CURR%"=="" (
  set EXTRA=%EXTRA% --currencies "%CURR%"
)

echo [INFO] Running Index fetcher...
python %SCRIPT% ^
  --start "%START%" ^
  --end "%END%" ^
  --freq "%FREQ%" ^
  --outdir "%OUTDIR%" %EXTRA%

IF %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Index fetcher failed with code %ERRORLEVEL%.
  exit /b %ERRORLEVEL%
)

echo [INFO] Index data download complete.
exit /b 0
