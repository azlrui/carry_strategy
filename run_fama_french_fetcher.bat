@echo off
REM ==========================================================
REM Fama-French Factors Runner (.bat at project root)
REM Calls: src\fetch_fama_french_data.py with CLI args
REM Usage:
REM   .\run_fama_french_fetcher.bat 2001-01-01 2025-01-01 month "data/" [--as-decimal]
REM   or with aliases: 1mo / 1w / 1d
REM ==========================================================

set START=%~1
set END=%~2
set FREQ=%~3
set OUTDIR=%~4

IF "%START%"=="" (
  echo.
  echo Usage: run_fama_french_fetcher.bat START END FREQ OUTDIR [--as-decimal]
  echo Example: run_fama_french_fetcher.bat 2001-01-01 2025-01-01 1mo "data/"
  echo.
  exit /b 1
)

REM Optional: activate virtualenv if present
IF EXIST ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

REM Pass through a 5th argument if provided (e.g., --as-decimal)
set EXTRA=%~5

python src\fetch_fama_french_data.py ^
  --start "%START%" ^
  --end "%END%" ^
  --freq "%FREQ%" ^
  --outdir "%OUTDIR%" %EXTRA%

IF %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Fama-French fetcher failed with code %ERRORLEVEL%.
  exit /b %ERRORLEVEL%
)

echo [INFO] Fama-French factors download complete.
exit /b 0
