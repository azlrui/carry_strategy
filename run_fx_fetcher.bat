@echo off
REM ==========================================================
REM FX Project Runner - Master .bat file at project root
REM ==========================================================
REM Usage:
REM   .\run_fx_fetcher.bat "data\physical_currency_list.csv" USD 2001-01-01 2025-01-01 1mo "data/"
REM ==========================================================

set FX_FILE=%~1
set TO_CUR=%~2
set START=%~3
set END=%~4
set FREQ=%~5
set OUTDIR=%~6

IF "%FX_FILE%"=="" (
  echo.
  echo Usage: run_fx_fetcher.bat "data\physical_currency_list.csv" USD 2001-01-01 2025-01-01 1d "data/"
  echo.
  exit /b 1
)

echo [INFO] Running FX Data Fetcher...
python src\fetch_fx_data.py ^
  --fx-file "%FX_FILE%" ^
  --to "%TO_CUR%" ^
  --start "%START%" ^
  --end "%END%" ^
  --freq "%FREQ%" ^
  --outdir "%OUTDIR%"

IF %ERRORLEVEL% NEQ 0 (
  echo [ERROR] FX fetcher failed with code %ERRORLEVEL%.
  exit /b %ERRORLEVEL%
)

echo [INFO] FX Data Fetching complete.
echo All operations finished successfully.
exit /b 0
