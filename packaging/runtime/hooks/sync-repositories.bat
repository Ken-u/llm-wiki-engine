@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "CONFIG_PATH=%~1"
if "%CONFIG_PATH%"=="" set "CONFIG_PATH=%RUNTIME_CONFIG%"
if "%CONFIG_PATH%"=="" set "CONFIG_PATH=runtime-config.yaml"

where python >nul 2>nul
if %errorlevel%==0 (
  python "%SCRIPT_DIR%sync-repositories.py" "%CONFIG_PATH%"
  exit /b %errorlevel%
)

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%SCRIPT_DIR%sync-repositories.py" "%CONFIG_PATH%"
  exit /b %errorlevel%
)

echo python or py is required to run sync-repositories.py 1>&2
exit /b 127
