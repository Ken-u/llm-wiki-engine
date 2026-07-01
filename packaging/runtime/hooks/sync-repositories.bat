@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "RUNTIME_DIR=%SCRIPT_DIR%.."
set "CONFIG_PATH=%~1"
if "%CONFIG_PATH%"=="" set "CONFIG_PATH=%RUNTIME_DIR%\runtime-config.yaml"

"%RUNTIME_DIR%\llm-wiki-runtime.exe" --config "%CONFIG_PATH%" --sync-repositories
exit /b %errorlevel%
