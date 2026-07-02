@echo off
setlocal

cd /d "%~dp0\.."

if not "%SKIP_RUNTIME_UI_BUILD%"=="1" if exist ..\llm-wiki-ui\package.json (
  npm --prefix ..\llm-wiki-ui run build:runtime
  if errorlevel 1 exit /b %errorlevel%
)

uv sync --extra dev
if errorlevel 1 exit /b %errorlevel%

uv run pyinstaller --clean --noconfirm packaging\runtime\llm-wiki-runtime.spec
if errorlevel 1 exit /b %errorlevel%

if not exist dist\hooks mkdir dist\hooks
xcopy /E /I /Y packaging\runtime\hooks dist\hooks >nul
if errorlevel 1 exit /b %errorlevel%

echo Runtime binary written to dist\llm-wiki-runtime.exe
