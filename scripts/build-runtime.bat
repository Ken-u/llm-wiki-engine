@echo off
setlocal

cd /d "%~dp0\.."

uv sync --extra dev
if errorlevel 1 exit /b %errorlevel%

uv run pyinstaller --clean --noconfirm packaging\runtime\llm-wiki-runtime.spec
if errorlevel 1 exit /b %errorlevel%

if not exist dist\hooks mkdir dist\hooks
xcopy /E /I /Y packaging\runtime\hooks dist\hooks >nul
if errorlevel 1 exit /b %errorlevel%

echo Runtime binary written to dist\llm-wiki-runtime.exe
