@echo off
setlocal

cd /d "%~dp0\.."

uv sync --extra dev
if errorlevel 1 exit /b %errorlevel%

uv run pyinstaller --clean --noconfirm packaging\runtime\llm-wiki-runtime.spec
if errorlevel 1 exit /b %errorlevel%

echo Runtime binary written to dist\llm-wiki-runtime.exe

