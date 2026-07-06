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

set OUTDIR=dist\runtime\windows-x86_64
set ZIP_PATH=dist\runtime-windows-x86_64.zip

if exist "%OUTDIR%" rmdir /S /Q "%OUTDIR%"
if exist "%ZIP_PATH%" del "%ZIP_PATH%"
mkdir "%OUTDIR%"

copy /Y dist\llm-wiki-runtime.exe "%OUTDIR%\llm-wiki-runtime.exe" >nul
copy /Y runtime-config.example.yaml "%OUTDIR%\runtime-config.example.yaml" >nul
copy /Y scripts\build-runtime-bundle.sh "%OUTDIR%\build-runtime-bundle.sh" >nul
copy /Y scripts\build-runtime-bundle.bat "%OUTDIR%\build-runtime-bundle.bat" >nul
if not exist "%OUTDIR%\hooks" mkdir "%OUTDIR%\hooks"
xcopy /E /I /Y packaging\runtime\hooks "%OUTDIR%\hooks" >nul
if errorlevel 1 exit /b %errorlevel%

python -c "from pathlib import Path; from zipfile import ZipFile, ZIP_DEFLATED; platform_dir='windows-x86_64'; outdir=Path(r'%OUTDIR%'); zip_path=Path(r'%ZIP_PATH%'); zf=ZipFile(zip_path,'w',ZIP_DEFLATED); [zf.write(p, Path(platform_dir) / p.relative_to(outdir)) for p in sorted(outdir.rglob('*')) if p.is_file()]; zf.close()"
if errorlevel 1 exit /b %errorlevel%

echo Runtime binary written to %OUTDIR%\llm-wiki-runtime.exe
echo Runtime package written to %ZIP_PATH%
