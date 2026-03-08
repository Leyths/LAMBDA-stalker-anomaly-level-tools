@echo off
setlocal enabledelayedexpansion

:: Build LAMBDA.exe - standalone Windows executable
::
:: Prerequisites: Python 3.11+ must be available in PATH
::
:: Output: dist\ folder ready to zip and distribute

echo ============================================
echo  Building LAMBDA.exe
echo ============================================
echo.

:: Find python
where python >nul 2>nul
if !errorlevel! equ 0 (
    set "PYTHON=python"
) else (
    echo Error: python not found in PATH
    pause
    exit /b 1
)

echo Using Python: %PYTHON%
%PYTHON% --version
echo.

:: Create/update build venv
if not exist "build_venv\Scripts\python.exe" (
    echo Creating build virtual environment...
    %PYTHON% -m venv build_venv
)

:: Install dependencies
echo Installing build dependencies...
build_venv\Scripts\pip install --quiet pyinstaller numpy open3d
echo.

:: Clean previous dist
if exist "dist" (
    echo Cleaning previous dist...
    rmdir /S /Q dist
)

:: Run PyInstaller
echo Running PyInstaller...
build_venv\Scripts\pyinstaller lambda.spec --noconfirm
echo.

if !errorlevel! neq 0 (
    echo.
    echo Build FAILED.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Assembling distribution...
echo ============================================
echo.

:: Config files
echo Copying config files...
copy /Y levels.ini dist\  >nul
copy /Y anomaly.ini dist\  >nul
copy /Y cultured.ini dist\  >nul
copy /Y gamma.ini dist\  >nul
copy /Y spawn_blacklist.ini dist\  >nul
copy /Y level_changers.ini dist\  >nul

:: Anomaly spawn data
echo Copying anomaly data...
xcopy /E /I /Y /Q anomaly dist\anomaly >nul

:: Levels (only level.ai, level.spawn, level.game per level)
echo Copying level data...
for /D %%L in (levels\*) do (
    mkdir "dist\%%L" 2>nul
    if exist "%%L\level.ai"    copy /Y "%%L\level.ai"    "dist\%%L\" >nul
    if exist "%%L\level.spawn" copy /Y "%%L\level.spawn" "dist\%%L\" >nul
    if exist "%%L\level.game"  copy /Y "%%L\level.game"  "dist\%%L\" >nul
)

:: Mods
echo Copying mods...
xcopy /E /I /Y /Q mods dist\mods >nul

:: Docs and README
echo Copying docs...
xcopy /E /I /Y /Q docs dist\docs >nul
if exist "README.md" copy /Y README.md dist\ >nul

echo.
echo ============================================
echo  Build complete: dist\
echo ============================================
echo.
echo The dist\ folder is ready to zip and distribute.
echo.
pause
