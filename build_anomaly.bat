@echo off
setlocal enabledelayedexpansion

:: Build all.spawn - main game graph compilation script
::
:: Usage:
::   build_anomaly.bat              Normal build
::   build_anomaly.bat --force      Force rebuild all cross tables

:: Find python - prefer virtual environment if available
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON=python"
    ) else (
        echo Error: python not found
        echo.
        echo Make sure Python 3.12 is installed and available in your PATH.
        echo If using pyenv, run: pyenv shell 3.12.0
        pause
        exit /b 1
    )
)

:: Show which python we're using
echo Using Python: %PYTHON%
%PYTHON% --version

:: Check for required dependencies
%PYTHON% -c "import numpy" >nul 2>nul
if !errorlevel! neq 0 (
    echo.
    echo Error: numpy is not installed
    echo Install it with: %PYTHON% -m pip install numpy
    pause
    exit /b 1
)

cd /d "%~dp0compiler" || (
    echo Error: failed to change to compiler directory
    pause
    exit /b 1
)

%PYTHON% build_all_spawn.py ^
    --config ../levels.ini ^
    --output ../gamedata/spawns/all.spawn ^
    --blacklist ../spawn_blacklist.ini ^
    --basemod anomaly ^
    %*

if !errorlevel! neq 0 (
    echo.
    echo Build failed (code: !errorlevel!).
    pause
    exit /b !errorlevel!
)
