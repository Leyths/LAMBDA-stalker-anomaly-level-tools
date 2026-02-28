@echo off
setlocal enabledelayedexpansion

:: Launch the node graph visualiser
::
:: Usage:
::   visualise.bat

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
%PYTHON% -c "import open3d" >nul 2>nul
if !errorlevel! neq 0 (
    echo.
    echo Error: open3d is not installed
    echo Install it with: %PYTHON% -m pip install open3d
    pause
    exit /b 1
)

cd /d "%~dp0visualiser" || (
    echo Error: failed to change to visualiser directory
    pause
    exit /b 1
)

echo Starting visualiser...
echo.
%PYTHON% run_visualiser.py %*

if !errorlevel! neq 0 (
    echo.
    echo Visualiser exited with an error (code: !errorlevel!).
    pause
    exit /b !errorlevel!
)
