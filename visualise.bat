@echo off
setlocal

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
    if %errorlevel% equ 0 (
        set "PYTHON=python"
    ) else (
        echo Error: python not found >&2
        pause
        exit /b 1
    )
)

:: Check for required dependencies
%PYTHON% -c "import open3d" >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: open3d is not installed >&2
    echo Install it with: %PYTHON% -m pip install open3d >&2
    pause
    exit /b 1
)

cd /d "%~dp0visualiser" || (
    echo Error: failed to change to visualiser directory >&2
    pause
    exit /b 1
)

%PYTHON% run_visualiser.py %*

if %errorlevel% neq 0 (
    pause
    exit /b %errorlevel%
)
