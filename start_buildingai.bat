@echo off
setlocal
title BuildingAI Desktop

rem Always start from the directory containing this launcher.
cd /d "%~dp0"

rem Prefer a project-owned virtual environment when one is available.
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    set "BUILDINGAI_ENV=.venv"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
    set "BUILDINGAI_ENV=venv"
) else (
    set "BUILDINGAI_ENV=Python from PATH"
)

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Python was not found.
    echo Create .venv or venv in this project, or install Python and add it to PATH.
    echo.
    pause
    exit /b 1
)

echo Starting BuildingAI with %BUILDINGAI_ENV%...
python app.py
set "BUILDINGAI_EXIT=%ERRORLEVEL%"

if not "%BUILDINGAI_EXIT%"=="0" (
    echo.
    echo [ERROR] BuildingAI exited with code %BUILDINGAI_EXIT%.
    echo Review the error message above, then press any key to close this window.
    pause
)

exit /b %BUILDINGAI_EXIT%
