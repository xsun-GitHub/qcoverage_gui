@echo off
cd /d "D:\Data\code\qcoverage_gui"

if not exist "qcoverage_v3.py" (
    echo ERROR: Cannot find:
    echo D:\Data\code\qcoverage_gui\qcoverage_v3.py
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "qcoverage_v3.py"
) else (
    python "qcoverage_v3.py"
)

if errorlevel 1 pause