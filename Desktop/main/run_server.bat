@echo off
REM Batch script to run the FastAPI server for unified_api.py

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Change directory to the project folder
cd /d %~dp0

REM Run the FastAPI server
python unified_api.py

REM Pause to keep the window open after server stops
pause
