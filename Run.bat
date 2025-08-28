@echo off
cd /d "%~dp0"
echo 🚀 Setting up environment...
python -m pip install --upgrade pip >nul 2>&1
echo 🔍 Checking and installing dependencies...

:: Open Model.py in a new CMD window
start cmd /k "python Model.py"
