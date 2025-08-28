@echo off
echo 🚀 Launching Streamlit App...

:: Kill existing processes to ensure a clean start
taskkill /F /IM streamlit.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1

:: Run Streamlit in the background without opening a new console window
start "" /B pythonw -m streamlit run app.py --server.headless true --server.port 8501 --server.fileWatcherType none

:: Give Streamlit some time to start
timeout /t 5 >nul

:: Open the app in the default browser
start "" http://localhost:8501

exit
