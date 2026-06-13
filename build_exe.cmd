@echo off
setlocal

uv run pyinstaller --noconfirm --clean --windowed --onefile --name OscilloscopeViewer main.py
if errorlevel 1 exit /b %errorlevel%

echo Built dist\OscilloscopeViewer.exe
