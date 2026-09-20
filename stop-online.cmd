@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_online.ps1"
if errorlevel 1 pause
