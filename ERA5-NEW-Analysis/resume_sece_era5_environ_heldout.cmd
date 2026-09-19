@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_sece_era5_environ_heldout_resilient.ps1"
