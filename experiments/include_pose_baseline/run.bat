@echo off
REM SANKET Live — ek click: venv banao (sirf pehli baar), packages, phir app chalao.
REM Extra options aage pass hote hain, jaise:  run.bat --manual   ya   run.bat --camera 1
cd /d "%~dp0"
if not exist model\config.json (
  echo [!] model\ me config.json nahi mila. Colab wali sanket_include_model.zip ko model\ folder me unzip karo.
  pause & exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo [setup] Python venv bana rahe hain...
  py -3 -m venv .venv 2>nul || python -m venv .venv || (echo [!] Python nahi mila - python.org se 3.11-3.13 install karo & pause & exit /b 1)
)
.venv\Scripts\python -m pip install -q --disable-pip-version-check -r requirements.txt || (echo [!] pip install fail & pause & exit /b 1)
.venv\Scripts\python sanket_live.py %*
pause
