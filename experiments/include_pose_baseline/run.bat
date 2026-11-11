@echo off
REM SANKET Live - ek click: venv banao (sirf pehli baar), packages, phir app chalao.
REM Extra options aage pass hote hain, jaise:  run.bat --manual   ya   run.bat --camera 1
REM Kyun uv: system pe naya Python (jaise 3.15) ho to mediapipe/onnxruntime ke packages nahi milte.
REM          uv apna alag Python 3.13 download karke venv banata hai; system Python ko chhuta nahi.
cd /d "%~dp0"
if not exist model\config.json (
  echo [!] model\ me config.json nahi mila. Colab wali sanket_include_model.zip ko model\ folder me unzip karo.
  pause & exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo [setup] uv + Python 3.13 venv bana rahe hain - sirf pehli baar...
  python -m pip install -q --disable-pip-version-check uv || (echo [!] Python nahi mila - python.org se install karo & pause & exit /b 1)
  python -m uv venv --python 3.13 .venv || (echo [!] venv nahi bana & pause & exit /b 1)
)
python -m uv pip install -q --python .venv\Scripts\python.exe -r requirements.txt || (echo [!] packages install fail & pause & exit /b 1)
.venv\Scripts\python sanket_live.py %*
pause
