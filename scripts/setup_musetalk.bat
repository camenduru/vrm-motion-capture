@echo off
setlocal
cd /d "%~dp0.."

REM MuseTalk needs its own interpreter: this app runs on Python 3.13 while the
REM MuseTalk stack (torch/diffusers/transformers) lives on 3.10/3.11.
set "PY=%MUSETALK_PYTHON%"
if "%PY%"=="" if exist "MuseTalk\.venv\Scripts\python.exe" set "PY=MuseTalk\.venv\Scripts\python.exe"
if "%PY%"=="" if exist "MuseTalk\.musetalk_python" set /p PY=<MuseTalk\.musetalk_python
if "%PY%"=="" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"

if not exist "%PY%" (
  echo [!] No MuseTalk Python found.
  echo     Install Python 3.10, or set MUSETALK_PYTHON to its python.exe.
  exit /b 1
)
echo [i] MuseTalk interpreter: %PY%

echo == torch check ==
"%PY%" -c "import torch" 2>nul
if errorlevel 1 (
  echo [i] installing torch ^(CUDA 12.6^)
  "%PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 || exit /b 1
)

echo == inference dependencies ==
"%PY%" -m pip install diffusers accelerate transformers omegaconf librosa ^
  einops soundfile imageio imageio-ffmpeg ffmpeg-python opencv-python ^
  "huggingface_hub[hf_xet]" || exit /b 1

echo == weights ==
"%PY%" scripts\setup_musetalk.py || exit /b 1

echo.
echo MuseTalk is ready. Start the app and use the "Lip sync..." button.
endlocal
