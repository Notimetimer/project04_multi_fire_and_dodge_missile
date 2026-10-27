@echo off

:: ==========================================
:: Configuration
set ENV_NAME=38
set PORT=6007
:: ==========================================

echo --- 1. Switching to current script directory ---
cd /d "%~dp0"
echo Current directory: %cd%
echo.

echo --- 2. Activating Conda environment '%ENV_NAME%' ---
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to activate Conda environment '%ENV_NAME%'!
    echo Please ensure Conda is installed and environment '%ENV_NAME%' exists.
    echo You can run `conda info --envs` to list all available environments.
    goto end
)
echo Environment activated successfully.
echo.

echo --- 3. Starting TensorBoard (Port: %PORT%) ---
echo Log directory: %cd%
echo Access URL: http://localhost:%PORT%/
echo.
python -m tensorboard.main --logdir=. --port=%PORT%
echo.

:end
echo --- 4. Done. Press any key to exit ---
pause > nul