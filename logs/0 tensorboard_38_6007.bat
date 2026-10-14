@echo off
chcp 65001 > nul

:: ==========================================
:: 配置区域
set ENV_NAME=38
set PORT=6007
:: ==========================================

echo --- 1. 正在切换到当前脚本所在的目录 ---
cd /d "%~dp0"
echo 当前目录: %cd%
echo.

echo --- 2. 正在激活 Conda 环境 '%ENV_NAME%' ---
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo.
    echo 错误：激活 Conda 环境 '%ENV_NAME%' 失败！
    echo 请确保已安装 Conda，并且环境 '%ENV_NAME%' 存在。
    echo 您可以通过在命令行运行 `conda info --envs` 来查看所有环境。
    goto end
)
echo 环境激活成功。
echo.

echo --- 3. 正在启动 TensorBoard (端口: %PORT%) ---
echo 日志目录: %cd%
echo 访问网址: http://localhost:%PORT%/
echo.
python -m tensorboard.main --logdir=. --port=%PORT%
echo.

:end
echo --- 4. 操作已完成，按任意键退出 ---
pause > nul