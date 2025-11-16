@echo off
REM PaperBench 评测 - Windows 快速启动脚本
REM 运行: run_paperbench.bat [debug|mini|full] [code-dev|complete]

setlocal enabledelayedexpansion

echo ======================================
echo MS-Agent PaperBench 评测快速启动 (Windows)
echo ======================================

REM 1. 检查环境
echo.
echo [1/5] 检查环境...

where python >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未安装 Python
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✓ Python 版本: %PYTHON_VERSION%

REM 2. 检查 PaperBench 数据
echo.
echo [2/5] 准备 PaperBench 数据...

if "%PAPERBENCH_DATA_DIR%"=="" (
    echo ❌ 错误: 环境变量 PAPERBENCH_DATA_DIR 未设置
    echo.
    echo 请按以下步骤设置:
    echo 1. git clone https://github.com/openai/frontier-evals.git --filter=blob:none
    echo 2. cd frontier-evals
    echo 3. git lfs fetch --include "project/paperbench/data/**"
    echo 4. git lfs checkout project/paperbench/data
    echo 5. setx PAPERBENCH_DATA_DIR "完整路径/frontier-evals/project/paperbench/data"
    echo 6. 重启 PowerShell/CMD
    exit /b 1
)

if not exist "%PAPERBENCH_DATA_DIR%\papers" (
    echo ❌ 错误: 找不到 %PAPERBENCH_DATA_DIR%\papers
    exit /b 1
)

setlocal enabledelayedexpansion
set count=0
for /d %%D in ("%PAPERBENCH_DATA_DIR%\papers\*") do (
    set /a count+=1
)
echo ✓ 已找到 !count! 篇论文

REM 3. 检查 API Key
echo.
echo [3/5] 检查 API 密钥...

if "%OPENAI_API_KEY%"=="" (
    echo ❌ 错误: 环境变量 OPENAI_API_KEY 未设置
    echo.
    echo PowerShell 设置:
    echo $Env:OPENAI_API_KEY = "sk-xxx..."
    echo.
    echo CMD 设置:
    echo set OPENAI_API_KEY=sk-xxx...
    exit /b 1
)

echo ✓ OpenAI API Key 已设置

REM 4. 检查 MS-Agent 项目
echo.
echo [4/5] 检查 MS-Agent 项目...

if not exist "projects\code_scratch" (
    echo ❌ 错误: 找不到 projects\code_scratch
    echo 请确保在 MS-Agent 项目根目录运行本脚本
    exit /b 1
)

if not exist "projects\code_scratch\evaluate_paperbench.py" (
    echo ❌ 错误: 找不到 evaluate_paperbench.py
    exit /b 1
)

echo ✓ MS-Agent Code Scratch 项目已就绪

REM 5. 运行评测
echo.
echo [5/5] 运行评测...

REM 获取参数，默认 debug 和 code-dev
set EVAL_MODE=%1
if "%EVAL_MODE%"=="" set EVAL_MODE=debug

set EVAL_TYPE=%2
if "%EVAL_TYPE%"=="" set EVAL_TYPE=code-dev

echo.
echo 评测配置:
echo   - 论文分割: %EVAL_MODE% (debug=3篇, mini=10篇, full=20篇)
echo   - 评测类型: %EVAL_TYPE% (code-dev=仅代码, complete=包含执行)
echo.

python projects\code_scratch\evaluate_paperbench.py ^
    --split %EVAL_MODE% ^
    --type %EVAL_TYPE% ^
    --paperbench-dir "%PAPERBENCH_DATA_DIR%"

echo.
echo ======================================
echo ✓ 评测完成！
echo ======================================
echo.
echo 结果位置:
for /f "tokens=*" %%d in ('powershell -Command "Get-ChildItem paperbench_results -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do (
    echo %%d
)
echo.
echo 查看详细结果:
echo   powershell -Command "Get-Content paperbench_results\*\results_final.json | ConvertFrom-Json | ConvertTo-Json"
echo.

pause
