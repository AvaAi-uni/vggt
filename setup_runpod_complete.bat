@echo off
REM ============================================================================
REM RunPod 环境完整配置脚本 (Windows CMD)
REM
REM 本脚本将完成以下任务：
REM 1. 安装所有依赖包
REM 2. 下载 ETH3D 数据集
REM 3. 验证环境配置
REM
REM 使用方法：
REM   setup_runpod_complete.bat
REM ============================================================================

echo ============================================================================
echo VGGT 量化训练 - RunPod 环境完整配置
echo ============================================================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 Python！请先安装 Python 3.8+
    pause
    exit /b 1
)

echo [1/5] 检测 Python 环境...
python --version
echo.

REM 升级pip
echo [2/5] 升级 pip...
python -m pip install --upgrade pip
echo.

REM 安装依赖包
echo [3/5] 安装依赖包...
echo 这可能需要几分钟...
pip install -r requirements.txt

REM 检查是否需要安装PyTorch
python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] 未检测到 PyTorch，开始安装...
    echo 请选择您的CUDA版本：
    echo   1. CUDA 11.8
    echo   2. CUDA 12.1
    echo   3. CPU Only
    choice /c 123 /n /m "请输入选项 (1-3): "

    if errorlevel 3 (
        pip install torch torchvision torchaudio
    ) else if errorlevel 2 (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
)

REM 安装7z工具（用于解压数据集）
echo.
echo [4/5] 检查 7z 工具...
where 7z >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] 未找到 7z 工具
    echo 请下载并安装 7-Zip: https://www.7-zip.org/
    echo 或者安装 Python 库: pip install py7zr
    pip install py7zr
)
echo.

REM 下载ETH3D数据集
echo [5/5] 下载 ETH3D 数据集...
echo 数据集大小约 3-5 GB，下载可能需要较长时间...
echo.
set /p download_data="是否下载 ETH3D 数据集? (y/n): "
if /i "%download_data%"=="y" (
    python scripts/download_eth3d.py --output_dir data/eth3d
) else (
    echo 跳过数据集下载。您可以稍后手动运行：
    echo   python scripts/download_eth3d.py --output_dir data/eth3d
)

echo.
echo ============================================================================
echo 环境配置完成！
echo ============================================================================
echo.
echo 下一步：
echo   1. 查看可用的训练配置：
echo      dir training\config\eth3d_*.yaml
echo.
echo   2. 开始训练（FP32 baseline）：
echo      cd training
echo      python launch.py --config eth3d_fp32_baseline
echo.
echo   3. 或使用完整的实验脚本：
echo      python scripts/run_all_experiments.py
echo.
echo ============================================================================

pause
