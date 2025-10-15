@echo off
REM ============================================================================
REM RunPod 环境配置脚本 - Windows CMD版本
REM ============================================================================
REM
REM 这个脚本自动配置整个训练环境，包括：
REM 1. 检查Python和CUDA环境
REM 2. 安装所有必要的依赖包
REM 3. 下载和解压ETH3D数据集
REM 4. 验证环境配置
REM
REM 使用方法：
REM   setup_runpod_environment.bat
REM
REM ============================================================================

echo ============================================================================
echo           RunPod 环境配置 - VGGT 多精度量化训练
echo ============================================================================
echo.

REM 设置颜色
color 0A

REM 检查管理员权限
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] 以管理员权限运行
) else (
    echo [WARNING] 建议以管理员权限运行此脚本
)

echo.
echo ============================================================================
echo 步骤 1/5: 检查Python环境
echo ============================================================================
echo.

REM 检查Python版本
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python未安装或不在PATH中
    echo 请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

python --version
echo [OK] Python已安装

REM 检查pip
pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] pip未安装
    pause
    exit /b 1
)
echo [OK] pip已安装

echo.
echo ============================================================================
echo 步骤 2/5: 检查CUDA环境
echo ============================================================================
echo.

REM 检查CUDA
nvcc --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] CUDA未检测到，将安装CPU版本的PyTorch
    set CUDA_AVAILABLE=0
) else (
    nvcc --version
    echo [OK] CUDA已安装
    set CUDA_AVAILABLE=1
)

REM 检查nvidia-smi
nvidia-smi >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] nvidia-smi不可用，GPU可能不可用
) else (
    echo [OK] NVIDIA驱动已安装
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)

echo.
echo ============================================================================
echo 步骤 3/5: 安装Python依赖包
echo ============================================================================
echo.

echo [INFO] 升级pip...
python -m pip install --upgrade pip

echo.
echo [INFO] 安装PyTorch和相关包...
if %CUDA_AVAILABLE%==1 (
    echo [INFO] 安装CUDA版本的PyTorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [INFO] 安装CPU版本的PyTorch
    pip install torch torchvision torchaudio
)

echo.
echo [INFO] 安装核心依赖...
pip install numpy opencv-python pillow
pip install hydra-core omegaconf
pip install tensorboard
pip install fvcore
pip install iopath
pip install einops
pip install timm

echo.
echo [INFO] 安装科学计算和可视化包...
pip install scipy matplotlib seaborn
pip install tqdm rich

echo.
echo [INFO] 安装开发工具...
pip install pytest black flake8

echo.
echo [OK] 所有依赖包安装完成

echo.
echo ============================================================================
echo 步骤 4/5: 下载和配置ETH3D数据集
echo ============================================================================
echo.

REM 检查data目录
if not exist "vggt\data" (
    echo [INFO] 创建data目录...
    mkdir vggt\data
)

REM 检查7z
where 7z >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] 7z未安装
    echo.
    echo 请下载并安装7-Zip：https://www.7-zip.org/
    echo 安装后确保7z.exe在PATH中
    echo.
    echo 或者手动下载ETH3D数据集：
    echo URL: https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z
    echo 解压到: vggt\data\eth3d\
    echo.
    pause
    exit /b 1
)

echo [OK] 7z已安装

REM 运行ETH3D下载脚本
if exist "vggt\scripts\download_eth3d.py" (
    echo [INFO] 运行ETH3D数据集下载脚本...
    cd vggt
    python scripts\download_eth3d.py --output_dir data\eth3d
    cd ..

    if %errorLevel% neq 0 (
        echo [ERROR] ETH3D下载失败
        pause
        exit /b 1
    )
    echo [OK] ETH3D数据集配置完成
) else (
    echo [WARNING] download_eth3d.py未找到
    echo 请手动下载ETH3D数据集到 vggt\data\eth3d\
)

echo.
echo ============================================================================
echo 步骤 5/5: 验证环境配置
echo ============================================================================
echo.

echo [INFO] 验证PyTorch安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo [INFO] 验证其他依赖...
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"
python -c "import hydra; print(f'Hydra版本: {hydra.__version__}')"

echo.
echo [INFO] 检查数据集...
if exist "vggt\data\eth3d" (
    dir /b vggt\data\eth3d
    echo [OK] ETH3D数据集已配置
) else (
    echo [WARNING] ETH3D数据集目录不存在
)

echo.
echo ============================================================================
echo                         环境配置完成！
echo ============================================================================
echo.
echo 接下来你可以：
echo.
echo 1. 查看训练配置：
echo    notepad vggt\training\config\eth3d_quantization.yaml
echo.
echo 2. 开始训练（单GPU）：
echo    cd vggt\training
echo    torchrun --nproc_per_node=1 launch.py --config eth3d_quantization
echo.
echo 3. 查看完整使用文档：
echo    type RUNPOD_USAGE.md
echo.
echo ============================================================================

pause
