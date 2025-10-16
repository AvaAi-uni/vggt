#!/bin/bash
# ============================================================================
# VGGT RunPod 完整环境配置脚本
# ============================================================================
#
# 使用方法：
#   chmod +x runpod_setup.sh
#   ./runpod_setup.sh
#
# 或直接运行：
#   bash runpod_setup.sh
#
# ============================================================================

set -e  # 遇到错误立即退出

echo "============================================================================"
echo "VGGT RunPod 环境配置"
echo "============================================================================"
echo ""

# 检查 Python 版本
echo "[1/7] 检查 Python 环境..."
python --version
echo ""

# 升级 pip
echo "[2/7] 升级 pip..."
pip install --upgrade pip
echo ""

# 卸载不兼容的 NumPy（如果存在）
echo "[3/7] 修复 NumPy 版本..."
pip uninstall -y numpy || true
pip install "numpy>=1.21.0,<2.0.0"
echo ""

# 安装 PyTorch（检测 CUDA 版本）
echo "[4/7] 安装 PyTorch..."
if python -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch 已安装"
    python -c "import torch; print(f'PyTorch 版本: {torch.__version__}')"
    python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
else
    echo "安装 PyTorch (CUDA 11.8)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi
echo ""

# 安装其他依赖
echo "[5/7] 安装项目依赖..."
pip install -r requirements.txt
echo ""

# 安装 7z 工具
echo "[6/7] 安装 7z 解压工具..."
if command -v 7z &> /dev/null; then
    echo "✓ 7z 已安装"
else
    echo "安装 p7zip-full..."
    apt-get update -qq
    apt-get install -y p7zip-full
fi
echo ""

# 下载 ETH3D 数据集
echo "[7/7] 下载 ETH3D 数据集..."
if [ -d "data/eth3d" ] && [ "$(ls -A data/eth3d)" ]; then
    echo "✓ ETH3D 数据集已存在"
else
    echo "开始下载 ETH3D 数据集（约 3-5 GB）..."
    python scripts/download_eth3d.py --output_dir data/eth3d
fi
echo ""

# 验证安装
echo "============================================================================"
echo "验证安装..."
echo "============================================================================"

# 检查关键模块
python -c "
import sys
print('Python 版本:', sys.version)

try:
    import numpy as np
    print(f'✓ NumPy: {np.__version__}')
    assert np.__version__.startswith('1.'), 'NumPy 必须是 1.x 版本'
except Exception as e:
    print(f'✗ NumPy: {e}')
    sys.exit(1)

try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'✓ CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ CUDA 版本: {torch.version.cuda}')
        print(f'✓ GPU 数量: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
except Exception as e:
    print(f'✗ PyTorch: {e}')
    sys.exit(1)

try:
    import cv2
    print(f'✓ OpenCV: {cv2.__version__}')
except Exception as e:
    print(f'✗ OpenCV: {e}')
    sys.exit(1)

try:
    import hydra
    print(f'✓ Hydra: {hydra.__version__}')
except Exception as e:
    print(f'✗ Hydra: {e}')
    sys.exit(1)

try:
    from wcmatch import fnmatch
    print(f'✓ wcmatch: 已安装')
except Exception as e:
    print(f'✗ wcmatch: {e}')
    sys.exit(1)

try:
    import fvcore
    print(f'✓ fvcore: 已安装')
except Exception as e:
    print(f'✗ fvcore: {e}')
    sys.exit(1)

print()
print('✓ 所有依赖检查通过！')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✓ 环境配置完成！"
    echo "============================================================================"
    echo ""
    echo "下一步："
    echo "  1. 快速测试（5-10分钟）："
    echo "     cd training"
    echo "     python launch.py --config eth3d_fp32_quick_test"
    echo ""
    echo "  2. 正式训练："
    echo "     python launch.py --config eth3d_fp32_baseline"
    echo ""
    echo "  3. 批量运行所有实验："
    echo "     cd .."
    echo "     python scripts/run_all_experiments.py"
    echo ""
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "✗ 环境配置失败！请检查错误信息"
    echo "============================================================================"
    exit 1
fi
