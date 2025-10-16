#!/bin/bash
# ============================================================================
# VGGT RunPod 一键启动脚本
# 功能：安装依赖 → 下载数据集 → 启动训练
# ============================================================================

set -e
cd /workspace/vggt

echo "============================================================================"
echo "VGGT RunPod 一键启动"
echo "============================================================================"
echo ""

# 1. 安装依赖
echo "[1/4] 安装依赖..."
apt-get update -qq && apt-get install -y p7zip-full wget -qq
pip uninstall -y numpy 2>/dev/null || true
pip install "numpy>=1.21.0,<2.0.0" wcmatch>=8.4.0 -q
pip install -e . -q
echo "✓ 依赖已安装"
echo ""

# 2. 下载数据集
echo "[2/4] 下载 ETH3D 数据集..."
if [ -d "data/eth3d/training" ] && [ "$(ls -A data/eth3d/training 2>/dev/null | wc -l)" -gt 5 ]; then
    echo "✓ 数据集已存在"
else
    mkdir -p data/eth3d
    cd data/eth3d

    echo "下载中 (~1.5 GB)..."
    wget -q --show-progress --continue \
        https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z

    echo "解压中..."
    7z x multi_view_training_dslr_undistorted.7z -y > /dev/null
    rm multi_view_training_dslr_undistorted.7z

    cd /workspace/vggt
    echo "✓ 数据集已下载"
fi
echo ""

# 3. 验证数据集
echo "[3/4] 验证数据集..."
num_images=$(find data/eth3d/training -name "*.JPG" 2>/dev/null | wc -l)
echo "找到 $num_images 张图像"
if [ $num_images -eq 0 ]; then
    echo "✗ 数据集为空"
    exit 1
fi
echo ""

# 4. 启动训练
echo "[4/4] 启动快速测试..."
echo ""
bash train.sh eth3d_fp32_quick_test
