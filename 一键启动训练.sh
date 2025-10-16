#!/bin/bash
# ============================================================================
# VGGT 一键下载数据集并启动训练
# ============================================================================

set -e  # 遇到错误立即退出

echo "============================================================================"
echo "VGGT 一键启动训练"
echo "============================================================================"
echo ""

# 步骤1: 检查数据集是否存在
if [ -d "/workspace/vggt/data/eth3d/training" ]; then
    echo "✓ ETH3D 数据集已存在"
else
    echo "[1/2] 下载 ETH3D 数据集..."
    echo ""

    # 创建目录
    mkdir -p /workspace/vggt/data/eth3d
    cd /workspace/vggt/data/eth3d

    # 安装 p7zip
    echo "安装解压工具..."
    apt-get update -qq && apt-get install -y p7zip-full -qq

    # 下载数据集
    echo ""
    echo "下载 ETH3D 数据集 (~1.5 GB)..."
    wget --progress=bar:force \
        --show-progress \
        --continue \
        -O multi_view_training_dslr_undistorted.7z \
        https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z

    # 解压
    echo ""
    echo "解压数据集..."
    7z x multi_view_training_dslr_undistorted.7z -y > /dev/null

    # 清理
    rm multi_view_training_dslr_undistorted.7z

    echo "✓ 数据集下载完成"
fi

echo ""
echo "[2/2] 启动训练..."
echo ""

# 返回项目根目录
cd /workspace/vggt

# 启动训练
bash train.sh eth3d_fp32_quick_test
