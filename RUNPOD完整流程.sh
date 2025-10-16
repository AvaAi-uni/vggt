#!/bin/bash
# ============================================================================
# VGGT RunPod 完整流程 - 从零开始到训练运行
# ============================================================================
#
# 使用方法：
#   在 RunPod 终端中运行：
#   bash <(curl -s https://raw.githubusercontent.com/YOUR_REPO/RUNPOD完整流程.sh)
#
#   或者手动复制本脚本内容到 RunPod 终端执行
#
# ============================================================================

set -e  # 遇到错误立即退出

echo "============================================================================"
echo "VGGT RunPod 完整安装流程"
echo "============================================================================"
echo ""
echo "本脚本将完成："
echo "  1. Clone 代码仓库（如果需要）"
echo "  2. 安装所有依赖"
echo "  3. 下载 ETH3D 数据集"
echo "  4. 配置环境"
echo "  5. 启动快速测试训练"
echo ""
read -p "按 Enter 继续，或 Ctrl+C 取消..."
echo ""

# ============================================================================
# 步骤 1: 检查或 Clone 代码
# ============================================================================
echo "[1/8] 检查代码仓库..."

if [ ! -d "/workspace/vggt" ]; then
    echo "代码不存在，正在 clone..."
    cd /workspace

    # 如果你有 git 仓库，取消下面的注释并替换 URL
    # git clone https://github.com/YOUR_USERNAME/vggt.git

    # 否则，假设代码已经通过其他方式上传到 /workspace/vggt
    echo "请确保代码已上传到 /workspace/vggt/"
    exit 1
else
    echo "✓ 代码仓库已存在"
fi

cd /workspace/vggt
echo "当前目录: $(pwd)"
echo ""

# ============================================================================
# 步骤 2: 系统依赖
# ============================================================================
echo "[2/8] 安装系统依赖..."

apt-get update -qq
apt-get install -y p7zip-full wget -qq

echo "✓ 系统依赖已安装"
echo ""

# ============================================================================
# 步骤 3: Python 环境
# ============================================================================
echo "[3/8] 检查 Python 环境..."

python --version
pip --version

echo "✓ Python 环境正常"
echo ""

# ============================================================================
# 步骤 4: 修复 NumPy 和安装依赖
# ============================================================================
echo "[4/8] 安装 Python 依赖..."

# 卸载旧的 NumPy
pip uninstall -y numpy 2>/dev/null || true

# 安装正确版本的 NumPy
pip install "numpy>=1.21.0,<2.0.0" -q

# 安装其他依赖
pip install wcmatch>=8.4.0 -q

# 安装 vggt 包（editable mode）
pip install -e . -q

echo "✓ Python 依赖已安装"
echo ""

# ============================================================================
# 步骤 5: 验证环境
# ============================================================================
echo "[5/8] 验证环境..."

python -c "
import sys
import numpy as np
import torch
import vggt

print(f'✓ NumPy: {np.__version__}')
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ vggt 模块: 已安装')
"

if [ $? -ne 0 ]; then
    echo "✗ 环境验证失败"
    exit 1
fi

echo ""

# ============================================================================
# 步骤 6: 下载 ETH3D 数据集
# ============================================================================
echo "[6/8] 下载 ETH3D 数据集..."

if [ -d "/workspace/vggt/data/eth3d/training" ] && [ "$(ls -A /workspace/vggt/data/eth3d/training 2>/dev/null | wc -l)" -gt 5 ]; then
    echo "✓ ETH3D 数据集已存在，跳过下载"
else
    echo "开始下载 ETH3D 数据集 (~1.5 GB)..."

    mkdir -p /workspace/vggt/data/eth3d
    cd /workspace/vggt/data/eth3d

    # 下载数据集
    wget --progress=bar:force \
        --show-progress \
        --continue \
        -O multi_view_training_dslr_undistorted.7z \
        https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z

    if [ $? -ne 0 ]; then
        echo "✗ 下载失败"
        exit 1
    fi

    # 解压
    echo "解压数据集..."
    7z x multi_view_training_dslr_undistorted.7z -y > /dev/null

    # 删除压缩包
    rm multi_view_training_dslr_undistorted.7z

    echo "✓ 数据集下载完成"
fi

cd /workspace/vggt
echo ""

# ============================================================================
# 步骤 7: 验证数据集
# ============================================================================
echo "[7/8] 验证数据集..."

# 检查数据集结构
if [ ! -d "/workspace/vggt/data/eth3d/training" ]; then
    echo "✗ 数据集结构错误：未找到 training 目录"
    exit 1
fi

# 统计场景数量
num_scenes=$(ls -1 /workspace/vggt/data/eth3d/training | wc -l)
echo "找到 $num_scenes 个场景"

# 统计图像数量
total_images=0
for scene in /workspace/vggt/data/eth3d/training/*; do
    if [ -d "$scene/images" ]; then
        num_images=$(ls -1 "$scene/images"/*.JPG 2>/dev/null | wc -l)
        scene_name=$(basename "$scene")
        echo "  - $scene_name: $num_images 张图像"
        total_images=$((total_images + num_images))
    fi
done

echo "总计: $total_images 张图像"

if [ $total_images -eq 0 ]; then
    echo "✗ 错误：未找到任何图像"
    exit 1
fi

echo "✓ 数据集验证完成"
echo ""

# ============================================================================
# 步骤 8: 修复配置文件路径
# ============================================================================
echo "[8/8] 修复配置文件路径..."

# 修复所有配置文件中的数据集路径
for config_file in training/config/eth3d*.yaml; do
    if [ -f "$config_file" ]; then
        # 将 ETH3D_DIR 改为 /workspace/vggt/data/eth3d/training
        sed -i 's|ETH3D_DIR:.*|ETH3D_DIR: /workspace/vggt/data/eth3d/training|g' "$config_file"
        echo "✓ 已修复: $config_file"
    fi
done

echo ""

# ============================================================================
# 完成安装
# ============================================================================
echo "============================================================================"
echo "✓ 安装完成！"
echo "============================================================================"
echo ""
echo "环境信息："
echo "  - 项目路径: /workspace/vggt"
echo "  - 数据集路径: /workspace/vggt/data/eth3d/training"
echo "  - 场景数量: $num_scenes"
echo "  - 图像总数: $total_images"
echo ""
echo "============================================================================"
echo "🚀 现在启动快速测试训练（5-10 分钟）"
echo "============================================================================"
echo ""

# 启动快速测试
cd /workspace/vggt
bash train.sh eth3d_fp32_quick_test
