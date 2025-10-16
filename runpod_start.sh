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
echo "[1/5] 安装系统依赖..."
apt-get update -qq && apt-get install -y p7zip-full wget -qq
echo "✓ 系统依赖已安装"
echo ""

echo "[2/5] 安装 Python 依赖..."
pip uninstall -y numpy 2>/dev/null || true
pip install -r requirements.txt -q
pip install -e . -q
echo "✓ Python 依赖已安装"
echo ""

# 2. 下载数据集
echo "[3/5] 下载 ETH3D 数据集..."
if [ -d "data/eth3d/training" ] && [ "$(find data/eth3d/training -name '*.JPG' 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "✓ 数据集已存在"
else
    mkdir -p data/eth3d
    cd data/eth3d

    echo "下载中 (~1.5 GB)..."
    wget -q --show-progress --continue \
        https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z

    echo "解压中..."
    7z x multi_view_training_dslr_undistorted.7z -y
    rm multi_view_training_dslr_undistorted.7z

    cd /workspace/vggt
    echo "✓ 数据集已下载"
fi
echo ""

# 3. 检查解压结果
echo "[4/5] 验证数据集..."
echo "数据集结构："
ls -la data/eth3d/ | head -20

# 查找所有 JPG 文件
num_images=$(find data/eth3d -name "*.JPG" 2>/dev/null | wc -l)
echo ""
echo "找到 $num_images 张图像"

if [ $num_images -eq 0 ]; then
    echo "✗ 数据集为空，检查目录结构："
    find data/eth3d -type d | head -20
    exit 1
fi

# 检查 training 目录是否存在
if [ ! -d "data/eth3d/training" ]; then
    echo "警告：training 目录不存在，尝试修复..."

    # 检查是否有 multi_view_training_dslr_undistorted 目录
    if [ -d "data/eth3d/multi_view_training_dslr_undistorted" ]; then
        echo "发现 multi_view_training_dslr_undistorted 目录，重命名为 training..."
        mv data/eth3d/multi_view_training_dslr_undistorted data/eth3d/training
    else
        echo "查找场景目录..."
        # 查找第一个包含 images 的目录
        scene_dir=$(find data/eth3d -type d -name "images" | head -1 | xargs dirname)
        if [ ! -z "$scene_dir" ]; then
            parent_dir=$(dirname "$scene_dir")
            echo "发现场景在：$parent_dir"
            if [ "$parent_dir" != "data/eth3d/training" ]; then
                echo "创建 training 目录并移动场景..."
                mkdir -p data/eth3d/training
                mv "$parent_dir"/* data/eth3d/training/
            fi
        fi
    fi
fi

# 再次验证
num_images=$(find data/eth3d/training -name "*.JPG" 2>/dev/null | wc -l)
echo "training 目录下有 $num_images 张图像"

if [ $num_images -eq 0 ]; then
    echo "✗ 数据集结构错误，请手动检查"
    exit 1
fi

echo "✓ 数据集验证完成"
echo ""

# 4. 启动训练
echo "[5/5] 启动快速测试..."
echo ""
bash train.sh eth3d_fp32_quick_test
