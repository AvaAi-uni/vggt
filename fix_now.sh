#!/bin/bash
# ============================================================================
# 快速修复当前遇到的两个问题
# 1. 缺少 hydra 模块
# 2. 数据集目录结构问题
# ============================================================================

cd /workspace/vggt

echo "============================================================================"
echo "快速修复"
echo "============================================================================"
echo ""

# 修复 1: 安装缺失的模块
echo "[1/3] 安装缺失的 Python 模块..."
pip install -r requirements.txt -q
pip install -e . -q
echo "✓ 模块已安装"
echo ""

# 修复 2: 检查和修复数据集目录
echo "[2/3] 检查数据集目录..."

# 查看当前结构
echo "当前 data/eth3d/ 结构："
ls -la data/eth3d/ 2>/dev/null || echo "data/eth3d 不存在"
echo ""

# 查找所有 JPG 文件
num_images=$(find data/eth3d -name "*.JPG" 2>/dev/null | wc -l)
echo "总共找到 $num_images 张图像"

if [ $num_images -eq 0 ]; then
    echo "✗ 没有找到图像，请重新下载数据集"
    exit 1
fi

# 检查 training 目录
if [ ! -d "data/eth3d/training" ]; then
    echo ""
    echo "training 目录不存在，尝试修复..."

    # 方案 1: 检查是否有 multi_view_training_dslr_undistorted
    if [ -d "data/eth3d/multi_view_training_dslr_undistorted" ]; then
        echo "发现 multi_view_training_dslr_undistorted，重命名为 training..."
        mv data/eth3d/multi_view_training_dslr_undistorted data/eth3d/training
        echo "✓ 已重命名"
    else
        # 方案 2: 查找场景目录
        echo "查找场景目录..."
        scene_dir=$(find data/eth3d -type d -name "images" | head -1 | xargs dirname 2>/dev/null)
        if [ ! -z "$scene_dir" ]; then
            parent_dir=$(dirname "$scene_dir")
            echo "发现场景在：$parent_dir"
            mkdir -p data/eth3d/training
            echo "移动场景到 training 目录..."
            mv "$parent_dir"/* data/eth3d/training/ 2>/dev/null || true
            echo "✓ 已移动"
        else
            echo "✗ 无法找到场景目录"
            echo "请手动检查 data/eth3d/ 的内容"
            exit 1
        fi
    fi
else
    echo "✓ training 目录已存在"
fi

# 最终验证
echo ""
num_images=$(find data/eth3d/training -name "*.JPG" 2>/dev/null | wc -l)
echo "training 目录下有 $num_images 张图像"

if [ $num_images -eq 0 ]; then
    echo "✗ training 目录下没有图像"
    echo "请检查数据集是否正确下载"
    exit 1
fi

echo "✓ 数据集修复完成"
echo ""

# 修复 3: 显示场景
echo "[3/3] 验证场景..."
ls -la data/eth3d/training/ | head -15

echo ""
echo "============================================================================"
echo "✓ 修复完成！"
echo "============================================================================"
echo ""
echo "现在可以运行："
echo "  bash train.sh eth3d_fp32_quick_test"
echo ""
