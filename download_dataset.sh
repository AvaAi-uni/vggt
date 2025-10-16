#!/bin/bash
# ============================================================================
# ETH3D 数据集快速下载脚本（适用于 RunPod）
# ============================================================================

echo "============================================================================"
echo "ETH3D 数据集下载"
echo "============================================================================"
echo ""

# 创建数据目录
mkdir -p data/eth3d
cd data/eth3d

# 检查是否已下载
if [ -d "training" ] && [ "$(ls -A training 2>/dev/null)" ]; then
    echo "✓ 数据集已存在，跳过下载"
    echo ""
    echo "数据集路径: $(pwd)"
    ls -la
    exit 0
fi

echo "[1/4] 安装 p7zip 解压工具..."
apt-get update -qq && apt-get install -y p7zip-full -qq
echo "✓ p7zip 已安装"
echo ""

echo "[2/4] 下载 ETH3D 数据集..."
echo "URL: https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z"
echo "大小: ~1.5 GB"
echo ""

wget --progress=bar:force \
    --show-progress \
    --continue \
    -O multi_view_training_dslr_undistorted.7z \
    https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z

if [ $? -ne 0 ]; then
    echo "✗ 下载失败"
    exit 1
fi

echo "✓ 下载完成"
echo ""

echo "[3/4] 解压数据集..."
7z x multi_view_training_dslr_undistorted.7z -y

if [ $? -ne 0 ]; then
    echo "✗ 解压失败"
    exit 1
fi

echo "✓ 解压完成"
echo ""

echo "[4/4] 清理和验证..."
rm multi_view_training_dslr_undistorted.7z
echo "✓ 压缩包已删除"
echo ""

# 验证数据集
echo "数据集结构:"
ls -la
echo ""

# 统计场景和图像数量
if [ -d "training" ]; then
    num_scenes=$(ls -1 training | wc -l)
    echo "训练场景数量: $num_scenes"

    # 统计图像总数
    total_images=0
    for scene in training/*; do
        if [ -d "$scene/images" ]; then
            num_images=$(ls -1 "$scene/images"/*.JPG 2>/dev/null | wc -l)
            echo "  - $(basename $scene): $num_images 张图像"
            total_images=$((total_images + num_images))
        fi
    done
    echo ""
    echo "总计: $total_images 张图像"
fi

echo ""
echo "============================================================================"
echo "✓ ETH3D 数据集准备完成！"
echo "============================================================================"
echo ""
echo "数据集路径: $(pwd)"
echo ""
echo "下一步:"
echo "  cd /workspace/vggt"
echo "  bash train.sh eth3d_fp32_quick_test"
echo ""
