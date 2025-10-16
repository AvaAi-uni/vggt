#!/bin/bash
# ============================================================================
# 彻底修复 ETH3D 数据集目录结构
# ============================================================================

set -e
cd /workspace/vggt

echo "============================================================================"
echo "修复 ETH3D 数据集结构"
echo "============================================================================"
echo ""

echo "[1] 当前结构："
ls -la data/eth3d/
echo ""

echo "[2] 检查是否有 training 子目录："
if [ -d "data/eth3d/training" ]; then
    echo "发现 data/eth3d/training/ 目录"
    echo "内容："
    ls -la data/eth3d/training/ | head -20
    echo ""

    # 检查 training 下是否有场景
    num_scenes=$(ls -d data/eth3d/training/*/ 2>/dev/null | wc -l)
    if [ $num_scenes -gt 0 ]; then
        echo "training 目录下有 $num_scenes 个子目录（场景）"
        echo "将场景移动到 data/eth3d/ 根目录..."

        # 移动所有场景到根目录
        mv data/eth3d/training/* data/eth3d/ 2>/dev/null || true

        # 删除空的 training 目录
        rmdir data/eth3d/training 2>/dev/null || rm -rf data/eth3d/training

        echo "✓ 已移动"
    else
        echo "training 目录为空或只有文件，删除它"
        rm -rf data/eth3d/training
    fi
else
    echo "✓ 没有 training 子目录"
fi

echo ""
echo "[3] 最终结构："
ls -la data/eth3d/
echo ""

echo "[4] 验证每个场景："
for scene_dir in data/eth3d/*/; do
    scene_name=$(basename "$scene_dir")

    # 跳过非场景目录
    if [ "$scene_name" = "training" ]; then
        echo "跳过 training 目录"
        continue
    fi

    # 检查图像目录
    if [ -d "$scene_dir/dslr_undistorted_images" ]; then
        num=$(ls "$scene_dir/dslr_undistorted_images"/*.JPG 2>/dev/null | wc -l)
        echo "✓ $scene_name: $num 张图像 (dslr_undistorted_images)"
    elif [ -d "$scene_dir/images" ]; then
        num=$(ls "$scene_dir/images"/*.JPG 2>/dev/null | wc -l)
        echo "✓ $scene_name: $num 张图像 (images)"
    else
        echo "✗ $scene_name: 没有图像目录"
    fi
done

echo ""
total=$(find data/eth3d -name "*.JPG" | wc -l)
echo "总计: $total 张 JPG 文件"

echo ""
echo "============================================================================"
echo "✓ 结构修复完成！"
echo "============================================================================"
echo ""
echo "现在场景应该在: data/eth3d/courtyard/, data/eth3d/delivery_area/, ..."
echo ""
