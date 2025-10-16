#!/bin/bash
# 诊断 ETH3D 数据集结构

echo "============================================================================"
echo "ETH3D 数据集诊断"
echo "============================================================================"
echo ""

cd /workspace/vggt

echo "[1] 检查 data/eth3d/ 目录："
ls -la data/eth3d/
echo ""

echo "[2] 检查第一个场景的结构："
first_scene=$(ls data/eth3d/ | head -1)
echo "场景: $first_scene"
ls -la "data/eth3d/$first_scene/"
echo ""

echo "[3] 检查是否有 images 目录："
for scene in data/eth3d/*/; do
    scene_name=$(basename "$scene")
    if [ -d "$scene/images" ]; then
        num=$(ls "$scene/images"/*.JPG 2>/dev/null | wc -l)
        echo "✓ $scene_name/images/ - $num 张 JPG"
    elif [ -d "$scene/dslr_undistorted_images" ]; then
        num=$(ls "$scene/dslr_undistorted_images"/*.JPG 2>/dev/null | wc -l)
        echo "✓ $scene_name/dslr_undistorted_images/ - $num 张 JPG"
    else
        echo "✗ $scene_name - 没有 images 或 dslr_undistorted_images 目录"
        echo "  实际子目录："
        ls "$scene/"
    fi
done
echo ""

echo "[4] 查找所有 JPG 文件："
find data/eth3d -name "*.JPG" | head -20
echo "..."
total=$(find data/eth3d -name "*.JPG" | wc -l)
echo "总计: $total 张 JPG 文件"
echo ""

echo "[5] 检查第一个场景的完整结构："
tree -L 3 "data/eth3d/$first_scene/" 2>/dev/null || find "data/eth3d/$first_scene/" -type d | head -20
