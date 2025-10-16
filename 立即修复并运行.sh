#!/bin/bash
# ============================================================================
# 立即修复路径问题并启动训练
# ============================================================================

echo "============================================================================"
echo "修复数据集路径问题"
echo "============================================================================"
echo ""

cd /workspace/vggt

# 修复所有配置文件中的路径
echo "[1/2] 修复配置文件路径..."

for config_file in training/config/eth3d*.yaml; do
    if [ -f "$config_file" ]; then
        sed -i 's|ETH3D_DIR:.*|ETH3D_DIR: /workspace/vggt/data/eth3d/training|g' "$config_file"
        echo "✓ 已修复: $config_file"
    fi
done

echo ""
echo "[2/2] 验证数据集..."

# 检查数据集是否存在
if [ ! -d "/workspace/vggt/data/eth3d/training" ]; then
    echo "✗ 数据集不存在，请先下载："
    echo "  bash RUNPOD完整流程.sh"
    exit 1
fi

# 统计场景和图像
num_scenes=$(ls -1 /workspace/vggt/data/eth3d/training 2>/dev/null | wc -l)
total_images=0

for scene in /workspace/vggt/data/eth3d/training/*; do
    if [ -d "$scene/images" ]; then
        num_images=$(ls -1 "$scene/images"/*.JPG 2>/dev/null | wc -l)
        total_images=$((total_images + num_images))
    fi
done

echo "找到 $num_scenes 个场景，$total_images 张图像"

if [ $total_images -eq 0 ]; then
    echo "✗ 错误：未找到任何图像"
    echo "数据集可能未正确下载，请运行："
    echo "  bash RUNPOD完整流程.sh"
    exit 1
fi

echo ""
echo "============================================================================"
echo "✓ 修复完成！"
echo "============================================================================"
echo ""
echo "🚀 启动快速测试训练..."
echo ""

# 启动训练
bash train.sh eth3d_fp32_quick_test
