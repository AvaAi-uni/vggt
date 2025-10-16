#!/bin/bash
# ============================================================================
# VGGT 训练启动脚本 - 自动设置 PYTHONPATH
# ============================================================================
#
# 使用方法：
#   bash train.sh eth3d_fp32_quick_test
#   bash train.sh eth3d_fp32_baseline
#
# ============================================================================

# 获取项目根目录（脚本所在目录）
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 获取配置文件名（第一个参数，默认 eth3d_fp32_quick_test）
CONFIG_NAME=${1:-eth3d_fp32_quick_test}

echo "============================================================================"
echo "VGGT 训练"
echo "============================================================================"
echo "项目路径: ${PROJECT_ROOT}"
echo "配置文件: ${CONFIG_NAME}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "============================================================================"
echo ""

# 检测可用的 GPU 数量
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=1
fi

echo "检测到 ${NUM_GPUS} 个 GPU"
echo ""

# 切换到 training 目录并使用 torchrun 启动分布式训练
cd "${PROJECT_ROOT}/training"
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    launch.py --config ${CONFIG_NAME}
