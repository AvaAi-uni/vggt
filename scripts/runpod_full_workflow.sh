#!/bin/bash

################################################################################
# RunPod 完整量化实验 - 一键执行脚本
#
# 用途: 在RunPod上一键完成所有操作（从设置到实验完成）
# 使用: bash scripts/runpod_full_workflow.sh [quick|standard|full]
#
# 注意: 运行此脚本前，请确保代码已在 /workspace/vggt
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# 获取测试类型参数
TEST_TYPE=${1:-quick}

print_header "RunPod 完整量化实验 - 一键执行"
echo ""
print_info "测试类型: $TEST_TYPE (quick/standard/full)"
echo ""

################################################################################
# 步骤0: 检查当前位置
################################################################################

print_header "步骤0: 检查项目位置"

if [ ! -f "scripts/runpod_setup_comprehensive.sh" ]; then
    print_error "错误: 当前不在项目根目录！"
    print_info "请确保在 /workspace/vggt 目录下运行此脚本"
    print_info "运行命令: cd /workspace/vggt && bash scripts/runpod_full_workflow.sh"
    exit 1
fi

print_success "项目位置正确: $(pwd)"
echo ""

################################################################################
# 步骤1: 检查CUDA环境
################################################################################

print_header "步骤1: 检查CUDA环境"

if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi 未找到！请确保在GPU Pod上运行"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print_success "CUDA环境检查通过"
echo ""

################################################################################
# 步骤2: 安装依赖
################################################################################

print_header "步骤2: 安装Python依赖"

print_info "检查torch安装..."
if ! python -c "import torch" 2>/dev/null; then
    print_info "安装torch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
else
    print_success "torch已安装"
fi

print_info "安装其他依赖..."
pip install numpy matplotlib tqdm requests py7zr --quiet

print_success "依赖安装完成"
echo ""

# 验证CUDA
print_info "验证CUDA..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

################################################################################
# 步骤3: 下载ETH3D数据（如果不存在）
################################################################################

print_header "步骤3: 准备ETH3D数据"

DATA_DIR="/workspace/data/eth3d"
COURTYARD_IMAGES="$DATA_DIR/courtyard/dslr_images_undistorted"

if [ -d "$COURTYARD_IMAGES" ] && [ "$(ls -A $COURTYARD_IMAGES/*.JPG 2>/dev/null | wc -l)" -gt 0 ]; then
    IMAGE_COUNT=$(ls -1 $COURTYARD_IMAGES/*.JPG 2>/dev/null | wc -l)
    print_success "ETH3D数据已存在 (找到 $IMAGE_COUNT 张图像)"
else
    print_info "ETH3D数据不存在，开始下载..."
    print_info "预计下载时间: 10-20分钟（~15GB）"

    # 自动下载
    python scripts/download_eth3d.py --output_dir "$DATA_DIR" --skip_existing

    if [ -d "$COURTYARD_IMAGES" ]; then
        IMAGE_COUNT=$(ls -1 $COURTYARD_IMAGES/*.JPG 2>/dev/null | wc -l)
        print_success "数据下载完成 (找到 $IMAGE_COUNT 张图像)"
    else
        print_error "数据下载失败！"
        print_info "请手动下载: https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z"
        exit 1
    fi
fi

echo ""

################################################################################
# 步骤4: 创建输出目录
################################################################################

print_header "步骤4: 创建输出目录"

mkdir -p /workspace/results
mkdir -p /workspace/logs

print_success "输出目录创建完成"
echo ""

################################################################################
# 步骤5: 根据类型运行实验
################################################################################

print_header "步骤5: 运行量化实验"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case $TEST_TYPE in
    quick)
        MAX_IMAGES=5
        OUTPUT_DIR="/workspace/results/quick_test_$TIMESTAMP"
        print_info "运行快速测试（5张图像）"
        ;;
    standard)
        MAX_IMAGES=10
        OUTPUT_DIR="/workspace/results/standard_test_$TIMESTAMP"
        print_info "运行标准测试（10张图像）"
        ;;
    full)
        MAX_IMAGES=50
        OUTPUT_DIR="/workspace/results/full_test_$TIMESTAMP"
        print_info "运行完整测试（50张图像）"
        ;;
    *)
        print_error "未知的测试类型: $TEST_TYPE"
        print_info "请使用: quick, standard, 或 full"
        exit 1
        ;;
esac

print_info "输出目录: $OUTPUT_DIR"
echo ""

# 运行评估
print_info "开始评估..."
START_TIME=$(date +%s)

python scripts/comprehensive_evaluation.py \
    --image_folder "$COURTYARD_IMAGES" \
    --max_images $MAX_IMAGES \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

print_success "实验完成！用时: ${DURATION}秒"
echo ""

################################################################################
# 步骤6: 显示结果摘要
################################################################################

print_header "步骤6: 实验结果"

if [ -f "$OUTPUT_DIR/comprehensive_report.txt" ]; then
    print_success "生成的文件:"
    ls -lh "$OUTPUT_DIR/"
    echo ""

    print_info "结果摘要（前50行）:"
    echo ""
    head -50 "$OUTPUT_DIR/comprehensive_report.txt"
    echo ""

    print_success "完整报告: $OUTPUT_DIR/comprehensive_report.txt"
    print_success "JSON数据: $OUTPUT_DIR/comprehensive_results.json"
    print_success "可视化图表: $OUTPUT_DIR/comprehensive_visualizations.png"
else
    print_error "结果文件未生成！"
    exit 1
fi

echo ""

################################################################################
# 完成
################################################################################

print_header "✅ 所有步骤完成！"

echo ""
print_info "实验总结:"
echo "  • 测试类型: $TEST_TYPE"
echo "  • 图像数量: $MAX_IMAGES"
echo "  • 运行时间: ${DURATION}秒"
echo "  • 输出位置: $OUTPUT_DIR"
echo ""

print_info "查看结果:"
echo "  cat $OUTPUT_DIR/comprehensive_report.txt"
echo ""

print_info "查看可视化:"
echo "  可在RunPod界面下载: $OUTPUT_DIR/comprehensive_visualizations.png"
echo ""

print_info "创建压缩包用于下载:"
echo "  cd /workspace && tar -czf results_$TIMESTAMP.tar.gz results/"
echo ""

print_header "🎉 实验成功完成！"

exit 0
