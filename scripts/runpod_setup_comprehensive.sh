#!/bin/bash
# RunPod 完整量化实验环境设置脚本
# 版本: 2.0 - Comprehensive Framework
# 日期: 2025-10-16

set -e  # 遇到错误立即退出

echo "=============================================================================="
echo "RunPod 完整量化实验环境设置"
echo "=============================================================================="
echo ""

# ============================================================================
# 步骤 0: 环境检查
# ============================================================================
echo "[步骤 0/6] 检查环境..."
echo "------------------------------------------------------------------------------"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA 可用"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  警告: CUDA 不可用，将使用CPU（较慢）"
fi

# 检查Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo "✓ Python: $PYTHON_VERSION"
else
    echo "❌ 错误: Python 未安装"
    exit 1
fi

# 检查磁盘空间
DISK_AVAIL=$(df -h /workspace | tail -1 | awk '{print $4}')
echo "✓ 可用磁盘空间: $DISK_AVAIL"

echo ""

# ============================================================================
# 步骤 1: 创建目录结构
# ============================================================================
echo "[步骤 1/6] 创建目录结构..."
echo "------------------------------------------------------------------------------"

# 工作目录（RunPod持久化目录）
WORKSPACE_DIR="/workspace"
PROJECT_DIR="$WORKSPACE_DIR/vggt"
DATA_DIR="$WORKSPACE_DIR/data"
RESULTS_DIR="$WORKSPACE_DIR/results"
MODELS_DIR="$WORKSPACE_DIR/models"

# 创建目录
mkdir -p $PROJECT_DIR
mkdir -p $DATA_DIR
mkdir -p $RESULTS_DIR
mkdir -p $MODELS_DIR

echo "✓ 工作目录: $WORKSPACE_DIR"
echo "✓ 项目目录: $PROJECT_DIR"
echo "✓ 数据目录: $DATA_DIR"
echo "✓ 结果目录: $RESULTS_DIR"
echo "✓ 模型目录: $MODELS_DIR"
echo ""

# ============================================================================
# 步骤 2: 安装依赖
# ============================================================================
echo "[步骤 2/6] 安装Python依赖..."
echo "------------------------------------------------------------------------------"

# 检查是否已在项目目录
if [ ! -f "$PROJECT_DIR/scripts/comprehensive_evaluation.py" ]; then
    echo "⚠️  警告: 项目文件不存在"
    echo "请先上传项目文件到 $PROJECT_DIR"
    echo ""
    echo "上传方法:"
    echo "  方法1: 使用 git clone"
    echo "    cd /workspace"
    echo "    git clone <YOUR_REPO_URL> vggt"
    echo ""
    echo "  方法2: 使用 scp/rsync 上传本地文件"
    echo "    scp -r ./vggt runpod:$PROJECT_DIR"
    echo ""
    exit 1
fi

cd $PROJECT_DIR

# 升级pip
echo "升级 pip..."
pip install --upgrade pip -q

# 安装基础依赖
echo "安装基础依赖..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

# 安装其他依赖
echo "安装其他依赖..."
pip install -q \
    numpy \
    matplotlib \
    Pillow \
    scipy \
    tqdm \
    huggingface_hub

# 如果有requirements.txt，安装
if [ -f "requirements.txt" ]; then
    echo "从 requirements.txt 安装依赖..."
    pip install -r requirements.txt -q
fi

echo "✓ 依赖安装完成"
echo ""

# ============================================================================
# 步骤 3: 验证PyTorch和CUDA
# ============================================================================
echo "[步骤 3/6] 验证PyTorch和CUDA..."
echo "------------------------------------------------------------------------------"

python << EOF
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  警告: CUDA不可用，将使用CPU")
EOF

echo ""

# ============================================================================
# 步骤 4: 下载测试数据（可选）
# ============================================================================
echo "[步骤 4/6] 下载测试数据..."
echo "------------------------------------------------------------------------------"

read -p "是否下载ETH3D测试数据? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "下载ETH3D数据集..."

    if [ -f "$PROJECT_DIR/scripts/download_eth3d.py" ]; then
        python $PROJECT_DIR/scripts/download_eth3d.py \
            --output_dir $DATA_DIR/eth3d
        echo "✓ ETH3D数据集下载完成"
    else
        echo "⚠️  警告: download_eth3d.py 未找到，跳过数据下载"
        echo "你可以稍后手动下载数据或使用自己的图像"
    fi
else
    echo "跳过数据下载"
    echo ""
    echo "你可以稍后手动下载："
    echo "  python scripts/download_eth3d.py --output_dir $DATA_DIR/eth3d"
    echo ""
    echo "或使用自己的图像："
    echo "  mkdir -p $DATA_DIR/my_images"
    echo "  # 上传图像到 $DATA_DIR/my_images"
fi

echo ""

# ============================================================================
# 步骤 5: 准备示例测试图像
# ============================================================================
echo "[步骤 5/6] 检查测试图像..."
echo "------------------------------------------------------------------------------"

# 搜索可用的图像
IMAGE_COUNT=0

if [ -d "$DATA_DIR/eth3d" ]; then
    IMAGE_COUNT=$(find $DATA_DIR/eth3d -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)
fi

if [ $IMAGE_COUNT -gt 0 ]; then
    echo "✓ 找到 $IMAGE_COUNT 张测试图像"

    # 显示第一个图像目录
    FIRST_IMAGE_DIR=$(find $DATA_DIR/eth3d -type f \( -iname "*.jpg" -o -iname "*.png" \) | head -1 | xargs dirname)
    echo "  示例图像目录: $FIRST_IMAGE_DIR"
else
    echo "⚠️  警告: 未找到测试图像"
    echo ""
    echo "请准备测试图像："
    echo "  方法1: 下载ETH3D数据集"
    echo "    python scripts/download_eth3d.py --output_dir $DATA_DIR/eth3d"
    echo ""
    echo "  方法2: 上传自己的图像"
    echo "    mkdir -p $DATA_DIR/my_images"
    echo "    # 上传.jpg或.png图像到该目录"
fi

echo ""

# ============================================================================
# 步骤 6: 创建快捷命令
# ============================================================================
echo "[步骤 6/6] 创建快捷命令..."
echo "------------------------------------------------------------------------------"

# 创建快捷命令脚本
cat > $WORKSPACE_DIR/run_quick_test.sh << 'SCRIPT_EOF'
#!/bin/bash
# 快速测试（5张图像）
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/results/quick_test \
    --device cuda
SCRIPT_EOF

cat > $WORKSPACE_DIR/run_standard_test.sh << 'SCRIPT_EOF'
#!/bin/bash
# 标准测试（10张图像）
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 10 \
    --output_dir /workspace/results/standard_test \
    --device cuda
SCRIPT_EOF

cat > $WORKSPACE_DIR/run_full_test.sh << 'SCRIPT_EOF'
#!/bin/bash
# 完整测试（50张图像）
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 50 \
    --output_dir /workspace/results/full_test \
    --device cuda
SCRIPT_EOF

# 添加执行权限
chmod +x $WORKSPACE_DIR/run_quick_test.sh
chmod +x $WORKSPACE_DIR/run_standard_test.sh
chmod +x $WORKSPACE_DIR/run_full_test.sh

echo "✓ 快捷命令已创建:"
echo "  快速测试: bash /workspace/run_quick_test.sh"
echo "  标准测试: bash /workspace/run_standard_test.sh"
echo "  完整测试: bash /workspace/run_full_test.sh"
echo ""

# ============================================================================
# 完成
# ============================================================================
echo "=============================================================================="
echo "✅ 环境设置完成！"
echo "=============================================================================="
echo ""
echo "📂 目录结构:"
echo "  工作空间: $WORKSPACE_DIR"
echo "  项目目录: $PROJECT_DIR"
echo "  数据目录: $DATA_DIR"
echo "  结果目录: $RESULTS_DIR"
echo ""
echo "🚀 快速开始:"
echo ""
echo "  方法1: 使用快捷命令（推荐）"
echo "    bash /workspace/run_quick_test.sh"
echo ""
echo "  方法2: 手动运行"
echo "    cd /workspace/vggt"
echo "    python scripts/comprehensive_evaluation.py \\"
echo "      --image_folder /workspace/data/eth3d/courtyard/images \\"
echo "      --max_images 5 \\"
echo "      --output_dir /workspace/results/my_test"
echo ""
echo "📊 查看结果:"
echo "  cat /workspace/results/quick_test/comprehensive_report.txt"
echo ""
echo "📚 文档:"
echo "  cat /workspace/vggt/START_HERE_COMPREHENSIVE.md"
echo ""
echo "=============================================================================="
