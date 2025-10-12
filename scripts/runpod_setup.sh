#!/bin/bash
# RunPod 环境快速设置脚本
# 用于在 RunPod 上快速配置 VGGT INT8 量化环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "VGGT INT8 Quantization - RunPod Setup"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在 RunPod 环境
if [ ! -d "/workspace" ]; then
    echo -e "${YELLOW}Warning: /workspace directory not found. This script is designed for RunPod.${NC}"
    WORKSPACE_DIR="$PWD"
else
    WORKSPACE_DIR="/workspace"
fi

echo "Working directory: $WORKSPACE_DIR"
echo ""

# 1. 更新系统包
echo -e "${GREEN}[1/7] Updating system packages...${NC}"
apt-get update -qq
apt-get install -y -qq p7zip-full wget curl git tmux htop > /dev/null 2>&1
echo "✓ System packages updated"
echo ""

# 2. 克隆仓库 (如果还没有)
echo -e "${GREEN}[2/7] Setting up code repository...${NC}"
cd $WORKSPACE_DIR

if [ ! -d "vggt" ]; then
    echo "Cloning VGGT repository..."
    git clone https://github.com/facebookresearch/vggt.git
    cd vggt
else
    echo "Repository already exists, updating..."
    cd vggt
    git pull
fi

echo "✓ Repository ready"
echo ""

# 3. 安装 Python 依赖
echo -e "${GREEN}[3/7] Installing Python dependencies...${NC}"
pip install --upgrade pip -q

# 修复 torchaudio 版本冲突
echo "Removing conflicting torchaudio version..."
pip uninstall torchaudio -y > /dev/null 2>&1 || true

# 安装正确版本的 PyTorch 生态系统
echo "Installing PyTorch 2.3.1 with CUDA 11.8..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 -q

# 安装其他依赖
echo "Installing other dependencies..."
pip install numpy==1.26.1 Pillow huggingface_hub einops safetensors -q
pip install -r requirements_demo.txt -q

echo "✓ Dependencies installed"
echo ""

# 4. 验证 GPU
echo -e "${GREEN}[4/7] Verifying GPU...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "✓ GPU verification complete"
echo ""

# 5. 创建必要的目录
echo -e "${GREEN}[5/7] Creating directories...${NC}"
mkdir -p $WORKSPACE_DIR/data
mkdir -p $WORKSPACE_DIR/models
mkdir -p $WORKSPACE_DIR/outputs
mkdir -p $WORKSPACE_DIR/logs
echo "✓ Directories created:"
echo "  - $WORKSPACE_DIR/data (for datasets)"
echo "  - $WORKSPACE_DIR/models (for model checkpoints)"
echo "  - $WORKSPACE_DIR/outputs (for results)"
echo "  - $WORKSPACE_DIR/logs (for log files)"
echo ""

# 6. 下载 ETH3D 数据集 (可选)
echo -e "${GREEN}[6/7] ETH3D Dataset...${NC}"
read -p "Download ETH3D dataset? (requires ~10GB, takes ~15 mins) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading ETH3D dataset..."
    python scripts/download_eth3d.py --output_dir $WORKSPACE_DIR/data/eth3d
    echo "✓ ETH3D dataset downloaded"
else
    echo "⊘ Skipping ETH3D download"
    echo "  You can download it later with:"
    echo "  python scripts/download_eth3d.py --output_dir $WORKSPACE_DIR/data/eth3d"
fi
echo ""

# 7. 量化模型 (可选)
echo -e "${GREEN}[7/7] Model Quantization...${NC}"
read -p "Quantize VGGT model now? (requires model download ~4GB) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting model quantization..."
    echo "This will:"
    echo "  1. Download VGGT-1B model (~4GB)"
    echo "  2. Apply dynamic INT8 quantization"
    echo "  3. Save quantized model to $WORKSPACE_DIR/models/"
    echo ""

    python scripts/quantize_model.py \
        --model_name facebook/VGGT-1B \
        --quantization_type dynamic \
        --output_path $WORKSPACE_DIR/models/vggt_int8_dynamic.pt \
        --quantize_attention \
        --quantize_heads

    echo "✓ Model quantized successfully"
else
    echo "⊘ Skipping quantization"
    echo "  You can quantize later with:"
    echo "  python scripts/quantize_model.py --model_name facebook/VGGT-1B --quantization_type dynamic --output_path $WORKSPACE_DIR/models/vggt_int8_dynamic.pt"
fi
echo ""

# 完成
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check GPU: nvidia-smi"
echo "  2. Run inference: python inference_quantized.py"
echo "  3. Start Gradio demo: python demo_gradio.py"
echo "  4. Read full documentation: cat RUNPOD_DEPLOYMENT.md"
echo ""
echo "Quick commands:"
echo "  - Monitor GPU: watch -n 1 nvidia-smi"
echo "  - Keep session alive: tmux new -s vggt"
echo "  - View logs: tail -f $WORKSPACE_DIR/logs/quantize.log"
echo ""
echo "For help, see RUNPOD_DEPLOYMENT.md"
echo ""
