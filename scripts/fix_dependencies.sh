#!/bin/bash
# 快速修复脚本 - 解决 RunPod 环境中的依赖问题
# 用于修复 torchaudio 版本冲突等问题

set -e

echo "=========================================="
echo "VGGT Dependency Fix Script"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. 显示当前状态
echo -e "${GREEN}[1/4] Checking current environment...${NC}"
echo "Current PyTorch version:"
python -c "import torch; print(f'  torch: {torch.__version__}')" 2>/dev/null || echo "  torch: NOT INSTALLED"
python -c "import torchvision; print(f'  torchvision: {torchvision.__version__}')" 2>/dev/null || echo "  torchvision: NOT INSTALLED"
python -c "import torchaudio; print(f'  torchaudio: {torchaudio.__version__}')" 2>/dev/null || echo "  torchaudio: NOT INSTALLED"
echo ""

# 2. 卸载冲突的包
echo -e "${GREEN}[2/4] Removing conflicting packages...${NC}"
pip uninstall torchaudio -y 2>/dev/null || echo "  No torchaudio to uninstall"
echo "✓ Conflicting packages removed"
echo ""

# 3. 安装正确的 PyTorch 版本
echo -e "${GREEN}[3/4] Installing PyTorch 2.3.1 with CUDA 11.8...${NC}"
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch installed"
echo ""

# 4. 验证安装
echo -e "${GREEN}[4/4] Verifying installation...${NC}"
python -c "
import torch
import torchvision
import torchaudio

print('✓ All packages imported successfully')
print(f'  torch: {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  torchaudio: {torchaudio.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"
echo ""

# 完成
echo "=========================================="
echo -e "${GREEN}Fix Complete!${NC}"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  python scripts/quantize_model.py --help"
echo ""
