#!/bin/bash
# ============================================================================
# VGGT 快速修复脚本 - 专门解决 NumPy 和 wcmatch 问题
# ============================================================================
#
# 使用方法：
#   bash quick_fix.sh
#
# ============================================================================

echo "============================================================================"
echo "VGGT 快速修复 - 解决依赖问题"
echo "============================================================================"
echo ""

# 修复 NumPy 版本冲突
echo "[1/3] 修复 NumPy 版本冲突..."
pip uninstall -y numpy 2>/dev/null || true
pip install "numpy>=1.21.0,<2.0.0"
echo "✓ NumPy 版本已修复"
echo ""

# 安装缺失的 wcmatch
echo "[2/3] 安装 wcmatch..."
pip install wcmatch>=8.4.0
echo "✓ wcmatch 已安装"
echo ""

# 验证修复
echo "[3/3] 验证修复..."
python -c "
import sys

# 检查 NumPy
try:
    import numpy as np
    version = np.__version__
    major = int(version.split('.')[0])
    if major >= 2:
        print(f'✗ NumPy 版本错误: {version} (需要 1.x)')
        sys.exit(1)
    else:
        print(f'✓ NumPy: {version}')
except Exception as e:
    print(f'✗ NumPy 导入失败: {e}')
    sys.exit(1)

# 检查 wcmatch
try:
    from wcmatch import fnmatch
    print('✓ wcmatch: 已安装')
except Exception as e:
    print(f'✗ wcmatch 导入失败: {e}')
    sys.exit(1)

# 检查 PyTorch
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'✓ CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'✗ PyTorch 导入失败: {e}')
    sys.exit(1)

print()
print('✓ 所有依赖正常！可以开始训练了。')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✓ 修复完成！"
    echo "============================================================================"
    echo ""
    echo "现在可以运行："
    echo "  cd training"
    echo "  python launch.py --config eth3d_fp32_quick_test"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "✗ 修复失败！"
    echo "============================================================================"
    echo ""
    echo "请尝试完整安装："
    echo "  bash runpod_setup.sh"
    echo ""
    exit 1
fi
