#!/bin/bash
# ============================================================================
# VGGT 快速修复脚本 - 解决所有依赖和路径问题
# ============================================================================
#
# 使用方法：
#   bash quick_fix.sh
#
# ============================================================================

echo "============================================================================"
echo "VGGT 快速修复 - 解决依赖和路径问题"
echo "============================================================================"
echo ""

# 修复 NumPy 版本冲突
echo "[1/4] 修复 NumPy 版本冲突..."
pip uninstall -y numpy 2>/dev/null || true
pip install "numpy>=1.21.0,<2.0.0"
echo "✓ NumPy 版本已修复"
echo ""

# 安装缺失的 wcmatch
echo "[2/4] 安装 wcmatch..."
pip install wcmatch>=8.4.0
echo "✓ wcmatch 已安装"
echo ""

# 安装 vggt 包（editable mode）
echo "[3/4] 安装 vggt 包..."
pip install -e .
echo "✓ vggt 包已安装"
echo ""

# 验证修复
echo "[4/4] 验证修复..."
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

# 检查 vggt 模块
try:
    import vggt
    from vggt.utils.geometry import closed_form_inverse_se3
    print('✓ vggt 模块: 已安装')
except Exception as e:
    print(f'✗ vggt 模块: {e}')
    sys.exit(1)

print()
print('✓ 所有依赖和模块正常！可以开始训练了。')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✓ 修复完成！"
    echo "============================================================================"
    echo ""
    echo "现在可以运行："
    echo "  bash train.sh eth3d_fp32_quick_test"
    echo ""
    echo "或者："
    echo "  cd training"
    echo "  export PYTHONPATH=/workspace/vggt:\$PYTHONPATH"
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
