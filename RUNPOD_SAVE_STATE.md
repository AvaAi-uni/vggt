# 💾 RunPod 状态保存完全指南

## 📌 重要性

每次启动 RunPod 都需要：
- ❌ 重新下载 5GB VGGT 模型（5-10 分钟）
- ❌ 重新安装所有依赖（3-5 分钟）
- ❌ 重新克隆代码仓库（1 分钟）
- ❌ 重新配置环境（2 分钟）

**总计：每次浪费 10-20 分钟**

通过保存状态，下次启动时可以：
- ✅ 直接使用已下载的模型
- ✅ 直接使用已安装的依赖
- ✅ 直接开始量化实验

---

## 🎯 三种保存方法

### 方法 1：使用 RunPod Template（推荐）⭐⭐⭐⭐⭐

**优点**：
- 完全保存整个环境状态
- 下次启动时自动恢复所有内容
- 包含所有已安装的包、模型、代码

**步骤**：

#### 第 1 步：完成环境设置

在 RunPod 终端运行：

```bash
# 1. 克隆代码
cd /workspace
git clone https://github.com/YOUR_USERNAME/vggt.git
cd vggt

# 2. 修复依赖
pip uninstall torchaudio -y
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖
pip install -r requirements.txt
pip install matplotlib scikit-image open3d

# 4. 下载模型（这是最耗时的步骤！）
python << 'EOF'
from vggt.models.vggt import VGGT
print("Downloading VGGT-1B model...")
model = VGGT.from_pretrained("facebook/VGGT-1B")
print("✅ Model downloaded and cached!")
EOF

# 5. 验证安装
python -c "from vggt.quantization import quantize_model_advanced; print('✅ All imports OK')"
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
nvidia-smi
```

#### 第 2 步：在 RunPod 控制台保存模板

1. **停止 Pod**（不要终止！）
   - 在 RunPod 网页控制台，点击 "Stop"
   - ⚠️ **不要点击 "Terminate"**！

2. **创建 Template**
   - 点击你的 Pod
   - 点击 "Save as Template"
   - 命名：`VGGT-Quantization-Ready`
   - 描述：`VGGT model pre-downloaded, all dependencies installed`
   - 点击 "Save"

3. **下次使用**
   - 点击 "Deploy" → "Templates"
   - 选择 `VGGT-Quantization-Ready`
   - 启动 Pod
   - 所有内容都已准备好！

**成本**：
- Template 存储：免费
- 只在 Pod 运行时计费

---

### 方法 2：使用 Network Volume（推荐用于大型项目）⭐⭐⭐⭐

**优点**：
- 永久存储，不会丢失
- 可以在多个 Pod 之间共享
- 独立于 Pod 生命周期

**缺点**：
- 需要额外的存储费用（约 $0.10/GB/月）

**步骤**：

#### 创建 Network Volume

1. **在 RunPod 控制台**
   - 点击 "Storage" → "Network Volumes"
   - 点击 "Create Network Volume"
   - 名称：`vggt-workspace`
   - 大小：50 GB（足够存储模型、数据、结果）
   - 区域：选择最近的

2. **挂载到 Pod**
   - 创建新 Pod 时，在 "Volume" 选项中选择 `vggt-workspace`
   - 挂载路径：`/workspace`

#### 设置环境（首次）

```bash
# 所有内容都会保存到 Network Volume
cd /workspace

# 检查是否是首次设置
if [ ! -d "/workspace/vggt" ]; then
    echo "首次设置，开始下载..."

    # 克隆代码
    git clone https://github.com/YOUR_USERNAME/vggt.git
    cd vggt

    # 安装依赖
    pip uninstall torchaudio -y
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    pip install matplotlib scikit-image open3d

    # 下载模型到 Network Volume
    python << 'EOF'
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained("facebook/VGGT-1B")
print("✅ Model cached in Network Volume!")
EOF

    echo "✅ 环境设置完成！下次启动会直接使用。"
else
    echo "✅ 环境已存在，跳过设置。"
    cd /workspace/vggt
fi
```

#### 下次使用

```bash
# 只需要重新安装 pip 包（它们不在 Network Volume 中）
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118 -q

cd /workspace/vggt

# 模型已经在 Network Volume 中，无需重新下载！
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/quantization_comparison
```

**成本**：
- 50 GB Network Volume：$5/月
- 模型只下载一次！

---

### 方法 3：使用 Hugging Face 缓存目录（快速方案）⭐⭐⭐

**原理**：将模型下载到持久化目录

**步骤**：

#### 创建启动脚本

保存为 `/workspace/setup_vggt.sh`（在 Pod 启动时运行）：

```bash
#!/bin/bash

# VGGT 快速启动脚本

echo "🚀 VGGT 快速启动..."

# 1. 安装依赖（约 2 分钟）
echo "[1/3] 安装依赖..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118 -q
pip install matplotlib scikit-image open3d -q

# 2. 克隆代码（如果不存在）
if [ ! -d "/workspace/vggt" ]; then
    echo "[2/3] 克隆代码..."
    cd /workspace
    git clone https://github.com/YOUR_USERNAME/vggt.git
else
    echo "[2/3] 代码已存在，跳过克隆"
fi

# 3. 检查模型缓存
cd /workspace/vggt
echo "[3/3] 检查模型..."

# 设置 Hugging Face 缓存目录（保存到 RunPod 的持久化存储）
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

python << 'EOF'
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'

from vggt.models.vggt import VGGT
try:
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    print("✅ 模型加载成功！")
except:
    print("⏬ 首次下载模型（约 5 分钟）...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    print("✅ 模型下载完成！")
EOF

echo "✅ 环境就绪！开始你的实验吧。"
```

#### 使用方法

每次启动 Pod 后运行：

```bash
bash /workspace/setup_vggt.sh
```

---

## 🎯 推荐方案对比

| 方案 | 首次设置 | 后续启动 | 成本 | 推荐指数 |
|------|---------|---------|------|---------|
| **Template** | 15 分钟 | **10 秒** | 免费 | ⭐⭐⭐⭐⭐ |
| **Network Volume** | 15 分钟 | 2 分钟 | $5/月 | ⭐⭐⭐⭐ |
| **启动脚本** | 15 分钟 | 2-5 分钟 | 免费 | ⭐⭐⭐ |

---

## 📋 完整设置检查清单

### ✅ 首次设置（保存前）

```bash
# 1. 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 2. 验证 CUDA
nvidia-smi

# 3. 验证模型已下载
python -c "from vggt.models.vggt import VGGT; model = VGGT.from_pretrained('facebook/VGGT-1B'); print('✅ Model OK')"

# 4. 验证量化模块
python -c "from vggt.quantization import quantize_model_advanced; print('✅ Quantization OK')"

# 5. 检查文件结构
ls -lh /workspace/vggt/
ls -lh ~/.cache/huggingface/hub/  # 模型缓存位置
```

如果所有检查都通过 ✅，可以保存状态了！

---

## 🔄 快速恢复流程（使用 Template）

### 第 1 次启动（设置环境）

```bash
# 总时间：15 分钟
cd /workspace
git clone https://github.com/YOUR_USERNAME/vggt.git
cd vggt

# 运行完整设置脚本
bash << 'EOF'
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install matplotlib scikit-image open3d

# 下载模型
python -c "from vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')"
echo "✅ 环境设置完成！现在保存 Template。"
EOF

# 停止 Pod → Save as Template
```

### 第 2 次及以后（使用 Template）

```bash
# 总时间：10 秒！
cd /workspace/vggt

# 直接开始工作！
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/quantization_comparison
```

---

## 🎁 额外提示

### 1. 保存实验结果

即使使用 Template，实验结果也应该下载到本地：

```bash
# 在 RunPod 上打包
cd /workspace
tar -czf results_$(date +%Y%m%d).tar.gz \
    quantization_comparison/ \
    models/ \
    visualizations/

# 在本地下载
scp -P <PORT> root@<POD_IP>:/workspace/results_*.tar.gz ./
```

### 2. 自动化启动脚本

创建 `/workspace/vggt/quick_start.sh`：

```bash
#!/bin/bash
cd /workspace/vggt
echo "🚀 VGGT 量化实验环境"
echo "选择操作："
echo "  1) 运行量化对比（5 张图）"
echo "  2) 运行量化对比（10 张图）"
echo "  3) 下载 ETH3D 数据集"
echo "  4) 查看上次结果"
read -p "输入选项 [1-4]: " choice

case $choice in
    1)
        python scripts/compare_quantization.py \
            --image_folder /workspace/data/eth3d/courtyard/images \
            --max_images 5 \
            --output_dir /workspace/quantization_comparison
        ;;
    2)
        python scripts/compare_quantization.py \
            --image_folder /workspace/data/eth3d/courtyard/images \
            --max_images 10 \
            --output_dir /workspace/quantization_comparison
        ;;
    3)
        python scripts/download_eth3d.py --output_dir /workspace/data/eth3d
        ;;
    4)
        cat /workspace/quantization_comparison/comparison_summary.txt
        ;;
esac
```

使用：
```bash
bash /workspace/vggt/quick_start.sh
```

### 3. 检查 Template 内容

在保存 Template 之前，验证这些文件存在：

```bash
# 检查模型缓存
ls ~/.cache/huggingface/hub/ | grep VGGT

# 检查代码
ls /workspace/vggt/scripts/

# 检查 Python 包
pip list | grep torch
```

---

## 🆘 故障排除

### 问题 1：Template 保存后模型仍需重新下载

**原因**：Hugging Face 缓存可能在 `/tmp` 或其他临时目录

**解决**：
```bash
# 设置永久缓存目录
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/.cache/huggingface' >> ~/.bashrc
source ~/.bashrc

# 重新下载模型到正确位置
python -c "from vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')"
```

### 问题 2：Template 太大

**解决**：清理不必要的文件
```bash
# 清理 pip 缓存
pip cache purge

# 清理临时文件
rm -rf /tmp/*

# 清理 Jupyter 缓存
jupyter --paths
rm -rf ~/.local/share/jupyter/
```

### 问题 3：Network Volume 没有挂载

**检查**：
```bash
df -h | grep workspace
ls -la /workspace/
```

如果没有看到 Network Volume，在 RunPod 控制台重新挂载。

---

## 📊 成本对比

### 不使用状态保存

每次启动：
- 设置时间：15 分钟
- GPU 成本：15/60 × $0.40 = **$0.10**
- 每月 10 次启动：**$1.00**

### 使用 Template

每次启动：
- 设置时间：10 秒
- GPU 成本：0 分钟（立即开始工作）
- 每月节省：**$1.00**
- Template 存储：**免费**

**年度节省：$12 + 大量时间！**

---

## ✅ 推荐工作流程

### 最佳实践：Template + Network Volume

1. **创建 Network Volume**（一次性）
   - 50 GB
   - 存储数据集和实验结果

2. **创建 Template**（一次性）
   - 包含所有依赖和模型
   - 快速启动环境

3. **日常使用**
   - 使用 Template 启动 Pod（10 秒）
   - 数据和结果存储在 Network Volume
   - 实验完成后停止 Pod
   - 重要结果下载到本地

**总成本**：$5/月（Network Volume）+ GPU 按需计费

---

## 🎯 总结

**最推荐方案：使用 RunPod Template ⭐⭐⭐⭐⭐**

**原因**：
- ✅ 完全免费
- ✅ 最快启动（10 秒）
- ✅ 包含所有内容
- ✅ 简单易用

**步骤**：
1. 首次设置环境（15 分钟）
2. 保存 Template
3. 以后每次使用 Template 启动

**下次启动时**：
```bash
cd /workspace/vggt
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/quantization_comparison
```

**立即开始工作，无需等待！** 🚀

---

**最后更新**：2025-10-13
**维护者**：Your Team
