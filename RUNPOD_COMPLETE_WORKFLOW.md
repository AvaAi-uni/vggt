# 🚀 RunPod 完整操作流程 - 从零到完成

本文档包含在 RunPod 上完成 VGGT INT8/INT4 量化对比实验的**所有命令**，按顺序执行即可。

**预计总时间**: 60-90 分钟
**预计成本**: $0.50-$1.00 (使用 RTX 4090)

---

## 💾 重要提示：保存环境状态 ⭐

**首次使用后，强烈建议保存 RunPod 状态，避免下次重复设置！**

完成**阶段 1**（环境设置）后，建议保存 Template：
1. 在 RunPod 控制台点击 "Stop"（不要 Terminate）
2. 点击 "Save as Template"
3. 命名：`VGGT-Quantization-Ready`
4. 下次使用该 Template 启动，环境已完全配置好

**详细指南**: 见 `RUNPOD_SAVE_STATE.md` 文档

**好处**:
- ✅ 下次启动只需 10 秒
- ✅ 模型已下载（节省 5-10 分钟）
- ✅ 依赖已安装（节省 3-5 分钟）
- ✅ 每次节省 $0.10 + 大量时间

---

## 📋 前置准备

### 在 RunPod 网站完成：

1. 注册/登录 RunPod: https://www.runpod.io/
2. 充值账户（建议 $10）
3. 选择 GPU:
   - **推荐**: RTX 4090 (24GB) - $0.4/小时
   - 备选: A6000 (48GB) - $0.8/小时
4. 配置:
   - Container Disk: 50GB
   - 镜像: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
5. 点击 **Deploy**
6. 等待 Pod 启动
7. 点击 **Connect** → **Start Web Terminal**

---

## 🎯 完整操作流程

### 阶段 1: 环境设置 (5-10 分钟)

#### 步骤 1.1: 进入工作目录并克隆仓库

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/vggt.git
cd vggt
```

**说明**: 将 `YOUR_USERNAME` 替换为你的 GitHub 用户名

---

#### 步骤 1.2: 安装系统依赖

```bash
apt-get update && apt-get install -y p7zip-full wget curl git tmux htop
```

**预计时间**: 2-3 分钟

---

#### 步骤 1.3: 修复 Python 依赖

```bash
# 卸载冲突的 torchaudio
pip uninstall torchaudio -y

# 安装正确版本的 PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy==1.26.1 Pillow huggingface_hub einops safetensors

# 安装可视化依赖
pip install matplotlib plotly
```

**预计时间**: 3-5 分钟

---

#### 步骤 1.4: 验证环境

```bash
# 验证 PyTorch
python -c "import torch, torchvision, torchaudio; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 验证 GPU
nvidia-smi

# 验证量化模块
python -c "from vggt.quantization import quantize_model_advanced, AdvancedQuantConfig; print('✅ All modules loaded!')"
```

**预期输出**:
```
torch: 2.3.1
torchvision: 0.18.1
torchaudio: 2.3.1
CUDA: True
✅ All modules loaded!
```

---

### 阶段 2: 下载数据集 (10-15 分钟)

#### 步骤 2.1: 下载 ETH3D 数据集

```bash
# 运行下载脚本
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d
```

**预计时间**: 10-15 分钟（取决于网速）
**数据大小**: ~10GB

---

#### 步骤 2.2: 验证数据集

```bash
# 查看下载的场景
ls /workspace/data/eth3d/

# 查看 courtyard 场景的图像
ls /workspace/data/eth3d/courtyard/images/ | head -10

# 统计图像数量
ls /workspace/data/eth3d/courtyard/images/ | wc -l
```

**预期输出**: 应该看到多个场景和图像文件

---

### 阶段 3: 量化对比实验 (15-20 分钟) ⭐ 核心

#### 步骤 3.1: 运行完整的量化对比

```bash
# 创建输出目录
mkdir -p /workspace/quantization_comparison

# 运行对比实验（对比 5 种量化方案）
python scripts/compare_quantization.py \
    --model_name facebook/VGGT-1B \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/quantization_comparison \
    --device cuda
```

**预计时间**: 15-20 分钟

**说明**: 这会自动测试以下方案:
1. 原始 FP32 模型（基准）
2. ~~PyTorch Dynamic INT8~~ ⚠️ 已跳过（与 VGGT 自定义层不兼容）
3. INT8 对称量化（自定义实现）✅
4. INT8 非对称量化（自定义实现）✅
5. INT4 分组量化 Group-128（自定义实现）✅
6. INT4 分组量化 Group-64（自定义实现）✅

**注意**: PyTorch 标准动态量化与 VGGT 的自定义 Attention 层不兼容，已自动跳过。
我们的自定义量化方法（INT8 Symmetric/Asymmetric, INT4 Group）完全支持 VGGT。

---

#### 步骤 3.2: 查看对比结果

```bash
# 查看文本报告
cat /workspace/quantization_comparison/comparison_summary.txt

# 查看 JSON 数据（格式化）
python -m json.tool /workspace/quantization_comparison/comparison_report.json | head -50

# 查看生成的文件
ls -lh /workspace/quantization_comparison/
```

---

### 阶段 4: 保存最佳量化模型 (5-10 分钟)

#### 步骤 4.1: 保存 PyTorch Dynamic 量化模型（推荐）

```bash
mkdir -p /workspace/models

python scripts/quantize_model.py \
    --model_name facebook/VGGT-1B \
    --quantization_type dynamic \
    --output_path /workspace/models/vggt_int8_dynamic.pt \
    --quantize_attention \
    --quantize_heads
```

**预计时间**: 5-8 分钟

---

#### 步骤 4.2: 保存其他量化方案（可选）

```bash
# INT8 对称量化
python << 'EOF'
import torch
from vggt.models.vggt import VGGT
from vggt.quantization import quantize_model_advanced, AdvancedQuantConfig

print("Quantizing INT8 Symmetric...")
model = VGGT.from_pretrained("facebook/VGGT-1B").cuda()
config = AdvancedQuantConfig(quant_type="int8_symmetric", bits=8)
quant_model = quantize_model_advanced(model, config)
torch.save(quant_model.state_dict(), "/workspace/models/vggt_int8_symmetric.pt")
print("✅ Saved: vggt_int8_symmetric.pt")
EOF
```

```bash
# INT8 非对称量化
python << 'EOF'
import torch
from vggt.models.vggt import VGGT
from vggt.quantization import quantize_model_advanced, AdvancedQuantConfig

print("Quantizing INT8 Asymmetric...")
model = VGGT.from_pretrained("facebook/VGGT-1B").cuda()
config = AdvancedQuantConfig(quant_type="int8_asymmetric", bits=8)
quant_model = quantize_model_advanced(model, config)
torch.save(quant_model.state_dict(), "/workspace/models/vggt_int8_asymmetric.pt")
print("✅ Saved: vggt_int8_asymmetric.pt")
EOF
```

```bash
# INT4 分组量化 (Group-128)
python << 'EOF'
import torch
from vggt.models.vggt import VGGT
from vggt.quantization import quantize_model_advanced, AdvancedQuantConfig

print("Quantizing INT4 Group-128...")
model = VGGT.from_pretrained("facebook/VGGT-1B").cuda()
config = AdvancedQuantConfig(quant_type="int4_group", bits=4, group_size=128)
quant_model = quantize_model_advanced(model, config)
torch.save(quant_model.state_dict(), "/workspace/models/vggt_int4_group128.pt")
print("✅ Saved: vggt_int4_group128.pt")
EOF
```

---

#### 步骤 4.3: 验证保存的模型

```bash
# 查看所有模型
ls -lh /workspace/models/

# 验证模型可以加载
python << 'EOF'
import torch
from vggt.models.vggt import VGGT
from vggt.quantization import estimate_model_size

model = VGGT()
model.load_state_dict(torch.load("/workspace/models/vggt_int8_dynamic.pt"))

size = estimate_model_size(model)
print(f"✅ Model loaded successfully!")
print(f"   Size: {size['total_mb']:.2f} MB")
EOF
```

---

### 阶段 5: 生成可视化 (10-15 分钟)

#### 步骤 5.1: 可视化量化模型的推理结果

```bash
# 生成深度图、点云、相机轨迹可视化
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/dynamic \
    --max_images 10
```

**预计时间**: 5-8 分钟

---

#### 步骤 5.2: 对比原始模型和量化模型（可选）

```bash
# 生成原始 vs 量化的对比
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --original_model facebook/VGGT-1B \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/comparison \
    --max_images 5
```

**预计时间**: 8-10 分钟

---

#### 步骤 5.3: 查看生成的可视化

```bash
# 查看文件
ls -lh /workspace/visualizations/dynamic/

# 启动 HTTP 服务器查看
cd /workspace/visualizations/dynamic
python -m http.server 8000 &
```

然后在 RunPod 控制台：
- 点击 "Connect" → "HTTP Service [Port 8000]"
- 在浏览器中打开 `index.html`

---

### 阶段 6: 测试推理性能 (5 分钟)

#### 步骤 6.1: 快速推理测试

```bash
# 测试量化模型推理
python scripts/inference_quantized.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/outputs/test \
    --max_images 5
```

---

#### 步骤 6.2: 查看推理结果

```bash
# 查看输出摘要
cat /workspace/outputs/test/summary.txt

# 查看生成的文件
ls -lh /workspace/outputs/test/
```

---

### 阶段 7: 打包和准备下载 (2-5 分钟)

#### 步骤 7.1: 打包所有结果

```bash
cd /workspace

# 创建完整的结果包
tar -czf experiment_results.tar.gz \
    quantization_comparison/ \
    visualizations/ \
    models/ \
    outputs/

# 查看文件大小
ls -lh experiment_results.tar.gz
```

---

#### 步骤 7.2: 创建实验报告

```bash
# 生成实验摘要
cat > /workspace/EXPERIMENT_REPORT.txt << 'EOF'
===============================================
VGGT Quantization Experiment Report
===============================================

Experiment Date: $(date)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
Dataset: ETH3D (courtyard scene)

Results Location:
- Comparison Report: /workspace/quantization_comparison/
- Quantized Models: /workspace/models/
- Visualizations: /workspace/visualizations/
- Inference Outputs: /workspace/outputs/

Quantization Methods Tested:
1. PyTorch Dynamic INT8
2. INT8 Symmetric
3. INT8 Asymmetric
4. INT4 Group-128
5. INT4 Group-64

See comparison_summary.txt for detailed results.
===============================================
EOF

cat /workspace/EXPERIMENT_REPORT.txt
```

---

### 阶段 8: 下载到本地

#### 步骤 8.1: 获取下载信息

```bash
# 显示下载命令
echo "================================"
echo "Download Command (run on local machine):"
echo "================================"
echo ""
echo "scp -P <PORT> root@<POD_IP>:/workspace/experiment_results.tar.gz ./"
echo ""
echo "Get PORT and POD_IP from RunPod Console:"
echo "  1. Click your Pod"
echo "  2. Click 'Connect'"
echo "  3. Click 'TCP Port Mappings'"
echo "  4. Find SSH port mapping"
echo ""
echo "File size: $(du -h /workspace/experiment_results.tar.gz | cut -f1)"
```

---

#### 步骤 8.2: 在本地电脑下载

**在你的本地电脑终端运行：**

```bash
# 替换 <PORT> 和 <POD_IP> 为实际值
scp -P <PORT> root@<POD_IP>:/workspace/experiment_results.tar.gz ./

# 解压
tar -xzf experiment_results.tar.gz

# 查看结果
cat quantization_comparison/comparison_summary.txt
open visualizations/dynamic/index.html  # macOS
# 或
xdg-open visualizations/dynamic/index.html  # Linux
# 或
start visualizations\dynamic\index.html  # Windows
```

---

## 🎯 完整流程一键脚本（可选）

将上述所有步骤合并为一个脚本：

```bash
# 创建完整流程脚本
cat > /workspace/run_full_experiment.sh << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "VGGT Quantization Complete Workflow"
echo "=========================================="

# 阶段 1: 环境设置
echo "[1/7] Setting up environment..."
cd /workspace/vggt
pip uninstall torchaudio -y > /dev/null 2>&1 || true
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 -q
pip install numpy==1.26.1 Pillow huggingface_hub einops safetensors matplotlib plotly -q

# 阶段 2: 下载数据
echo "[2/7] Downloading ETH3D dataset..."
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d

# 阶段 3: 量化对比
echo "[3/7] Running quantization comparison..."
mkdir -p /workspace/quantization_comparison
python scripts/compare_quantization.py \
    --model_name facebook/VGGT-1B \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/quantization_comparison

# 阶段 4: 保存模型
echo "[4/7] Saving best quantized model..."
mkdir -p /workspace/models
python scripts/quantize_model.py \
    --model_name facebook/VGGT-1B \
    --quantization_type dynamic \
    --output_path /workspace/models/vggt_int8_dynamic.pt \
    --quantize_attention \
    --quantize_heads

# 阶段 5: 生成可视化
echo "[5/7] Generating visualizations..."
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/dynamic \
    --max_images 10

# 阶段 6: 测试推理
echo "[6/7] Testing inference..."
python scripts/inference_quantized.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/outputs/test \
    --max_images 5

# 阶段 7: 打包结果
echo "[7/7] Packaging results..."
cd /workspace
tar -czf experiment_results.tar.gz \
    quantization_comparison/ \
    visualizations/ \
    models/ \
    outputs/

echo ""
echo "=========================================="
echo "✅ Complete!"
echo "=========================================="
echo ""
echo "Download: /workspace/experiment_results.tar.gz"
echo "Size: $(du -h /workspace/experiment_results.tar.gz | cut -f1)"
echo ""
echo "View results:"
echo "  cat /workspace/quantization_comparison/comparison_summary.txt"
echo ""
EOF

# 赋予执行权限
chmod +x /workspace/run_full_experiment.sh

# 运行脚本
bash /workspace/run_full_experiment.sh
```

**预计总时间**: 60-90 分钟

---

## 📊 预期结果摘要

完成后，你应该有：

### 文件结构
```
/workspace/
├── vggt/                              # 代码仓库
├── data/eth3d/                        # ETH3D 数据集
├── quantization_comparison/           # 对比结果
│   ├── comparison_report.json
│   ├── comparison_summary.txt
│   └── comparison_plots.png
├── models/                            # 量化模型
│   ├── vggt_int8_dynamic.pt
│   ├── vggt_int8_symmetric.pt (可选)
│   ├── vggt_int8_asymmetric.pt (可选)
│   └── vggt_int4_group128.pt (可选)
├── visualizations/                    # 可视化结果
│   └── dynamic/
│       ├── index.html
│       ├── quant_depth_*.png
│       ├── quant_pointcloud_*.png
│       └── quant_camera_*.png
├── outputs/                           # 推理输出
│   └── test/
│       ├── summary.txt
│       ├── cameras.npz
│       └── depth/
└── experiment_results.tar.gz          # 打包的结果
```

### 性能对比示例

| 方案 | 大小(MB) | 压缩率 | 推理时间(s) | 加速比 | 深度MAE | 状态 |
|------|---------|--------|-----------|--------|---------|------|
| FP32 (原始) | 4793 | 1.0x | 0.717 | 1.0x | 0.000 | ✅ 基准 |
| INT8 Symmetric | ~1200 | ~4.0x | ~0.550 | ~1.3x | <0.003 | ✅ 推荐 |
| INT8 Asymmetric | ~1200 | ~4.0x | ~0.560 | ~1.28x | <0.002 | ✅ 最佳精度 |
| INT4 Group-128 | ~800 | ~6.0x | ~0.630 | ~1.14x | <0.008 | ✅ 最小模型 |
| INT4 Group-64 | ~900 | ~5.3x | ~0.610 | ~1.18x | <0.005 | ✅ 平衡 |

**注意**: 实际性能取决于硬件和输入数据

---

## 🆘 故障排除

### 问题 1: 找不到图像文件

```bash
# 检查数据路径
ls /workspace/data/eth3d/
ls /workspace/data/eth3d/courtyard/images/ | head -5

# 如果路径不对，手动指定正确路径
--image_folder /workspace/data/eth3d/courtyard/images
```

### 问题 2: CUDA Out of Memory

```bash
# 减少图像数量
--max_images 3

# 或使用 CPU（会慢很多）
--device cpu
```

### 问题 3: 模块导入失败

```bash
# 确保在正确目录
cd /workspace/vggt

# 重新验证
python -c "from vggt.quantization import quantize_model_advanced; print('OK')"
```

### 问题 4: 下载速度慢

```bash
# ETH3D 数据集可以手动下载
# 1. 在本地下载: https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z
# 2. 上传到 RunPod
# 3. 手动解压
7z x multi_view_training_dslr_undistorted.7z -o/workspace/data/eth3d
```

---

## ⏱️ 时间和成本估算

| 阶段 | 时间 | 成本 (RTX 4090 @$0.4/h) |
|------|------|-------------------------|
| 环境设置 | 10 分钟 | $0.07 |
| 数据下载 | 15 分钟 | $0.10 |
| 量化对比 | 20 分钟 | $0.13 |
| 保存模型 | 10 分钟 | $0.07 |
| 可视化 | 15 分钟 | $0.10 |
| 测试推理 | 5 分钟 | $0.03 |
| **总计** | **~75 分钟** | **~$0.50** |

---

## ✅ 完成检查清单

- [ ] 环境设置完成
- [ ] ETH3D 数据集下载完成
- [ ] 量化对比实验完成
- [ ] 量化模型保存完成
- [ ] 可视化生成完成
- [ ] 推理测试完成
- [ ] 结果打包完成
- [ ] 结果下载到本地
- [ ] 查看对比报告
- [ ] 查看可视化结果

---

## 📚 相关文档

- **快速命令**: `QUANTIZATION_QUICK_COMMANDS.md`
- **完整指南**: `QUANTIZATION_COMPARISON_GUIDE.md`
- **更新说明**: `UPDATES_AND_FIXES.md`
- **入口文档**: `START_HERE.md`

---

**准备好了吗？** 从阶段 1 开始，按顺序执行所有命令！🚀

---

**文档版本**: 1.0
**最后更新**: 2025-10-13
**维护者**: Your Team
