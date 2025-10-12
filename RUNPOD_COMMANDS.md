# RunPod 完整操作指令

本文档包含在 RunPod 上部署和运行 VGGT INT8 量化的**所有**命令，按顺序执行即可。

---

## 📋 前置准备

### 1. 创建 RunPod Pod

在 RunPod 网站上：
- 选择 GPU: **RTX 4090** (24GB) 或 **A6000** (48GB)
- Container Disk: **50GB**
- 镜像: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- 点击 **Deploy**

### 2. 连接到 Pod

等待 Pod 启动完成，然后点击 **Connect** → **Start Web Terminal**

---

## 🚀 第一步：环境准备

### 1.1 进入工作目录

```bash
cd /workspace
```

### 1.2 克隆代码仓库

```bash
# 如果你已经 fork 了仓库，使用你自己的 URL
git clone https://github.com/YOUR_USERNAME/vggt.git
cd vggt

# 如果没有 fork，使用原始仓库
# git clone https://github.com/facebookresearch/vggt.git
# cd vggt
```

### 1.3 验证当前 PyTorch 版本

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

---

## 🔧 第二步：修复依赖问题

### 2.1 卸载旧版本 torchaudio

```bash
pip uninstall torchaudio -y
```

### 2.2 安装正确版本的依赖

```bash
# 方法 A: 安装完整的 PyTorch 套件（推荐）
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# 方法 B: 或者使用 requirements.txt
pip install -r requirements.txt
```

### 2.3 安装其他依赖

```bash
pip install -r requirements_demo.txt
```

### 2.4 验证安装

```bash
python -c "import torch, torchvision, torchaudio; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

应该显示：
```
torch: 2.3.1
torchvision: 0.18.1
torchaudio: 2.3.1
CUDA available: True
```

---

## 📦 第三步：下载 ETH3D 数据集（可选但推荐）

### 3.1 安装 7z 工具

```bash
apt-get update && apt-get install -y p7zip-full
```

### 3.2 下载数据集

```bash
# 自动下载和解压（需要 10-15 分钟）
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d
```

**注意**：如果下载速度太慢，可以手动下载：

```bash
# 手动下载方法
wget https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z -O /workspace/eth3d.7z

# 解压
7z x /workspace/eth3d.7z -o/workspace/data/eth3d

# 删除压缩包（节省空间）
rm /workspace/eth3d.7z
```

### 3.3 验证数据集

```bash
# 查看下载的场景
ls /workspace/data/eth3d/

# 查看某个场景的图像数量
ls /workspace/data/eth3d/courtyard/images/ | wc -l
```

---

## 🔄 第四步：量化模型

### 4.1 创建模型保存目录

```bash
mkdir -p /workspace/models
```

### 4.2 方法 A：动态量化（最简单，推荐入门）

```bash
python scripts/quantize_model.py \
    --model_name facebook/VGGT-1B \
    --quantization_type dynamic \
    --output_path /workspace/models/vggt_int8_dynamic.pt \
    --quantize_attention \
    --quantize_heads
```

**预期时间**: 5-10 分钟（首次运行会下载模型，约 4GB）

### 4.3 方法 B：静态量化（精度更高，推荐生产）

**前提**：必须先完成第三步（下载 ETH3D 数据集）

```bash
python scripts/quantize_model.py \
    --model_name facebook/VGGT-1B \
    --quantization_type static \
    --calibration_data /workspace/data/eth3d/courtyard/images \
    --calibration_samples 100 \
    --calibration_batch_size 1 \
    --observer_type minmax \
    --output_path /workspace/models/vggt_int8_static.pt \
    --quantize_attention \
    --quantize_heads
```

**预期时间**: 15-20 分钟

### 4.4 方法 C：静态量化 + 精度对比

```bash
python scripts/quantize_model.py \
    --model_name facebook/VGGT-1B \
    --quantization_type static \
    --calibration_data /workspace/data/eth3d/courtyard/images \
    --calibration_samples 100 \
    --observer_type minmax \
    --output_path /workspace/models/vggt_int8_static_compared.pt \
    --quantize_attention \
    --quantize_heads \
    --compare_outputs \
    --test_image /workspace/data/eth3d/courtyard/images/DSC_0001.JPG
```

### 4.5 验证量化模型

```bash
# 查看生成的文件
ls -lh /workspace/models/

# 查看配置文件
cat /workspace/models/vggt_int8_dynamic_config.txt
```

---

## 🎯 第五步：运行推理

### 5.1 创建输出目录

```bash
mkdir -p /workspace/outputs
```

### 5.2 使用量化模型推理

```bash
# 使用动态量化模型
python scripts/inference_quantized.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/outputs/courtyard_dynamic \
    --max_images 10

# 如果使用静态量化模型
python scripts/inference_quantized.py \
    --model_path /workspace/models/vggt_int8_static.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/outputs/courtyard_static \
    --max_images 10
```

### 5.3 查看推理结果

```bash
# 查看输出目录结构
tree /workspace/outputs/courtyard_dynamic -L 2

# 或使用 ls
ls -lh /workspace/outputs/courtyard_dynamic/

# 查看摘要
cat /workspace/outputs/courtyard_dynamic/summary.txt
```

---

## 🎨 第六步：可视化（可选）

### 6.1 使用 Gradio Web 界面

```bash
python demo_gradio.py --share
```

访问输出的 URL（如 `https://xxxxx.gradio.live`）

### 6.2 使用 Viser 3D 可视化

```bash
python demo_viser.py --image_folder /workspace/data/eth3d/courtyard/images
```

### 6.3 导出为 COLMAP 格式

```bash
# 基础导出
python demo_colmap.py --scene_dir /workspace/data/eth3d/courtyard

# 带 Bundle Adjustment（更准确但更慢）
python demo_colmap.py --scene_dir /workspace/data/eth3d/courtyard --use_ba
```

结果保存在 `/workspace/data/eth3d/courtyard/sparse/`

---

## 📊 第七步：性能测试（可选）

### 7.1 监控 GPU 使用

在另一个终端窗口运行：

```bash
watch -n 1 nvidia-smi
```

### 7.2 测试推理速度

```bash
# 测试脚本
python -c "
import torch
import time
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 加载模型
model = VGGT()
model.load_state_dict(torch.load('/workspace/models/vggt_int8_dynamic.pt'))
model = model.cuda()
model.eval()

# 测试图像
images = load_and_preprocess_images([
    '/workspace/data/eth3d/courtyard/images/DSC_0001.JPG',
    '/workspace/data/eth3d/courtyard/images/DSC_0002.JPG',
]).cuda()

# 预热
with torch.no_grad():
    _ = model(images)

# 计时
start = time.time()
for _ in range(10):
    with torch.no_grad():
        _ = model(images)
torch.cuda.synchronize()
end = time.time()

print(f'Average time per inference: {(end-start)/10:.3f}s')
print(f'FPS: {10/(end-start):.2f}')
"
```

---

## 💾 第八步：保存结果

### 8.1 打包输出

```bash
# 打包所有结果
cd /workspace
tar -czf vggt_results.tar.gz models/ outputs/

# 查看文件大小
ls -lh vggt_results.tar.gz
```

### 8.2 下载到本地

在**本地电脑**上运行：

```bash
# 获取 Pod 的 SSH 信息（从 RunPod 控制台）
# 然后下载文件
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<POD_IP>:/workspace/vggt_results.tar.gz ./
```

---

## 🛠️ 故障排除

### 问题 1: torchaudio 版本冲突

```bash
# 解决方案
pip uninstall torchaudio torch torchvision -y
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

### 问题 2: CUDA Out of Memory

```bash
# 解决方案：减少图像数量
python scripts/inference_quantized.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/outputs/courtyard \
    --max_images 5  # 减少到 5 张
```

### 问题 3: 下载模型失败

```bash
# 手动下载模型
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt -O /workspace/vggt_1b_original.pt

# 然后在 Python 中加载
python -c "
from vggt.models.vggt import VGGT
import torch
model = VGGT()
model.load_state_dict(torch.load('/workspace/vggt_1b_original.pt'))
"
```

### 问题 4: 7z 解压失败

```bash
# 重新安装 7z
apt-get update
apt-get install -y p7zip-full p7zip-rar

# 再次尝试解压
7z x /workspace/eth3d.7z -o/workspace/data/eth3d
```

### 问题 5: 导入错误

```bash
# 确保在正确的目录
cd /workspace/vggt

# 验证文件存在
ls vggt/quantization/

# 重新测试导入
python -c "from vggt.quantization import quantize_model; print('Import successful!')"
```

---

## 🔄 完整流程（一键复制）

如果你想一次性运行所有命令（不包括可选步骤）：

```bash
# ==================== 环境设置 ====================
cd /workspace
git clone https://github.com/YOUR_USERNAME/vggt.git
cd vggt

# 修复依赖
pip uninstall torchaudio -y
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_demo.txt

# ==================== 下载数据 ====================
apt-get update && apt-get install -y p7zip-full
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d

# ==================== 量化模型 ====================
mkdir -p /workspace/models
python scripts/quantize_model.py \
    --model_name facebook/VGGT-1B \
    --quantization_type dynamic \
    --output_path /workspace/models/vggt_int8_dynamic.pt \
    --quantize_attention \
    --quantize_heads

# ==================== 运行推理 ====================
mkdir -p /workspace/outputs
python scripts/inference_quantized.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/outputs/courtyard \
    --max_images 10

# ==================== 查看结果 ====================
cat /workspace/outputs/courtyard/summary.txt
ls -lh /workspace/models/
```

---

## 📌 重要提示

1. **首次运行**会自动下载 VGGT-1B 模型（约 4GB），需要耐心等待
2. **ETH3D 数据集**约 10GB，下载需要 10-15 分钟
3. **量化过程**在 RTX 4090 上约需 5-10 分钟（动态）或 15-20 分钟（静态）
4. **推理速度**：动态量化约 40ms/图像，静态量化约 35ms/图像
5. **显存占用**：量化后约 2GB，原始模型约 6GB

---

## 💰 成本估算

使用 RTX 4090 ($0.4/小时)：

| 任务 | 时间 | 成本 |
|------|------|------|
| 环境设置 | 5 分钟 | $0.03 |
| 下载数据集 | 15 分钟 | $0.10 |
| 动态量化 | 10 分钟 | $0.07 |
| 静态量化 | 20 分钟 | $0.13 |
| 推理测试 | 5 分钟 | $0.03 |
| **总计** | **~1 小时** | **$0.40** |

---

## 📚 相关文档

- [快速开始指南](QUANTIZATION_README.md)
- [详细部署教程](RUNPOD_DEPLOYMENT.md)
- [实现总结](IMPLEMENTATION_SUMMARY.md)

---

## ✅ 检查清单

完成后，你应该有：

- [ ] 量化模型文件: `/workspace/models/vggt_int8_dynamic.pt`
- [ ] ETH3D 数据集: `/workspace/data/eth3d/`
- [ ] 推理结果: `/workspace/outputs/courtyard/`
- [ ] 深度图: `/workspace/outputs/courtyard/depth/`
- [ ] 点云: `/workspace/outputs/courtyard/points/`
- [ ] 相机参数: `/workspace/outputs/courtyard/cameras.npz`

---

**最后更新**: 2025-10-13
**版本**: 1.0
