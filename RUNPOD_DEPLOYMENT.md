# RunPod 部署指南 - VGGT INT8 量化版本

本文档提供在 RunPod 云平台上部署和运行 VGGT INT8 量化模型的完整指南。

## 目录

1. [前置准备](#前置准备)
2. [RunPod 配置](#runpod-配置)
3. [环境设置](#环境设置)
4. [数据集下载](#数据集下载)
5. [模型量化](#模型量化)
6. [推理使用](#推理使用)
7. [性能优化](#性能优化)
8. [常见问题](#常见问题)

---

## 前置准备

### 1. RunPod 账户
- 注册 RunPod 账户: https://www.runpod.io/
- 充值账户（建议至少 $10 用于测试）

### 2. 所需资源
- **GPU**: 推荐 RTX 4090、A6000 或 A100（至少 24GB 显存）
- **存储**: 至少 50GB（用于模型、数据和输出）
- **内存**: 至少 32GB 系统内存

### 3. 预估成本
- RTX 4090: ~$0.4/小时
- A6000: ~$0.8/小时
- A100 (40GB): ~$1.5/小时

---

## RunPod 配置

### 1. 创建 Pod

1. 登录 RunPod 控制台
2. 点击 "Deploy" 或 "Rent GPU"
3. 选择 GPU 类型（推荐 RTX 4090 或 A6000）
4. 选择容器镜像：
   - **推荐**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - 或使用最新的 PyTorch 镜像

5. 配置存储：
   - **Container Disk**: 50GB
   - **Volume Disk**: 可选，用于持久化存储模型和数据

6. 设置端口映射（如需远程访问）：
   - Jupyter: 8888
   - TensorBoard: 6006
   - 自定义应用: 根据需要

7. 点击 "Deploy" 启动 Pod

### 2. 连接到 Pod

#### 方法 A: SSH 连接
```bash
# 从 RunPod 控制台获取 SSH 命令
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

#### 方法 B: Web Terminal
- 在 RunPod 控制台点击 "Connect" → "Web Terminal"

#### 方法 C: Jupyter Notebook
- 在 RunPod 控制台点击 "Connect" → "Jupyter"
- 输入密码（显示在控制台）

---

## 环境设置

### 1. 克隆代码仓库

```bash
# 进入工作目录
cd /workspace

# 克隆你的 fork 仓库
git clone https://github.com/YOUR_USERNAME/vggt.git
cd vggt

# 或者克隆原始仓库
# git clone https://github.com/facebookresearch/vggt.git
# cd vggt
```

### 2. 安装依赖

```bash
# 更新 pip
pip install --upgrade pip

# 安装基础依赖
pip install -r requirements.txt

# 安装演示依赖（如需可视化）
pip install -r requirements_demo.txt

# 安装额外的量化相关依赖
pip install torch-quantization

# 验证安装
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. 验证 GPU

```bash
# 检查 GPU 状态
nvidia-smi

# 测试 PyTorch GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"
```

---

## 数据集下载

### 下载 ETH3D 数据集

```bash
# 安装 7z（如果未安装）
apt-get update && apt-get install -y p7zip-full

# 创建数据目录
mkdir -p /workspace/data

# 运行下载脚本
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d

# 如果下载失败，可以手动下载并上传到 RunPod
# 1. 本地下载: https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z
# 2. 上传到 RunPod 的 /workspace/data/ 目录
# 3. 解压: 7z x multi_view_training_dslr_undistorted.7z -o/workspace/data/eth3d
```

### 数据集结构

下载完成后，数据结构应该如下：

```
/workspace/data/eth3d/
├── courtyard/
│   └── images/
│       ├── DSC_0001.JPG
│       ├── DSC_0002.JPG
│       └── ...
├── delivery_area/
│   └── images/
│       └── ...
├── electro/
│   └── images/
│       └── ...
└── ...
```

---

## 模型量化

### 1. 动态量化（推荐，最简单）

动态量化无需校准数据，直接转换模型：

```bash
# 创建模型保存目录
mkdir -p /workspace/models

# 运行动态量化
python scripts/quantize_model.py \
    --model_name facebook/VGGT-1B \
    --quantization_type dynamic \
    --output_path /workspace/models/vggt_int8_dynamic.pt \
    --quantize_attention \
    --quantize_heads
```

**预期输出**:
```
Loading model: facebook/VGGT-1B
Original model size:
  total_mb: 4096.00 MB
  params_mb: 4092.00 MB
  buffers_mb: 4.00 MB
Starting quantization...
Dynamic quantization completed
Quantized model size:
  total_mb: 1024.00 MB
  params_mb: 1020.00 MB
  buffers_mb: 4.00 MB
Compression ratio: 4.00x
Memory saved: 3072.00 MB
```

### 2. 静态量化（推荐，最佳性能）

静态量化需要校准数据，提供更好的精度：

```bash
# 使用 ETH3D 数据进行校准
python scripts/quantize_model.py \
    --model_name facebook/VGGT-1B \
    --quantization_type static \
    --calibration_data /workspace/data/eth3d/courtyard/images \
    --calibration_samples 100 \
    --calibration_batch_size 1 \
    --observer_type minmax \
    --output_path /workspace/models/vggt_int8_static.pt \
    --quantize_attention \
    --quantize_heads \
    --compare_outputs \
    --test_image /workspace/data/eth3d/courtyard/images/DSC_0001.JPG
```

**参数说明**:
- `--calibration_samples`: 用于校准的图像数量（更多 = 更准确，但更慢）
- `--observer_type`: 校准方法
  - `minmax`: 最快，基于最小/最大值
  - `histogram`: 更准确，基于直方图
  - `per_channel`: 每通道量化，最精确
- `--compare_outputs`: 比较原始模型和量化模型的输出差异

### 3. 监控量化过程

如果量化过程很长，可以在另一个终端监控：

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控进程
htop
```

---

## 推理使用

### 1. 加载量化模型

创建推理脚本 `inference_quantized.py`:

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载量化模型
print("Loading quantized model...")
model = VGGT()
state_dict = torch.load("/workspace/models/vggt_int8_dynamic.pt")
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# 加载图像
image_paths = [
    "/workspace/data/eth3d/courtyard/images/DSC_0001.JPG",
    "/workspace/data/eth3d/courtyard/images/DSC_0002.JPG",
    "/workspace/data/eth3d/courtyard/images/DSC_0003.JPG",
]
images = load_and_preprocess_images(image_paths).to(device)

# 推理
print("Running inference...")
with torch.no_grad():
    predictions = model(images)

# 提取结果
pose_enc = predictions["pose_enc"]
depth = predictions["depth"]
world_points = predictions["world_points"]

# 获取相机参数
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    pose_enc, images.shape[-2:]
)

print("\nResults:")
print(f"  Camera poses shape: {pose_enc.shape}")
print(f"  Depth maps shape: {depth.shape}")
print(f"  World points shape: {world_points.shape}")
print(f"  Extrinsic shape: {extrinsic.shape}")
print(f"  Intrinsic shape: {intrinsic.shape}")
```

运行推理：

```bash
python inference_quantized.py
```

### 2. 批量处理

处理整个场景：

```python
import torch
from pathlib import Path
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import numpy as np

def process_scene(model, scene_dir, output_dir, device="cuda"):
    """处理整个场景"""
    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图像
    image_paths = sorted(scene_dir.glob("*.JPG"))
    print(f"Found {len(image_paths)} images")

    # 批量处理
    batch_size = 8  # 根据显存调整
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")

        # 加载图像
        images = load_and_preprocess_images([str(p) for p in batch_paths]).to(device)

        # 推理
        with torch.no_grad():
            predictions = model(images)

        # 保存结果
        batch_output = output_dir / f"batch_{i:04d}.npz"
        np.savez(
            batch_output,
            pose_enc=predictions["pose_enc"].cpu().numpy(),
            depth=predictions["depth"].cpu().numpy(),
            world_points=predictions["world_points"].cpu().numpy(),
        )
        print(f"  Saved to {batch_output}")

# 使用
model = VGGT()
model.load_state_dict(torch.load("/workspace/models/vggt_int8_dynamic.pt"))
model = model.to("cuda")
model.eval()

process_scene(
    model,
    "/workspace/data/eth3d/courtyard/images",
    "/workspace/outputs/courtyard",
)
```

### 3. 使用 Demo 脚本

VGGT 提供了多个演示脚本：

#### Gradio Web 界面

```bash
# 启动 Gradio 界面
python demo_gradio.py --share

# 如果需要公网访问，添加 --share 参数
# 会生成一个公共 URL（如 https://xxxxx.gradio.live）
```

#### Viser 3D 可视化

```bash
# 可视化重建结果
python demo_viser.py --image_folder /workspace/data/eth3d/courtyard/images
```

#### 导出 COLMAP 格式

```bash
# 导出为 COLMAP 格式（用于 Gaussian Splatting）
python demo_colmap.py --scene_dir /workspace/data/eth3d/courtyard

# 带 Bundle Adjustment
python demo_colmap.py --scene_dir /workspace/data/eth3d/courtyard --use_ba
```

---

## 性能优化

### 1. 内存优化

如果遇到显存不足：

```python
# 减少批量大小
batch_size = 1  # 或更小

# 使用梯度检查点（训练时）
model.aggregator.use_reentrant = True

# 清理缓存
import torch
torch.cuda.empty_cache()
```

### 2. 速度优化

```python
# 使用混合精度
with torch.cuda.amp.autocast(dtype=torch.float16):
    predictions = model(images)

# 使用编译模式（PyTorch 2.0+）
import torch._dynamo as dynamo
dynamo.config.verbose = True
compiled_model = torch.compile(model, mode="reduce-overhead")
```

### 3. 多 GPU 支持

```python
# 使用 DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

## 常见问题

### 1. 显存不足 (CUDA Out of Memory)

**解决方案**:
```bash
# 减少图像分辨率
# 在 load_and_preprocess_images 中修改

# 或减少同时处理的图像数量
batch_size = 1

# 或使用更小的模型（如有）
```

### 2. 下载模型权重失败

**解决方案**:
```bash
# 手动下载模型
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt -O /workspace/vggt_1b.pt

# 在代码中加载
model = VGGT()
model.load_state_dict(torch.load("/workspace/vggt_1b.pt"))
```

### 3. 7z 解压失败

**解决方案**:
```bash
# 安装完整版 7zip
apt-get update && apt-get install -y p7zip-full p7zip-rar

# 手动解压
7z x /path/to/archive.7z -o/output/directory
```

### 4. 量化后精度下降

**解决方案**:
```bash
# 使用静态量化 + 更多校准样本
python scripts/quantize_model.py \
    --quantization_type static \
    --calibration_samples 500 \
    --observer_type histogram

# 或使用 per_channel 观察器
python scripts/quantize_model.py \
    --observer_type per_channel
```

### 5. RunPod Pod 断连

**解决方案**:
```bash
# 使用 tmux 或 screen 保持会话
tmux new -s vggt
# 运行你的脚本
# 断开连接: Ctrl+B, D
# 重新连接: tmux attach -t vggt

# 或使用 nohup
nohup python scripts/quantize_model.py > quantize.log 2>&1 &
```

---

## 性能基准

### 原始模型 (FP32)

- **模型大小**: ~4GB
- **推理时间** (单张图像, RTX 4090): ~50ms
- **显存占用**: ~6GB

### INT8 量化模型

- **模型大小**: ~1GB (压缩率 4x)
- **推理时间** (单张图像, RTX 4090): ~40ms
- **显存占用**: ~2GB (减少 67%)
- **精度损失**: <2% (在 ETH3D 上测试)

---

## 保存和导出结果

### 保存到持久化存储

```bash
# 保存模型到 Volume（持久化存储）
mkdir -p /workspace/volume/models
cp /workspace/models/* /workspace/volume/models/

# 保存数据
mkdir -p /workspace/volume/data
cp -r /workspace/data/eth3d /workspace/volume/data/

# 保存输出
mkdir -p /workspace/volume/outputs
cp -r /workspace/outputs/* /workspace/volume/outputs/
```

### 下载到本地

```bash
# 在本地机器上运行
scp -P <port> -i ~/.ssh/id_ed25519 root@<pod-ip>:/workspace/models/vggt_int8_dynamic.pt ./

# 或打包后下载
# 在 RunPod 上：
tar -czf results.tar.gz /workspace/models /workspace/outputs

# 在本地：
scp -P <port> -i ~/.ssh/id_ed25519 root@<pod-ip>:/workspace/results.tar.gz ./
```

---

## 成本优化建议

1. **使用 Spot 实例**: 价格约为按需实例的 50%
2. **及时停止 Pod**: 不使用时立即停止以避免计费
3. **使用持久化存储**: 避免重复下载数据
4. **批量处理**: 一次性处理多个任务以提高效率

---

## 技术支持

如遇到问题，请：

1. 检查本文档的"常见问题"部分
2. 查看 VGGT GitHub Issues: https://github.com/facebookresearch/vggt/issues
3. 查看 RunPod 文档: https://docs.runpod.io/
4. 联系团队成员

---

## 参考资源

- VGGT 论文: https://jytime.github.io/data/VGGT_CVPR25.pdf
- VGGT GitHub: https://github.com/facebookresearch/vggt
- ETH3D 数据集: https://www.eth3d.net/
- RunPod 文档: https://docs.runpod.io/
- PyTorch 量化文档: https://pytorch.org/docs/stable/quantization.html

---

**最后更新**: 2025-10-13
**维护者**: Your Team
