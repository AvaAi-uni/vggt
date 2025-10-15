# RunPod 完整量化实验指南

**版本**: 2.0 - Comprehensive Framework
**平台**: RunPod.io
**日期**: 2025-10-16

---

## 📋 目录

1. [RunPod准备](#runpod准备)
2. [上传项目文件](#上传项目文件)
3. [环境设置](#环境设置)
4. [运行实验](#运行实验)
5. [下载结果](#下载结果)
6. [故障排查](#故障排查)
7. [成本优化](#成本优化)

---

## 🚀 快速开始（5步完成）

### 总览

```
步骤1: 创建RunPod实例 (5分钟)
  ↓
步骤2: 上传项目文件 (5-10分钟)
  ↓
步骤3: 运行设置脚本 (3-5分钟)
  ↓
步骤4: 运行量化实验 (5-15分钟)
  ↓
步骤5: 下载结果 (2分钟)

总时间: 20-40分钟
```

---

## 🎯 RunPod准备

### 步骤1: 注册RunPod账号

1. 访问 https://runpod.io
2. 注册账号
3. 充值余额（推荐$10起）

### 步骤2: 选择GPU实例

**推荐配置**:

| GPU | 显存 | 价格 | 推荐场景 |
|-----|------|------|----------|
| **RTX 4090** | 24GB | $0.39/hr | ⭐ 推荐（性价比最高） |
| RTX A6000 | 48GB | $0.79/hr | 大模型/大批量 |
| RTX 3090 | 24GB | $0.44/hr | 备选 |
| RTX A5000 | 24GB | $0.64/hr | 稳定性优先 |

**本实验推荐**: RTX 4090（24GB显存足够，价格最低）

### 步骤3: 创建Pod

1. 点击 "Deploy" → "GPU Pods"
2. 选择GPU类型: **RTX 4090**
3. 选择模板: **PyTorch** 或 **RunPod PyTorch**
4. 配置:
   - **Container Disk**: 50GB（最小）
   - **Volume Disk**: 50GB（持久化存储）
   - **Expose HTTP Port**: 可选
   - **Expose TCP Port**: 可选

5. 点击 "Deploy On-Demand"

**预计费用**:
- 设置 + 测试: ~0.5小时 = $0.20
- 标准实验: ~1小时 = $0.40
- 完整实验: ~2小时 = $0.80

### 步骤4: 连接到Pod

方法1: **Web Terminal（推荐新手）**
```
在RunPod界面点击 "Connect" → "Start Web Terminal"
```

方法2: **SSH（推荐高级用户）**
```bash
# 在本地终端
ssh root@<POD_IP> -p <POD_PORT> -i ~/.ssh/id_ed25519
```

方法3: **Jupyter（可视化）**
```
点击 "Connect" → "Connect to Jupyter Lab"
```

---

## 📦 上传项目文件

### 方法1: 使用Git（推荐）

**如果你的代码在GitHub/GitLab上**:

```bash
# 在RunPod终端
cd /workspace
git clone <YOUR_REPO_URL> vggt
cd vggt
```

**示例**:
```bash
cd /workspace
git clone https://github.com/yourusername/vggt.git vggt
cd vggt
```

---

### 方法2: 使用RunPod文件上传

**适合小文件（<100MB）**:

1. 在RunPod界面打开 "File Browser"
2. 导航到 `/workspace`
3. 创建文件夹 `vggt`
4. 上传项目文件

---

### 方法3: 使用SCP（推荐大文件）

**在本地电脑上**:

```bash
# Windows（使用Git Bash或WSL）
cd C:\Users\Ava Ai\Desktop\8539Project\code
scp -r -P <POD_SSH_PORT> ./vggt root@<POD_IP>:/workspace/

# Linux/Mac
cd ~/projects/8539Project/code
scp -r -P <POD_SSH_PORT> ./vggt root@<POD_IP>:/workspace/
```

**参数说明**:
- `<POD_SSH_PORT>`: RunPod显示的SSH端口
- `<POD_IP>`: RunPod显示的IP地址

**示例**:
```bash
scp -r -P 12345 ./vggt root@123.45.67.89:/workspace/
```

---

### 方法4: 使用rsync（最快，推荐大文件）

```bash
# Linux/Mac/WSL
rsync -avz -e "ssh -p <POD_SSH_PORT>" \
    ./vggt/ \
    root@<POD_IP>:/workspace/vggt/

# 示例
rsync -avz -e "ssh -p 12345" \
    ./vggt/ \
    root@123.45.67.89:/workspace/vggt/
```

---

### 验证上传

```bash
# 在RunPod终端
cd /workspace/vggt
ls -la

# 应该看到:
# vggt/
# ├── scripts/
# │   ├── comprehensive_evaluation.py
# │   ├── runpod_setup_comprehensive.sh
# │   └── ...
# ├── vggt/
# │   ├── quantization/
# │   └── ...
# ├── START_HERE_COMPREHENSIVE.md
# ├── COMPREHENSIVE_QUANTIZATION_GUIDE.md
# └── ...
```

---

## ⚙️ 环境设置

### 一键设置（推荐）

```bash
# 在RunPod终端
cd /workspace/vggt
bash scripts/runpod_setup_comprehensive.sh
```

**脚本会自动**:
- ✅ 检查CUDA和Python环境
- ✅ 创建目录结构
- ✅ 安装所有依赖
- ✅ 验证PyTorch和CUDA
- ✅ 下载测试数据（可选）
- ✅ 创建快捷命令

**预计时间**: 3-5分钟

---

### 手动设置（备用）

如果自动设置失败，可以手动执行：

```bash
# 1. 创建目录
mkdir -p /workspace/data
mkdir -p /workspace/results
mkdir -p /workspace/models

# 2. 安装依赖
cd /workspace/vggt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib Pillow scipy tqdm huggingface_hub

# 3. 验证环境
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 4. 下载测试数据（可选）
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d
```

---

## 🔬 运行实验

### 方法1: 使用快捷命令（推荐）

设置完成后，会自动创建3个快捷命令：

#### 快速测试（5张图像，5-10分钟）

```bash
bash /workspace/run_quick_test.sh
```

**输出**: `/workspace/results/quick_test/`

---

#### 标准测试（10张图像，10-15分钟）

```bash
bash /workspace/run_standard_test.sh
```

**输出**: `/workspace/results/standard_test/`

---

#### 完整测试（50张图像，30-60分钟）

```bash
bash /workspace/run_full_test.sh
```

**输出**: `/workspace/results/full_test/`

---

### 方法2: 手动运行

```bash
cd /workspace/vggt

# 基础命令
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 10 \
    --output_dir /workspace/results/my_experiment \
    --device cuda

# 使用自己的图像
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/my_images \
    --max_images 10 \
    --output_dir /workspace/results/my_experiment \
    --device cuda
```

---

### 方法3: 后台运行（推荐长时间实验）

```bash
# 使用nohup后台运行
cd /workspace/vggt
nohup python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 50 \
    --output_dir /workspace/results/full_test \
    --device cuda \
    > /workspace/results/run.log 2>&1 &

# 查看进程
ps aux | grep comprehensive_evaluation

# 查看日志
tail -f /workspace/results/run.log

# 停止运行（如需要）
pkill -f comprehensive_evaluation
```

---

## 📊 查看结果

### 在RunPod终端查看

```bash
# 查看文本报告
cat /workspace/results/quick_test/comprehensive_report.txt

# 查看JSON数据
python -m json.tool /workspace/results/quick_test/comprehensive_results.json | less

# 列出所有输出文件
ls -lh /workspace/results/quick_test/
```

### 预览图表（Jupyter Lab）

如果使用Jupyter Lab:

```python
# 在Jupyter Notebook中
from IPython.display import Image
Image('/workspace/results/quick_test/comprehensive_visualizations.png')
```

---

## 💾 下载结果

### 方法1: 通过RunPod界面

1. 打开 "File Browser"
2. 导航到 `/workspace/results/`
3. 右键点击文件夹
4. 选择 "Download"

---

### 方法2: 使用SCP（推荐）

**在本地电脑上**:

```bash
# Windows (Git Bash/WSL)
scp -r -P <POD_SSH_PORT> \
    root@<POD_IP>:/workspace/results/quick_test \
    C:/Users/Ava\ Ai/Desktop/results/

# Linux/Mac
scp -r -P <POD_SSH_PORT> \
    root@<POD_IP>:/workspace/results/quick_test \
    ~/Desktop/results/

# 示例
scp -r -P 12345 \
    root@123.45.67.89:/workspace/results/quick_test \
    ~/Desktop/results/
```

---

### 方法3: 使用rsync（最快）

```bash
# Linux/Mac/WSL
rsync -avz -e "ssh -p <POD_SSH_PORT>" \
    root@<POD_IP>:/workspace/results/ \
    ~/Desktop/results/

# 示例
rsync -avz -e "ssh -p 12345" \
    root@123.45.67.89:/workspace/results/ \
    ~/Desktop/results/
```

---

### 方法4: 使用Jupyter Lab

1. 在Jupyter Lab中打开 Terminal
2. 压缩结果:
```bash
cd /workspace
tar -czf results.tar.gz results/
```
3. 在File Browser中下载 `results.tar.gz`
4. 在本地解压:
```bash
tar -xzf results.tar.gz
```

---

## 🔍 故障排查

### 问题1: CUDA out of memory

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:

```bash
# 方案1: 减少图像数量
python scripts/comprehensive_evaluation.py \
    --max_images 3 \
    --output_dir /workspace/results/small_test

# 方案2: 使用CPU（慢）
python scripts/comprehensive_evaluation.py \
    --device cpu \
    --max_images 5 \
    --output_dir /workspace/results/cpu_test

# 方案3: 清理GPU缓存
python << EOF
import torch
torch.cuda.empty_cache()
print("GPU缓存已清理")
EOF
```

---

### 问题2: 找不到测试图像

**症状**:
```
ERROR: No images found in /workspace/data/eth3d/courtyard/images
```

**解决方案**:

```bash
# 方案1: 检查图像路径
find /workspace/data -name "*.jpg" -o -name "*.png" | head

# 方案2: 重新下载数据
cd /workspace/vggt
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d

# 方案3: 使用找到的图像路径
python scripts/comprehensive_evaluation.py \
    --image_folder <FOUND_PATH> \
    --max_images 5
```

---

### 问题3: 依赖缺失

**症状**:
```
ModuleNotFoundError: No module named 'XXX'
```

**解决方案**:

```bash
# 重新安装依赖
pip install torch torchvision torchaudio numpy matplotlib Pillow scipy tqdm

# 或重新运行设置脚本
cd /workspace/vggt
bash scripts/runpod_setup_comprehensive.sh
```

---

### 问题4: 连接断开

**症状**:
Pod连接断开，实验中断

**解决方案**:

```bash
# 使用tmux保持会话
tmux new -s experiment

# 在tmux中运行实验
cd /workspace/vggt
python scripts/comprehensive_evaluation.py ...

# 分离tmux: Ctrl+B, 然后按 D

# 重新连接后，恢复会话
tmux attach -t experiment
```

---

### 问题5: 磁盘空间不足

**症状**:
```
OSError: [Errno 28] No space left on device
```

**解决方案**:

```bash
# 检查磁盘使用
df -h /workspace

# 清理不需要的文件
rm -rf /workspace/vggt/__pycache__
rm -rf /workspace/vggt/.git
rm -rf /tmp/*

# 只下载部分数据
python scripts/download_eth3d.py \
    --output_dir /workspace/data/eth3d \
    --max_scenes 1
```

---

## 💰 成本优化

### 1. 选择合适的GPU

| 任务 | 推荐GPU | 原因 |
|------|---------|------|
| 快速测试（5张） | RTX 4090 | 最便宜，足够快 |
| 标准实验（10-30张） | RTX 4090 | 性价比最高 |
| 大规模实验（50+张） | RTX A6000 | 显存大，稳定 |

### 2. 及时停止Pod

```bash
# 实验完成后立即停止
# 在RunPod界面点击 "Stop"
```

**重要**: RunPod按小时计费，即使空闲也会收费！

### 3. 使用Spot实例

- 价格: 比On-Demand便宜50-80%
- 风险: 可能被随时回收
- 适合: 可重新运行的实验

### 4. 批量运行

```bash
# 一次运行多个实验，避免重复设置
python scripts/comprehensive_evaluation.py --max_images 10 --output_dir /workspace/results/exp1
python scripts/comprehensive_evaluation.py --max_images 20 --output_dir /workspace/results/exp2
python scripts/comprehensive_evaluation.py --max_images 30 --output_dir /workspace/results/exp3

# 全部完成后再停止Pod
```

### 5. 使用Volume存储

- 创建Persistent Volume（持久化存储）
- 下载的数据和模型可以跨Pod复用
- 避免重复下载

---

## 📝 完整工作流程示例

### 场景: 运行标准量化实验

```bash
# ============================================================================
# 步骤1: 创建Pod（在RunPod界面）
# ============================================================================
# GPU: RTX 4090
# Container Disk: 50GB
# Volume Disk: 50GB


# ============================================================================
# 步骤2: 连接到Pod
# ============================================================================
# 点击 "Connect" → "Start Web Terminal"


# ============================================================================
# 步骤3: 上传项目（选择一种方法）
# ============================================================================

# 方法A: Git（如果代码在GitHub）
cd /workspace
git clone https://github.com/yourusername/vggt.git vggt
cd vggt

# 方法B: SCP（在本地电脑运行）
# scp -r -P 12345 ./vggt root@123.45.67.89:/workspace/


# ============================================================================
# 步骤4: 运行设置脚本
# ============================================================================
cd /workspace/vggt
bash scripts/runpod_setup_comprehensive.sh

# 选择下载ETH3D数据: y


# ============================================================================
# 步骤5: 运行实验
# ============================================================================

# 快速测试（5分钟）
bash /workspace/run_quick_test.sh

# 查看进度
# (等待5-10分钟)


# ============================================================================
# 步骤6: 查看结果
# ============================================================================
cat /workspace/results/quick_test/comprehensive_report.txt


# ============================================================================
# 步骤7: 下载结果（在本地电脑运行）
# ============================================================================
# scp -r -P 12345 \
#     root@123.45.67.89:/workspace/results/quick_test \
#     ~/Desktop/results/


# ============================================================================
# 步骤8: 停止Pod（在RunPod界面）
# ============================================================================
# 点击 "Stop" 停止计费
```

**总时间**: ~20分钟
**总费用**: ~$0.13 (RTX 4090 @ $0.39/hr)

---

## 🎓 高级技巧

### 技巧1: 使用tmux保持会话

```bash
# 创建新会话
tmux new -s quantization

# 在会话中运行实验
cd /workspace/vggt
bash /workspace/run_standard_test.sh

# 分离会话: Ctrl+B, 然后 D
# 此时可以关闭浏览器，实验继续运行

# 重新连接后
tmux attach -t quantization
```

---

### 技巧2: 监控GPU使用

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或使用gpustat
pip install gpustat
gpustat -i 1
```

---

### 技巧3: 并行运行多个实验

```bash
# 使用不同的输出目录
python scripts/comprehensive_evaluation.py \
    --max_images 10 \
    --output_dir /workspace/results/exp1 &

python scripts/comprehensive_evaluation.py \
    --max_images 20 \
    --output_dir /workspace/results/exp2 &

# 等待所有任务完成
wait
```

---

### 技巧4: 自动化实验

创建 `/workspace/run_all_experiments.sh`:

```bash
#!/bin/bash
# 运行所有实验

cd /workspace/vggt

# 实验1: 快速测试
echo "运行实验1: 快速测试..."
python scripts/comprehensive_evaluation.py \
    --max_images 5 \
    --output_dir /workspace/results/exp1_quick

# 实验2: 标准测试
echo "运行实验2: 标准测试..."
python scripts/comprehensive_evaluation.py \
    --max_images 10 \
    --output_dir /workspace/results/exp2_standard

# 实验3: 大规模测试
echo "运行实验3: 大规模测试..."
python scripts/comprehensive_evaluation.py \
    --max_images 30 \
    --output_dir /workspace/results/exp3_large

echo "所有实验完成！"
```

运行:
```bash
chmod +x /workspace/run_all_experiments.sh
bash /workspace/run_all_experiments.sh
```

---

## 📊 预期结果

运行完成后，你会得到：

```
/workspace/results/quick_test/
├── comprehensive_results.json          # JSON格式完整数据
├── comprehensive_report.txt            # 文本格式报告
└── comprehensive_visualizations.png    # 可视化图表
```

**文本报告示例**:

```
================================================================================
完整量化评估报告
================================================================================

生成时间: 2025-10-16 10:30:00
模型: facebook/VGGT-1B
测试图像数: 10
测试配置数: 8

================================================================================
实验结果概览
================================================================================

方案                         | 大小(MB) | 压缩率 | 时间(s) | 加速 | MAE      | CE
--------------------------------------------------------------------------------------------
Baseline_FP32                | 4000.00  | 1.00x  | 0.0500  | 1.00x| 0.000000 | 0.000000
INT8_Per_Channel_Symmetric   | 1010.00  | 3.96x  | 0.0385  | 1.30x| 0.000523 | 0.001023 ⭐
INT4_Group_128               |  500.00  | 8.00x  | 0.0350  | 1.43x| 0.007891 | 0.015234

================================================================================
实验总结
================================================================================

最高压缩率: INT4_Group_128 (8.00x)
最快推理: INT4_Group_128 (1.43x)
最高精度: INT8_Per_Channel_Asymmetric (MAE: 0.000498)
```

---

## 🔗 快速参考

### 常用命令

```bash
# 连接到Pod
ssh root@<POD_IP> -p <POD_PORT>

# 运行设置
cd /workspace/vggt && bash scripts/runpod_setup_comprehensive.sh

# 快速测试
bash /workspace/run_quick_test.sh

# 查看结果
cat /workspace/results/quick_test/comprehensive_report.txt

# 下载结果（本地）
scp -r -P <PORT> root@<IP>:/workspace/results ~/Desktop/

# 停止Pod
# 在RunPod界面点击 "Stop"
```

### 目录结构

```
/workspace/
├── vggt/                          # 项目目录
│   ├── scripts/
│   ├── vggt/
│   └── *.md
├── data/                          # 数据目录
│   └── eth3d/
├── results/                       # 结果目录
│   ├── quick_test/
│   ├── standard_test/
│   └── full_test/
├── models/                        # 模型目录
├── run_quick_test.sh             # 快捷命令
├── run_standard_test.sh
└── run_full_test.sh
```

---

## ✅ 检查清单

使用前检查：

- [ ] 创建了RunPod Pod
- [ ] 选择了合适的GPU（推荐RTX 4090）
- [ ] 上传了项目文件到 `/workspace/vggt`
- [ ] 运行了设置脚本
- [ ] 下载了测试数据或准备了自己的图像
- [ ] 验证了CUDA可用

实验后检查：

- [ ] 查看了文本报告
- [ ] 下载了所有结果文件
- [ ] 停止了Pod（避免持续计费）

---

## 🎉 总结

使用RunPod运行完整量化实验：

✅ **简单**: 一键设置脚本，快捷命令
✅ **快速**: RTX 4090，10分钟完成标准测试
✅ **便宜**: $0.39/小时，标准实验<$0.50
✅ **完整**: 8种方案，8种指标
✅ **专业**: 工业标准，可用于论文

**立即开始**:
```bash
cd /workspace/vggt
bash scripts/runpod_setup_comprehensive.sh
bash /workspace/run_quick_test.sh
```

**10分钟后**，你将拥有完整的量化实验结果！🚀

---

**需要帮助？** 查看其他文档：
- `START_HERE_COMPREHENSIVE.md` - 总体概览
- `COMPREHENSIVE_QUANTIZATION_GUIDE.md` - 详细指南
- `EXPERIMENT_PARAMETERS_EXPLAINED.md` - 参数说明
