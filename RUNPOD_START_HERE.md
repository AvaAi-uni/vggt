# RunPod 完整量化实验 - 从这里开始

**⚡ 3步完成RunPod实验！**

**版本**: 3.0 - 纯RunPod工作流
**平台**: RunPod.io
**预计时间**: 15-30分钟
**预计费用**: $0.10 - $0.40

---

## 🎯 你将得到什么

运行完成后，你将拥有：

✅ **完整的Baseline对比** - FP32原始模型性能
✅ **7种量化方案结果** - INT8和INT4多精度
✅ **8种评估指标** - 包括MAE和Cross Entropy
✅ **专业可视化图表** - 6个对比图
✅ **详细实验报告** - 可直接用于论文

---

## 🚀 3步快速开始（纯RunPod操作）

### 步骤1: 创建RunPod实例（5分钟）

1. 访问 https://runpod.io
2. 点击 "Deploy" → "GPU Pods"
3. 选择GPU: **RTX 4090**（推荐，$0.39/小时）
4. 选择模板: "PyTorch"
5. 配置:
   - Container Disk: 50GB
   - Volume Disk: 50GB（持久化）
6. 点击 "Deploy On-Demand"

**费用**: ~$0.15 用于设置和快速测试

---

### 步骤2: 获取项目代码（2分钟）

在RunPod终端运行：

```bash
# 方法A: 从GitHub克隆（推荐）
cd /workspace
git clone https://github.com/yourusername/vggt.git vggt
cd vggt
```

```bash
# 方法B: 从GitLab克隆
cd /workspace
git clone https://gitlab.com/yourusername/vggt.git vggt
cd vggt
```

```bash
# 方法C: 手动上传ZIP到RunPod，然后解压
cd /workspace
unzip vggt.zip
cd vggt
```

**注意**: 如果你的代码在Git仓库中，方法A最快最方便！

---

### 步骤3: 一键运行完整实验（10-30分钟）

在RunPod终端运行：

#### 选项A: 快速测试（推荐首次使用，10-15分钟）

```bash
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh quick
```

#### 选项B: 标准测试（推荐提交作业，15-20分钟）

```bash
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh standard
```

#### 选项C: 完整测试（用于论文，30-60分钟）

```bash
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh full
```

**这一个命令会自动完成所有操作：**
- ✅ 检查CUDA环境
- ✅ 安装所有依赖
- ✅ 自动下载ETH3D数据
- ✅ 运行量化实验
- ✅ 显示结果摘要

**实验会自动运行，你可以看着进度或去喝杯咖啡☕**

---

## ✅ 完成！查看结果

### 4.1 在RunPod终端查看文本报告

```bash
# 查看完整报告
cat /workspace/results/quick_test_*/comprehensive_report.txt

# 查看前50行
head -50 /workspace/results/quick_test_*/comprehensive_report.txt
```

### 4.2 下载结果到本地（可选）

在RunPod终端创建压缩包：

```bash
cd /workspace
tar -czf results.tar.gz results/
```

然后在RunPod界面点击"Files"，下载 `/workspace/results.tar.gz` 文件

**完成后停止Pod避免计费**: 在RunPod界面点击 "Stop"

---

## 📊 你将得到的结果文件

```
/workspace/results/quick_test_20251016_123456/
├── comprehensive_results.json          # 完整数据（JSON格式）
├── comprehensive_report.txt            # 文本报告（复制即用）
└── comprehensive_visualizations.png    # 可视化图表（6个子图）
```

### 文本报告示例

```
================================================================================
完整量化评估报告
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

## 💰 费用估算

| 任务 | 时间 | GPU | 费用 |
|------|------|-----|------|
| 快速测试 | ~15分钟 | RTX 4090 | ~$0.10 |
| 标准测试 | ~20分钟 | RTX 4090 | ~$0.13 |
| 完整测试 | ~60分钟 | RTX 4090 | ~$0.40 |

**总费用**: $0.10 - $0.40

---

## 🎓 超级快速流程（复制粘贴一行）

如果你的代码已在GitHub，可以复制这一行到RunPod终端：

```bash
# ============================================================================
# RunPod 完整量化实验 - 终极一键命令
# ============================================================================

cd /workspace && \
git clone https://github.com/yourusername/vggt.git vggt && \
cd vggt && \
bash scripts/runpod_full_workflow.sh quick
```

**只需要改一个地方**: 把 `yourusername/vggt` 改成你的仓库地址！

**15分钟后，你将拥有完整的实验结果！**

---

## ❓ 常见问题

### Q1: 如何获取项目代码到RunPod？

**A**: 最简单的三种方法：

```bash
# 方法1: Git克隆（推荐）
cd /workspace
git clone https://github.com/yourusername/vggt.git vggt

# 方法2: 从压缩包
# 先在RunPod界面上传vggt.zip，然后：
cd /workspace
unzip vggt.zip

# 方法3: 从网盘链接
cd /workspace
wget "https://your-drive-link.com/vggt.zip"
unzip vggt.zip
```

---

### Q2: CUDA out of memory怎么办？

**A**: 使用更少的图像测试：

```bash
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/dslr_images_undistorted \
    --max_images 3 \
    --output_dir /workspace/results/small_test
```

---

### Q3: 找不到测试图像怎么办？

**A**: 重新下载ETH3D数据：

```bash
cd /workspace/vggt
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d
```

或使用自己的图像（在RunPod上传图像后）：

```bash
# 假设你上传图像到 /workspace/my_images
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/my_images \
    --max_images 10 \
    --output_dir /workspace/results/my_test
```

---

### Q4: 连接断开了怎么办？

**A**: 使用tmux防止断开（在RunPod终端）：

```bash
# 创建tmux会话
tmux new -s quantization

# 在tmux中运行实验
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh standard

# 分离tmux: Ctrl+B, 然后按 D
# 重新连接: tmux attach -t quantization
```

---

### Q5: 如何下载结果到本地？

**A**: 在RunPod终端创建压缩包：

```bash
cd /workspace
tar -czf results.tar.gz results/
```

然后在RunPod Web界面：
1. 点击 "Files"
2. 找到 `/workspace/results.tar.gz`
3. 点击下载

---

## 📚 进阶使用（所有在RunPod终端运行）

### 自定义实验参数

```bash
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/dslr_images_undistorted \
    --max_images 20 \
    --output_dir /workspace/results/custom_test \
    --device cuda
```

### 后台运行长时间实验

```bash
cd /workspace/vggt
nohup bash scripts/runpod_full_workflow.sh full > /workspace/full_test.log 2>&1 &

# 查看日志
tail -f /workspace/full_test.log

# 查看进程
ps aux | grep runpod_full_workflow
```

### 批量实验

```bash
# 运行多个不同场景
cd /workspace/vggt

for scene in courtyard delivery_area facade; do
    python scripts/comprehensive_evaluation.py \
        --image_folder /workspace/data/eth3d/$scene/dslr_images_undistorted \
        --max_images 10 \
        --output_dir /workspace/results/${scene}_test
done
```

### 使用tmux运行长时间实验

```bash
# 创建tmux会话
tmux new -s full_experiment

# 在tmux中运行完整测试
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh full

# 按 Ctrl+B, 然后按 D 分离tmux
# 稍后重新连接: tmux attach -t full_experiment
```

---

## 📖 文档导航

### RunPod相关（推荐阅读顺序）

1. **[RUNPOD_START_HERE.md](RUNPOD_START_HERE.md)** ← 你在这里（快速开始，5分钟）
2. **[RUNPOD_QUICK_COMMANDS.md](RUNPOD_QUICK_COMMANDS.md)** - 快速命令参考（查找命令时使用）
3. **[RUNPOD_COMPREHENSIVE_GUIDE.md](RUNPOD_COMPREHENSIVE_GUIDE.md)** - 完整RunPod指南（深入学习）

### 实验相关

4. **[START_HERE_COMPREHENSIVE.md](START_HERE_COMPREHENSIVE.md)** - 项目总览
5. **[COMPREHENSIVE_QUANTIZATION_GUIDE.md](COMPREHENSIVE_QUANTIZATION_GUIDE.md)** - 量化原理
6. **[EXPERIMENT_PARAMETERS_EXPLAINED.md](EXPERIMENT_PARAMETERS_EXPLAINED.md)** - 参数详解

---

## 🎉 总结

使用RunPod运行完整量化实验：

| 维度 | 内容 |
|------|------|
| **时间** | 15-30分钟 |
| **费用** | $0.10 - $0.40 |
| **步骤** | 3步完成（纯RunPod） |
| **结果** | 8种方案 + 8种指标 |
| **输出** | JSON + 文本 + 图表 |

---

## ⚡ 立即开始（纯RunPod操作）

### 最简单的开始方式（3步）

**在RunPod终端依次执行：**

```bash
# 步骤1: 获取代码
cd /workspace
git clone https://github.com/yourusername/vggt.git vggt

# 步骤2: 一键运行
cd vggt
bash scripts/runpod_full_workflow.sh quick

# 步骤3: 查看结果
cat /workspace/results/quick_test_*/comprehensive_report.txt
```

**或者复制这一行（超级快速）：**

```bash
cd /workspace && git clone https://github.com/yourusername/vggt.git vggt && cd vggt && bash scripts/runpod_full_workflow.sh quick
```

**15分钟后，你将拥有完整的量化实验结果！** 🎊

### 下载结果

```bash
# 在RunPod终端
cd /workspace
tar -czf results.tar.gz results/

# 然后在RunPod Web界面下载 /workspace/results.tar.gz
```

### 完成后

在RunPod界面点击 "Stop" 停止Pod避免计费

---

## 🔗 快速链接

- **RunPod平台**: https://runpod.io
- **ETH3D数据集**: https://www.eth3d.net
- **问题反馈**: 查看项目GitHub Issues

---

## 💡 核心特点

✅ **纯RunPod工作流** - 无需本地操作
✅ **一键执行** - 单个命令完成所有步骤
✅ **自动化** - 环境设置、数据下载、实验运行全自动
✅ **低成本** - 快速测试仅需 $0.10
✅ **专业输出** - 8种方案完整对比

---

**需要帮助？**

- 查看 [常见问题](#常见问题) - 本页上方
- 查看 [RUNPOD_QUICK_COMMANDS.md](RUNPOD_QUICK_COMMANDS.md) - 命令参考
- 查看 [RUNPOD_COMPREHENSIVE_GUIDE.md](RUNPOD_COMPREHENSIVE_GUIDE.md) - 完整指南

**祝实验顺利！** 🚀
