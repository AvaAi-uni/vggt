# RunPod 完整量化实验 - 从这里开始

**⚡ 5步完成RunPod实验！**

**版本**: 2.0 - Comprehensive Framework
**平台**: RunPod.io
**预计时间**: 20-40分钟
**预计费用**: $0.13 - $0.50

---

## 🎯 你将得到什么

运行完成后，你将拥有：

✅ **完整的Baseline对比** - FP32原始模型性能
✅ **7种量化方案结果** - INT8和INT4多精度
✅ **8种评估指标** - 包括MAE和Cross Entropy
✅ **专业可视化图表** - 6个对比图
✅ **详细实验报告** - 可直接用于论文

---

## 🚀 5步快速开始

### 步骤1: 创建RunPod实例（5分钟）

1. 访问 https://runpod.io
2. 点击 "Deploy" → "GPU Pods"
3. 选择GPU: **RTX 4090**（推荐，$0.39/小时）
4. 选择模板: "PyTorch"
5. 配置:
   - Container Disk: 50GB
   - Volume Disk: 50GB（持久化）
6. 点击 "Deploy On-Demand"

**费用**: ~$0.20 用于设置

---

### 步骤2: 上传项目文件（5-10分钟）

在RunPod终端运行：

```bash
# 方法A: 使用Git（如果代码在GitHub）
cd /workspace
git clone https://github.com/yourusername/vggt.git vggt
cd vggt
```

或在**本地电脑**运行（使用SCP）:

```bash
# Windows (Git Bash)
cd "C:\Users\Ava Ai\Desktop\8539Project\code"
scp -r -P <POD_SSH_PORT> ./vggt root@<POD_IP>:/workspace/

# Linux/Mac
cd ~/projects/code
scp -r -P <POD_SSH_PORT> ./vggt root@<POD_IP>:/workspace/
```

**提示**: Pod的IP和端口在RunPod界面的"Connect"中查看

---

### 步骤3: 一键设置环境（3-5分钟）

在RunPod终端运行：

```bash
cd /workspace/vggt
bash scripts/runpod_setup_comprehensive.sh
```

脚本会自动：
- ✅ 检查CUDA环境
- ✅ 安装所有依赖
- ✅ 下载测试数据（可选）
- ✅ 创建快捷命令

**选择下载ETH3D数据**: 输入 `y` 然后回车

---

### 步骤4: 运行实验（5-15分钟）

选择一个命令运行：

#### 选项A: 快速测试（推荐新手，5-10分钟）

```bash
bash /workspace/run_quick_test.sh
```

#### 选项B: 标准测试（推荐提交，10-15分钟）

```bash
bash /workspace/run_standard_test.sh
```

#### 选项C: 完整测试（发论文，30-60分钟）

```bash
bash /workspace/run_full_test.sh
```

**实验会自动运行，你可以看着进度条或去喝杯咖啡☕**

---

### 步骤5: 下载结果（2分钟）

#### 5.1 先查看结果

在RunPod终端：

```bash
cat /workspace/results/quick_test/comprehensive_report.txt
```

#### 5.2 下载到本地

在**本地电脑**运行：

```bash
# Windows (Git Bash)
scp -r -P <POD_SSH_PORT> \
    root@<POD_IP>:/workspace/results/quick_test \
    C:/Users/Ava\ Ai/Desktop/results/

# Linux/Mac
scp -r -P <POD_SSH_PORT> \
    root@<POD_IP>:/workspace/results/quick_test \
    ~/Desktop/results/
```

或压缩后下载（更快）：

```bash
# 在RunPod终端压缩
cd /workspace
tar -czf results.tar.gz results/

# 在本地下载
scp -P <POD_SSH_PORT> root@<POD_IP>:/workspace/results.tar.gz ~/Desktop/

# 本地解压
tar -xzf results.tar.gz
```

---

## ✅ 完成！

**停止Pod避免计费**: 在RunPod界面点击 "Stop"

---

## 📊 你将得到的结果文件

```
results/quick_test/
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
| 设置 + 快速测试 | ~20分钟 | RTX 4090 | ~$0.13 |
| 标准测试 | ~40分钟 | RTX 4090 | ~$0.26 |
| 完整测试 | ~90分钟 | RTX 4090 | ~$0.59 |

**总费用**: $0.13 - $0.60

---

## 🎓 完整流程（复制粘贴整段）

如果你想一次性完成所有步骤，复制粘贴这整段到RunPod终端：

```bash
# ============================================================================
# RunPod 完整量化实验 - 一键流程
# 复制粘贴这整段到RunPod终端
# ============================================================================

# 假设项目已经上传到 /workspace/vggt

# 步骤1: 环境设置
echo "正在设置环境..."
cd /workspace/vggt
bash scripts/runpod_setup_comprehensive.sh <<< "y"

# 步骤2: 运行快速测试
echo "正在运行快速测试..."
bash /workspace/run_quick_test.sh

# 步骤3: 显示结果
echo ""
echo "=============================================================================="
echo "✅ 实验完成！"
echo "=============================================================================="
echo ""
cat /workspace/results/quick_test/comprehensive_report.txt | head -40
echo ""
echo "完整结果保存在: /workspace/results/quick_test/"
echo "请下载结果到本地："
echo "  scp -r -P <PORT> root@<IP>:/workspace/results/quick_test ~/Desktop/"
echo ""
echo "记得停止Pod以避免持续计费！"
echo "=============================================================================="
```

---

## ❓ 常见问题

### Q1: 如何找到Pod的IP和SSH端口？

**A**: 在RunPod界面点击"Connect" → "TCP Port Mappings"，查看：
- **SSH Port**: 如 `12345`
- **SSH String**: `ssh root@123.45.67.89 -p 12345`
- IP就是 `123.45.67.89`

---

### Q2: CUDA out of memory怎么办？

**A**: 减少测试图像数量：

```bash
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 3 \
    --output_dir /workspace/results/small_test
```

---

### Q3: 找不到测试图像怎么办？

**A**: 手动下载数据：

```bash
cd /workspace/vggt
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d
```

或使用自己的图像：
```bash
mkdir -p /workspace/data/my_images
# 上传图像到这个目录，然后：
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/my_images \
    --max_images 10 \
    --output_dir /workspace/results/my_test
```

---

### Q4: 连接断开了怎么办？

**A**: 使用tmux防止断开：

```bash
# 创建tmux会话
tmux new -s quantization

# 在tmux中运行实验
bash /workspace/run_standard_test.sh

# 分离tmux: Ctrl+B, 然后 D
# 重新连接: tmux attach -t quantization
```

---

## 📚 进阶使用

### 自定义实验

```bash
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder <YOUR_IMAGE_FOLDER> \
    --max_images <NUMBER> \
    --output_dir /workspace/results/<EXP_NAME> \
    --device cuda
```

### 后台运行

```bash
cd /workspace/vggt
nohup bash /workspace/run_full_test.sh > /workspace/results/run.log 2>&1 &

# 查看日志
tail -f /workspace/results/run.log
```

### 批量实验

```bash
# 运行多个实验
bash /workspace/run_quick_test.sh
bash /workspace/run_standard_test.sh

# 或并行（小心GPU内存）
bash /workspace/run_quick_test.sh &
bash /workspace/run_standard_test.sh &
wait
```

---

## 📖 文档导航

### RunPod相关

- **[RUNPOD_START_HERE.md](RUNPOD_START_HERE.md)** ← 你在这里（快速开始）
- **[RUNPOD_COMPREHENSIVE_GUIDE.md](RUNPOD_COMPREHENSIVE_GUIDE.md)** - 完整RunPod指南（30分钟阅读）
- **[RUNPOD_QUICK_COMMANDS.md](RUNPOD_QUICK_COMMANDS.md)** - 快速命令参考（查找命令）

### 实验相关

- **[START_HERE_COMPREHENSIVE.md](START_HERE_COMPREHENSIVE.md)** - 项目总览
- **[COMPREHENSIVE_QUANTIZATION_GUIDE.md](COMPREHENSIVE_QUANTIZATION_GUIDE.md)** - 量化指南
- **[EXPERIMENT_PARAMETERS_EXPLAINED.md](EXPERIMENT_PARAMETERS_EXPLAINED.md)** - 参数详解

---

## 🎉 总结

使用RunPod运行完整量化实验：

| 维度 | 内容 |
|------|------|
| **时间** | 20-40分钟 |
| **费用** | $0.13 - $0.50 |
| **步骤** | 5步完成 |
| **结果** | 8种方案 + 8种指标 |
| **输出** | JSON + 文本 + 图表 |

---

## ⚡ 立即开始

### 最简单的开始方式

1. **创建Pod**: RunPod.io → Deploy → RTX 4090
2. **上传代码**: `scp -r -P <PORT> ./vggt root@<IP>:/workspace/`
3. **运行这一行**:
```bash
cd /workspace/vggt && bash scripts/runpod_setup_comprehensive.sh <<< "y" && bash /workspace/run_quick_test.sh
```
4. **查看结果**: `cat /workspace/results/quick_test/comprehensive_report.txt`
5. **下载**: `scp -r -P <PORT> root@<IP>:/workspace/results ~/Desktop/`
6. **停止Pod**: 在RunPod界面点击"Stop"

**10-20分钟后，你将拥有完整的量化实验结果！** 🎊

---

## 🔗 快速链接

- RunPod: https://runpod.io
- 项目GitHub: <YOUR_REPO_URL>
- 问题反馈: <YOUR_ISSUE_URL>

---

**需要帮助？**

- 查看 [RUNPOD_COMPREHENSIVE_GUIDE.md](RUNPOD_COMPREHENSIVE_GUIDE.md) - 完整指南
- 查看 [RUNPOD_QUICK_COMMANDS.md](RUNPOD_QUICK_COMMANDS.md) - 命令参考
- 查看 [常见问题](#常见问题) - 本页上方

**祝实验顺利！** 🚀
