# RunPod 完整实验框架 - 总结

**✅ 所有文件已准备就绪！**

---

## 📦 已创建的文件清单

### 核心文件

| 文件 | 作用 | 何时使用 |
|------|------|----------|
| **scripts/runpod_setup_comprehensive.sh** | 一键环境设置脚本 | Pod创建后首次运行 |
| **scripts/comprehensive_evaluation.py** | 完整评估脚本 | 运行量化实验 |
| **vggt/quantization/comprehensive_quantizer.py** | 量化器实现 | 自动调用 |

### 文档文件（全部在项目根目录）

| 文档 | 页数 | 内容 | 适合谁 |
|------|------|------|--------|
| **RUNPOD_START_HERE.md** | 1页 | ⚡ 5步快速开始 | 所有人（必读） |
| **RUNPOD_COMPREHENSIVE_GUIDE.md** | 20页 | 完整RunPod指南 | 深入了解 |
| **RUNPOD_QUICK_COMMANDS.md** | 15页 | 命令速查手册 | 查找命令时 |
| **START_HERE_COMPREHENSIVE.md** | 5页 | 项目总览 | 了解框架 |
| **COMPREHENSIVE_QUANTIZATION_GUIDE.md** | 25页 | 量化实验指南 | 理解原理 |
| **EXPERIMENT_PARAMETERS_EXPLAINED.md** | 30页 | 参数详解 | 调优参数 |
| **NEW_FRAMEWORK_SUMMARY.md** | 15页 | 改进总结 | 向同伴展示 |

### 快捷脚本（设置后自动生成）

| 脚本 | 位置 | 功能 |
|------|------|------|
| `run_quick_test.sh` | /workspace/ | 快速测试（5张图） |
| `run_standard_test.sh` | /workspace/ | 标准测试（10张图） |
| `run_full_test.sh` | /workspace/ | 完整测试（50张图） |

---

## 🚀 RunPod完整流程（从零到结果）

### 方法1: 最快方式（推荐）

```bash
# ============================================================================
# 复制这整段到RunPod终端，一键完成所有操作
# ============================================================================

# 前提：项目文件已上传到 /workspace/vggt

# 1. 设置环境
cd /workspace/vggt && bash scripts/runpod_setup_comprehensive.sh <<< "y"

# 2. 运行快速测试
bash /workspace/run_quick_test.sh

# 3. 查看结果
cat /workspace/results/quick_test/comprehensive_report.txt

echo "✅ 完成！下载结果: scp -r -P <PORT> root@<IP>:/workspace/results ~/Desktop/"
```

**预计时间**: 15-20分钟
**预计费用**: ~$0.13 (RTX 4090)

---

### 方法2: 分步执行（适合学习）

#### 步骤1: 创建Pod（5分钟）

1. 访问 https://runpod.io
2. Deploy → GPU Pods
3. 选择 **RTX 4090** ($0.39/hr)
4. Container Disk: 50GB, Volume: 50GB
5. Deploy

#### 步骤2: 上传项目（5-10分钟）

**在本地电脑**:
```bash
# Windows (Git Bash)
cd "C:\Users\Ava Ai\Desktop\8539Project\code"
scp -r -P <POD_SSH_PORT> ./vggt root@<POD_IP>:/workspace/

# Linux/Mac
cd ~/path/to/code
scp -r -P <POD_SSH_PORT> ./vggt root@<POD_IP>:/workspace/
```

或**在RunPod终端**:
```bash
cd /workspace
git clone <YOUR_REPO_URL> vggt
```

#### 步骤3: 设置环境（3-5分钟）

**在RunPod终端**:
```bash
cd /workspace/vggt
bash scripts/runpod_setup_comprehensive.sh
```

选择下载数据: `y`

#### 步骤4: 运行实验（5-15分钟）

选择一个：
```bash
bash /workspace/run_quick_test.sh      # 快速（5-10分钟）
bash /workspace/run_standard_test.sh   # 标准（10-15分钟）
bash /workspace/run_full_test.sh       # 完整（30-60分钟）
```

#### 步骤5: 下载结果（2分钟）

**在本地电脑**:
```bash
scp -r -P <POD_SSH_PORT> root@<POD_IP>:/workspace/results/quick_test ~/Desktop/
```

#### 步骤6: 停止Pod

在RunPod界面点击 "Stop"

---

## 📖 文档使用指南

### 新手路径

1. **[RUNPOD_START_HERE.md](RUNPOD_START_HERE.md)** (5分钟)
   - 5步快速开始
   - 必读

2. 运行第一个实验 (10-20分钟)
   - 按照文档执行

3. **[START_HERE_COMPREHENSIVE.md](START_HERE_COMPREHENSIVE.md)** (10分钟)
   - 理解项目结构
   - 了解量化方案

### 进阶路径

4. **[COMPREHENSIVE_QUANTIZATION_GUIDE.md](COMPREHENSIVE_QUANTIZATION_GUIDE.md)** (30分钟)
   - 深入理解量化
   - 学习评估指标

5. **[EXPERIMENT_PARAMETERS_EXPLAINED.md](EXPERIMENT_PARAMETERS_EXPLAINED.md)** (按需)
   - 调优参数
   - 自定义实验

### 运维路径

6. **[RUNPOD_COMPREHENSIVE_GUIDE.md](RUNPOD_COMPREHENSIVE_GUIDE.md)** (按需)
   - 故障排查
   - 高级技巧

7. **[RUNPOD_QUICK_COMMANDS.md](RUNPOD_QUICK_COMMANDS.md)** (查询)
   - 快速查找命令
   - 复制即用

---

## 🎯 不同场景的使用方法

### 场景1: 第一次使用（我什么都不懂）

**阅读顺序**:
1. RUNPOD_START_HERE.md (5分钟)
2. 直接运行（照着做）

**命令**:
```bash
cd /workspace/vggt && bash scripts/runpod_setup_comprehensive.sh <<< "y" && bash /workspace/run_quick_test.sh
```

---

### 场景2: 需要提交报告

**阅读顺序**:
1. RUNPOD_START_HERE.md (复习)
2. START_HERE_COMPREHENSIVE.md (理解框架)
3. 运行标准测试

**命令**:
```bash
bash /workspace/run_standard_test.sh
```

**输出**: 10张图像，完整的8种方案对比

---

### 场景3: 需要调整参数

**阅读顺序**:
1. EXPERIMENT_PARAMETERS_EXPLAINED.md (30分钟)
2. 修改参数运行

**命令**:
```bash
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 20 \
    --output_dir /workspace/results/custom_test \
    --device cuda
```

---

### 场景4: 遇到问题

**查询顺序**:
1. RUNPOD_START_HERE.md → 常见问题
2. RUNPOD_COMPREHENSIVE_GUIDE.md → 故障排查
3. RUNPOD_QUICK_COMMANDS.md → 查找命令

**常见解决方案**:
```bash
# CUDA内存不足
python scripts/comprehensive_evaluation.py --max_images 3 --device cuda

# 找不到图像
find /workspace/data -name "*.jpg" | head

# 依赖问题
pip install --force-reinstall torch torchvision
```

---

### 场景5: 向同伴展示改进

**阅读**:
- NEW_FRAMEWORK_SUMMARY.md (15分钟)

**展示内容**:
- 新旧对比表格
- 8种方案对比
- 8种评估指标
- 完整的可视化

---

## 📊 预期结果

运行完成后，你会得到：

### 文件输出

```
/workspace/results/quick_test/
├── comprehensive_results.json          # 完整数据
├── comprehensive_report.txt            # 文本报告
└── comprehensive_visualizations.png    # 图表
```

### 报告内容

```
================================================================================
完整量化评估报告
================================================================================

方案                         | 大小(MB) | 压缩率 | 时间(s) | MAE      | PSNR  | CE
--------------------------------------------------------------------------------------------
Baseline_FP32                | 4000.00  | 1.00x  | 0.0500  | 0.000000 | ∞ dB  | 0.000000
INT8_Per_Tensor_Symmetric    | 1000.00  | 4.00x  | 0.0400  | 0.001247 | 35.2  | 0.002456
INT8_Per_Channel_Symmetric   | 1010.00  | 3.96x  | 0.0385  | 0.000523 | 41.8  | 0.001023 ⭐
INT4_Group_128               |  500.00  | 8.00x  | 0.0350  | 0.007891 | 28.3  | 0.015234
... (共8个方案)

实验总结:
- 最高压缩率: INT4_Group_128 (8.00x)
- 最快推理: INT4_Group_128 (1.43x)
- 最高精度: INT8_Per_Channel_Asymmetric (MAE: 0.000498)
```

### 图表内容

6个专业图表：
1. 模型大小对比（柱状图）
2. 推理时间对比（柱状图）
3. 压缩率对比（柱状图）
4. 加速比对比（柱状图）
5. MAE精度对比（柱状图）
6. 精度vs压缩率权衡（散点图）

---

## 💰 费用明细

| 任务 | 时间 | GPU | 费用 |
|------|------|-----|------|
| 环境设置 | 5分钟 | RTX 4090 | ~$0.03 |
| 快速测试（5张） | 10分钟 | RTX 4090 | ~$0.065 |
| 标准测试（10张） | 15分钟 | RTX 4090 | ~$0.10 |
| 完整测试（50张） | 60分钟 | RTX 4090 | ~$0.39 |

**总费用估算**:
- 快速体验: ~$0.10
- 标准实验: ~$0.13
- 完整实验: ~$0.42

---

## ✅ 检查清单

### 上传前（本地）

- [ ] 确认所有文件在 `vggt/` 目录
- [ ] 特别检查:
  - [ ] scripts/runpod_setup_comprehensive.sh
  - [ ] scripts/comprehensive_evaluation.py
  - [ ] vggt/quantization/comprehensive_quantizer.py
  - [ ] 所有.md文档

### RunPod上（首次）

- [ ] 创建Pod（RTX 4090）
- [ ] 上传项目到 `/workspace/vggt`
- [ ] 运行 `bash scripts/runpod_setup_comprehensive.sh`
- [ ] 选择下载数据 (y)

### 实验前

- [ ] 确认CUDA可用: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 确认图像存在: `find /workspace/data -name "*.jpg" | head`
- [ ] 确认磁盘空间: `df -h /workspace`

### 实验后

- [ ] 查看文本报告: `cat /workspace/results/*/comprehensive_report.txt`
- [ ] 下载所有结果到本地
- [ ] 停止Pod（避免持续计费）

---

## 🎓 最佳实践

### 1. 使用tmux防止断开

```bash
tmux new -s quantization
# 在tmux中运行实验
# Ctrl+B, D 分离
# tmux attach -t quantization 重新连接
```

### 2. 压缩后下载

```bash
cd /workspace
tar -czf results.tar.gz results/
# 本地: scp -P <PORT> root@<IP>:/workspace/results.tar.gz ~/
```

### 3. 批量实验

```bash
for n in 5 10 20; do
    python scripts/comprehensive_evaluation.py \
        --max_images $n \
        --output_dir /workspace/results/exp_${n}img
done
```

### 4. 自动记录

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python scripts/comprehensive_evaluation.py \
    --max_images 10 \
    --output_dir /workspace/results/exp_$TIMESTAMP \
    2>&1 | tee /workspace/logs/run_$TIMESTAMP.log
```

---

## 🔗 快速参考

### 最常用命令

```bash
# 设置环境
cd /workspace/vggt && bash scripts/runpod_setup_comprehensive.sh

# 快速测试
bash /workspace/run_quick_test.sh

# 查看结果
cat /workspace/results/quick_test/comprehensive_report.txt

# 监控GPU
watch -n 1 nvidia-smi

# 下载结果（本地）
scp -r -P <PORT> root@<IP>:/workspace/results ~/Desktop/
```

### Pod IP和端口

RunPod界面 → Connect → TCP Port Mappings:
- SSH Port: 如 `12345`
- SSH String: `ssh root@123.45.67.89 -p 12345`

### 文档快速链接

- 快速开始: RUNPOD_START_HERE.md
- 完整指南: RUNPOD_COMPREHENSIVE_GUIDE.md
- 命令速查: RUNPOD_QUICK_COMMANDS.md
- 项目总览: START_HERE_COMPREHENSIVE.md

---

## 🎉 总结

### 你现在拥有：

✅ **完整的RunPod实验框架**
- 1个设置脚本
- 1个评估脚本
- 3个快捷命令
- 7个详细文档（100+页）

✅ **从FP32到INT8/INT4的完整量化**
- 1个Baseline (FP32)
- 7种量化方案
- 8种评估指标
- 专业可视化

✅ **详细的使用指南**
- 5步快速开始
- 完整故障排查
- 命令速查手册
- 参数调优指南

### 下一步：

1. **立即开始**: 打开 RUNPOD_START_HERE.md
2. **创建Pod**: RunPod.io
3. **运行实验**: 复制粘贴命令
4. **10-20分钟后**: 拥有完整结果

---

## 📞 需要帮助？

### 快速解决

1. **查看常见问题**: RUNPOD_START_HERE.md 底部
2. **查找命令**: RUNPOD_QUICK_COMMANDS.md
3. **故障排查**: RUNPOD_COMPREHENSIVE_GUIDE.md 故障排查章节

### 文档导航

| 问题 | 查看 |
|------|------|
| 如何开始? | RUNPOD_START_HERE.md |
| 找不到命令 | RUNPOD_QUICK_COMMANDS.md |
| 遇到错误 | RUNPOD_COMPREHENSIVE_GUIDE.md 故障排查 |
| 理解原理 | COMPREHENSIVE_QUANTIZATION_GUIDE.md |
| 调整参数 | EXPERIMENT_PARAMETERS_EXPLAINED.md |
| 向同伴展示 | NEW_FRAMEWORK_SUMMARY.md |

---

**准备好了吗？开始你的RunPod实验！** 🚀

```bash
# 第一步：打开 RUNPOD_START_HERE.md
# 第二步：按照文档操作
# 第三步：10-20分钟后，享受完整的实验结果！
```

**祝实验顺利！** 🎊
