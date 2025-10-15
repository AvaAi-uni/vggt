# RunPod 完整实验框架 - 总结

**✅ 版本 3.0 - 纯RunPod工作流**

**核心改进**:
- ✅ 所有操作都在RunPod上完成
- ✅ 无需本地电脑命令
- ✅ 一键执行脚本
- ✅ 自动化程度更高

---

## 📦 核心文件清单

### 主要执行脚本

| 文件 | 作用 | 何时使用 |
|------|------|----------|
| **scripts/runpod_full_workflow.sh** | 🆕 一键完整流程（推荐） | 运行完整实验 |
| **scripts/runpod_setup_comprehensive.sh** | 环境设置脚本 | 单独设置环境时 |
| **scripts/comprehensive_evaluation.py** | 评估脚本 | 自定义参数时 |
| **vggt/quantization/comprehensive_quantizer.py** | 量化器实现 | 自动调用 |

### 文档文件（全部在项目根目录）

| 文档 | 页数 | 内容 | 适合谁 |
|------|------|------|--------|
| **RUNPOD_PURE_WORKFLOW.md** | 🆕 5页 | 纯RunPod工作流（推荐） | 所有人 |
| **RUNPOD_START_HERE.md** | 10页 | 3步快速开始 | 新手（必读） |
| **RUNPOD_QUICK_COMMANDS.md** | 15页 | 命令速查手册 | 查找命令时 |
| **RUNPOD_COMPREHENSIVE_GUIDE.md** | 20页 | 完整RunPod指南 | 深入了解 |
| **START_HERE_COMPREHENSIVE.md** | 5页 | 项目总览 | 了解框架 |
| **COMPREHENSIVE_QUANTIZATION_GUIDE.md** | 25页 | 量化实验指南 | 理解原理 |
| **EXPERIMENT_PARAMETERS_EXPLAINED.md** | 30页 | 参数详解 | 调优参数 |
| **NEW_FRAMEWORK_SUMMARY.md** | 15页 | 改进总结 | 向同伴展示 |

---

## 🚀 RunPod完整流程（纯RunPod操作）

### 方法1: 超级快速（一行命令，推荐）

**在RunPod终端复制粘贴这一行：**

```bash
cd /workspace && git clone https://github.com/yourusername/vggt.git vggt && cd vggt && bash scripts/runpod_full_workflow.sh quick
```

**修改**: 把 `yourusername/vggt` 改成你的仓库地址

**预计时间**: 15分钟
**预计费用**: ~$0.10 (RTX 4090)
**自动完成**: 代码克隆 → 环境设置 → 数据下载 → 实验运行 → 结果显示

---

### 方法2: 分步执行（所有在RunPod终端）

#### 步骤1: 创建Pod（5分钟）

1. 访问 https://runpod.io
2. Deploy → GPU Pods
3. 选择 **RTX 4090** ($0.39/hr)
4. Container Disk: 50GB, Volume: 50GB
5. Deploy
6. 等待Pod启动，点击"Connect" → "Start Web Terminal"

#### 步骤2: 获取代码（2分钟）

**在RunPod终端**:
```bash
# 方法A: Git克隆（推荐）
cd /workspace
git clone https://github.com/yourusername/vggt.git vggt

# 方法B: 从已上传的ZIP解压
cd /workspace
unzip vggt.zip
```

#### 步骤3: 运行实验（10-60分钟）

**在RunPod终端**选择一个：
```bash
# 快速测试（5张图，10分钟）
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh quick

# 标准测试（10张图，15分钟）
bash scripts/runpod_full_workflow.sh standard

# 完整测试（50张图，60分钟）
bash scripts/runpod_full_workflow.sh full
```

**一个命令自动完成**：
- ✅ 检查环境
- ✅ 安装依赖
- ✅ 下载数据
- ✅ 运行实验
- ✅ 显示结果

#### 步骤4: 查看和下载结果

**在RunPod终端**:
```bash
# 查看结果
cat /workspace/results/quick_test_*/comprehensive_report.txt

# 创建下载包
cd /workspace
tar -czf results.tar.gz results/
```

**在RunPod Web界面**:
1. 点击 "Files"
2. 找到 `/workspace/results.tar.gz`
3. 点击下载

#### 步骤5: 停止Pod

在RunPod界面点击 "Stop" 避免继续计费

---

## 📖 文档使用指南

### 🚀 新手路径（推荐）

1. **[RUNPOD_PURE_WORKFLOW.md](RUNPOD_PURE_WORKFLOW.md)** (5分钟) 🆕
   - 纯RunPod工作流
   - 复制粘贴即用
   - **最快上手方式**

2. **[RUNPOD_START_HERE.md](RUNPOD_START_HERE.md)** (10分钟)
   - 3步快速开始
   - 详细说明

3. 运行第一个实验 (15分钟)
   - 一行命令完成
   - 查看结果

### 📚 深入学习路径

4. **[START_HERE_COMPREHENSIVE.md](START_HERE_COMPREHENSIVE.md)** (10分钟)
   - 理解项目结构
   - 了解量化方案

5. **[COMPREHENSIVE_QUANTIZATION_GUIDE.md](COMPREHENSIVE_QUANTIZATION_GUIDE.md)** (30分钟)
   - 深入理解量化原理
   - 学习评估指标

6. **[EXPERIMENT_PARAMETERS_EXPLAINED.md](EXPERIMENT_PARAMETERS_EXPLAINED.md)** (按需)
   - 调优参数
   - 自定义实验

### 🔧 运维查询路径

7. **[RUNPOD_QUICK_COMMANDS.md](RUNPOD_QUICK_COMMANDS.md)** (查询时使用)
   - 快速查找命令
   - 复制即用

8. **[RUNPOD_COMPREHENSIVE_GUIDE.md](RUNPOD_COMPREHENSIVE_GUIDE.md)** (故障排查时)
   - 完整故障排查
   - 高级技巧

---

## 🎯 不同场景的使用方法（纯RunPod）

### 场景1: 第一次使用（我什么都不懂）

**阅读**:
1. RUNPOD_PURE_WORKFLOW.md (5分钟)

**在RunPod终端运行**:
```bash
cd /workspace && \
git clone https://github.com/yourusername/vggt.git vggt && \
cd vggt && \
bash scripts/runpod_full_workflow.sh quick
```

**结果**: 15分钟后自动显示完整实验结果

---

### 场景2: 需要提交报告

**阅读**:
1. RUNPOD_PURE_WORKFLOW.md (复习)
2. START_HERE_COMPREHENSIVE.md (理解框架)

**在RunPod终端运行**:
```bash
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh standard
```

**输出**: 10张图像，完整的8种方案对比

**下载报告**:
```bash
cd /workspace
tar -czf results.tar.gz results/
# 在RunPod界面下载 /workspace/results.tar.gz
```

---

### 场景3: 需要调整参数

**阅读**:
1. EXPERIMENT_PARAMETERS_EXPLAINED.md (30分钟)

**在RunPod终端运行**:
```bash
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/dslr_images_undistorted \
    --max_images 20 \
    --output_dir /workspace/results/custom_test \
    --device cuda
```

---

### 场景4: 遇到问题

**查询顺序**:
1. RUNPOD_PURE_WORKFLOW.md → 故障排查
2. RUNPOD_START_HERE.md → 常见问题
3. RUNPOD_QUICK_COMMANDS.md → 查找命令

**常见解决方案（在RunPod终端）**:
```bash
# CUDA内存不足
cd /workspace/vggt
python scripts/comprehensive_evaluation.py --max_images 3 --output_dir /workspace/results/tiny

# 找不到图像
find /workspace/data -name "*.JPG" | head

# 重新下载数据
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d

# 依赖问题
pip install --force-reinstall torch torchvision numpy matplotlib
```

---

### 场景5: 向同伴展示改进

**阅读**:
- NEW_FRAMEWORK_SUMMARY.md (15分钟)

**在RunPod终端准备展示材料**:
```bash
# 运行标准测试
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh standard

# 查看关键结果
cat /workspace/results/standard_test_*/comprehensive_report.txt

# 打包所有结果
cd /workspace
tar -czf presentation_results.tar.gz results/
```

**展示内容**:
- 新旧对比表格
- 8种方案对比
- 8种评估指标
- 完整的可视化图表

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

## 💰 费用明细（RTX 4090 @ $0.39/hr）

| 任务类型 | 时间 | 费用 | 图像数 | 适用场景 |
|---------|------|------|--------|----------|
| 快速测试 | ~15分钟 | ~$0.10 | 5张 | 首次测试、快速验证 |
| 标准测试 | ~20分钟 | ~$0.13 | 10张 | 作业提交、常规实验 |
| 完整测试 | ~60分钟 | ~$0.40 | 50张 | 论文发表、完整评估 |

**费用包含**:
- ✅ 环境设置和依赖安装
- ✅ ETH3D数据下载（首次）
- ✅ 完整实验运行（8种方案）
- ✅ 结果生成和可视化

**节省费用技巧**:
- 使用tmux防止连接断开
- 实验完成后立即停止Pod
- 多个实验一次性运行
- 使用持久化Volume保存数据（避免重复下载）

---

## ✅ 检查清单（纯RunPod操作）

### 代码准备（在Git仓库）

- [ ] 代码已push到GitHub/GitLab
- [ ] 或准备好ZIP文件上传到RunPod
- [ ] 记录仓库URL: `https://github.com/yourusername/vggt.git`

### RunPod Pod创建

- [ ] 已创建RunPod账户
- [ ] 选择RTX 4090 GPU
- [ ] Container Disk: 50GB
- [ ] Volume Disk: 50GB（可选，用于数据持久化）
- [ ] Pod已启动并可连接

### 运行实验前（在RunPod终端）

- [ ] 代码已克隆: `ls /workspace/vggt`
- [ ] 或已解压: `ls /workspace/vggt`
- [ ] 确认在项目目录: `pwd` 显示 `/workspace/vggt`

### 实验运行中（在RunPod终端）

- [ ] 使用tmux防止断开: `tmux new -s exp`
- [ ] 监控GPU: `watch -n 5 nvidia-smi`（另一个终端）
- [ ] 检查磁盘: `df -h /workspace`

### 实验完成后（在RunPod终端）

- [ ] 查看结果: `cat /workspace/results/*/comprehensive_report.txt`
- [ ] 验证所有文件: `ls -lh /workspace/results/*/`
- [ ] 创建压缩包: `tar -czf /workspace/results.tar.gz /workspace/results/`
- [ ] 在RunPod界面下载压缩包
- [ ] 停止Pod避免计费

### 可选检查（在RunPod终端）

- [ ] 确认CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 确认数据: `find /workspace/data -name "*.JPG" | wc -l`
- [ ] 查看日志: `ls -lh /workspace/*.log`

---

## 🎓 最佳实践（所有在RunPod终端）

### 1. 使用tmux防止断开（推荐）

```bash
# 在RunPod终端
tmux new -s quantization

# 在tmux中运行实验
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh standard

# 分离tmux: Ctrl+B, 然后按 D
# 重新连接: tmux attach -t quantization
# 查看会话: tmux ls
```

### 2. 后台运行长时间实验

```bash
# 在RunPod终端
cd /workspace/vggt
nohup bash scripts/runpod_full_workflow.sh full > /workspace/full_test.log 2>&1 &

# 查看日志
tail -f /workspace/full_test.log

# 查看进程
ps aux | grep runpod_full_workflow
```

### 3. 批量实验（不同参数）

```bash
# 在RunPod终端
cd /workspace/vggt

for n in 5 10 20 50; do
    python scripts/comprehensive_evaluation.py \
        --image_folder /workspace/data/eth3d/courtyard/dslr_images_undistorted \
        --max_images $n \
        --output_dir /workspace/results/exp_${n}img \
        --device cuda
done
```

### 4. 自动记录时间戳

```bash
# 在RunPod终端
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cd /workspace/vggt

python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/dslr_images_undistorted \
    --max_images 10 \
    --output_dir /workspace/results/exp_$TIMESTAMP \
    --device cuda \
    2>&1 | tee /workspace/logs/run_$TIMESTAMP.log
```

### 5. 使用持久化Volume

```bash
# 首次运行时下载数据到Volume
python scripts/download_eth3d.py --output_dir /workspace/persistent_data/eth3d

# 后续实验使用Volume中的数据
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/persistent_data/eth3d/courtyard/dslr_images_undistorted \
    --max_images 10 \
    --output_dir /workspace/results/test
```

---

## 🔗 快速参考（所有在RunPod终端）

### 🚀 最常用命令

```bash
# 一键完整流程（从克隆到结果）
cd /workspace && git clone https://github.com/yourusername/vggt.git vggt && cd vggt && bash scripts/runpod_full_workflow.sh quick

# 快速测试（代码已存在）
cd /workspace/vggt && bash scripts/runpod_full_workflow.sh quick

# 标准测试
cd /workspace/vggt && bash scripts/runpod_full_workflow.sh standard

# 完整测试
cd /workspace/vggt && bash scripts/runpod_full_workflow.sh full

# 查看结果
cat /workspace/results/*/comprehensive_report.txt

# 监控GPU
watch -n 1 nvidia-smi

# 创建下载包
cd /workspace && tar -czf results.tar.gz results/
```

### 📂 重要路径

| 路径 | 说明 |
|------|------|
| `/workspace/vggt/` | 项目根目录 |
| `/workspace/data/eth3d/` | 数据集位置 |
| `/workspace/results/` | 所有实验结果 |
| `/workspace/*.log` | 日志文件 |
| `/workspace/results.tar.gz` | 下载包 |

### 📖 文档快速链接

| 文档 | 用途 |
|------|------|
| **RUNPOD_PURE_WORKFLOW.md** | 纯RunPod快速参考（推荐） |
| **RUNPOD_START_HERE.md** | 快速开始指南 |
| **RUNPOD_QUICK_COMMANDS.md** | 命令速查手册 |
| **RUNPOD_COMPREHENSIVE_GUIDE.md** | 完整RunPod指南 |
| **START_HERE_COMPREHENSIVE.md** | 项目总览 |

---

## 🎉 总结

### ✅ 版本3.0更新 - 纯RunPod工作流

**核心改进**:
- 🆕 **一键执行脚本**: `runpod_full_workflow.sh` 自动完成所有操作
- 🆕 **纯RunPod操作**: 所有命令都在RunPod终端运行，无需本地电脑
- 🆕 **超级快速**: 一行命令15分钟完成实验
- 🆕 **文档简化**: 新增 `RUNPOD_PURE_WORKFLOW.md` 快速参考

### 你现在拥有：

✅ **完整的RunPod实验框架**
- 1个一键流程脚本（新）
- 1个环境设置脚本
- 1个评估脚本
- 8个详细文档（120+页）

✅ **从FP32到INT8/INT4的完整量化**
- 1个Baseline (FP32)
- 7种量化方案
- 8种评估指标
- 专业可视化

✅ **详细的使用指南**
- 3步快速开始（纯RunPod）
- 一键执行命令
- 完整故障排查
- 命令速查手册
- 参数调优指南

### 🚀 立即开始：

**最快方式（复制到RunPod终端）**:
```bash
cd /workspace && \
git clone https://github.com/yourusername/vggt.git vggt && \
cd vggt && \
bash scripts/runpod_full_workflow.sh quick
```

**15分钟后，你将拥有完整的实验结果！**

### 📖 推荐阅读顺序：

1. **RUNPOD_PURE_WORKFLOW.md** (5分钟) - 最快上手
2. **RUNPOD_START_HERE.md** (10分钟) - 详细说明
3. 直接运行实验 (15分钟)
4. 查看结果和下载

---

## 📞 需要帮助？

### 快速解决

1. **查看**: RUNPOD_PURE_WORKFLOW.md → 故障排查
2. **查找命令**: RUNPOD_QUICK_COMMANDS.md
3. **深入学习**: RUNPOD_COMPREHENSIVE_GUIDE.md

### 文档导航

| 问题 | 查看 |
|------|------|
| 如何快速开始? | RUNPOD_PURE_WORKFLOW.md 🆕 |
| 详细步骤说明? | RUNPOD_START_HERE.md |
| 找不到命令? | RUNPOD_QUICK_COMMANDS.md |
| 遇到错误? | RUNPOD_COMPREHENSIVE_GUIDE.md 故障排查 |
| 理解原理? | COMPREHENSIVE_QUANTIZATION_GUIDE.md |
| 调整参数? | EXPERIMENT_PARAMETERS_EXPLAINED.md |
| 向同伴展示? | NEW_FRAMEWORK_SUMMARY.md |

---

## 🎯 核心特点

✅ **纯RunPod工作流** - 所有操作在RunPod终端完成
✅ **超级简单** - 一行命令完成实验
✅ **自动化** - 环境、数据、实验全自动
✅ **低成本** - 快速测试仅需 $0.10
✅ **专业输出** - 8种方案完整对比

---

**准备好了吗？开始你的RunPod实验！** 🚀

```bash
# 在RunPod终端复制粘贴这一行（修改仓库地址）：
cd /workspace && git clone https://github.com/yourusername/vggt.git vggt && cd vggt && bash scripts/runpod_full_workflow.sh quick

# 15分钟后，享受完整的实验结果！
```

**祝实验顺利！** 🎊
