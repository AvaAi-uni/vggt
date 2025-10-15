# 项目结构说明

**v3.0 - 纯RunPod工作流** 🆕

---

## 📂 目录结构

```
vggt/
├── 📘 核心文档
│   ├── README.md                               # 项目介绍
│   ├── LICENSE.txt                             # 许可证
│   ├── PROJECT_STRUCTURE.md                    # 本文档（项目结构）
│   │
│   ├── 🆕 RUNPOD_PURE_WORKFLOW.md              # 纯RunPod工作流（最快上手）⭐
│   ├── RUNPOD_START_HERE.md                    # RunPod快速开始（3步）
│   ├── RUNPOD_QUICK_COMMANDS.md                # RunPod命令速查
│   ├── RUNPOD_COMPREHENSIVE_GUIDE.md           # RunPod完整指南
│   ├── RUNPOD_COMPLETE_SUMMARY.md              # RunPod总结
│   │
│   ├── START_HERE_COMPREHENSIVE.md             # 项目总览
│   ├── QUICK_START_COMPREHENSIVE.md            # 快速开始指南
│   ├── COMPREHENSIVE_QUANTIZATION_GUIDE.md     # 完整量化指南
│   ├── EXPERIMENT_PARAMETERS_EXPLAINED.md      # 参数详解
│   ├── NEW_FRAMEWORK_SUMMARY.md                # 改进总结
│   │
│   └── 🆕 UPDATE_V3.0_PURE_RUNPOD.md           # v3.0更新说明
│
├── 💻 核心代码
│   ├── vggt/                                   # 主模块
│   │   ├── quantization/                       # 量化模块
│   │   │   ├── __init__.py
│   │   │   ├── quantizer.py                    # 基础量化器
│   │   │   ├── advanced_quantizer.py           # 高级量化器
│   │   │   └── comprehensive_quantizer.py      # 完整量化器（新）
│   │   │
│   │   ├── models/                             # 模型
│   │   ├── layers/                             # 网络层
│   │   ├── heads/                              # 预测头
│   │   ├── utils/                              # 工具函数
│   │   └── dependency/                         # 依赖模块
│   │
│   ├── scripts/                                # 脚本
│   │   ├── 🆕 runpod_full_workflow.sh          # 一键完整流程（推荐）⭐
│   │   ├── runpod_setup_comprehensive.sh       # RunPod环境设置
│   │   ├── comprehensive_evaluation.py         # 完整评估脚本
│   │   ├── download_eth3d.py                   # ETH3D数据下载
│   │   ├── quantize_model.py                   # 量化脚本（旧）
│   │   ├── inference_quantized.py              # 推理脚本
│   │   ├── compare_quantization.py             # 对比脚本（旧）
│   │   └── visualize_results.py                # 可视化脚本
│   │
│   └── training/                               # 训练模块
│       ├── trainer.py
│       ├── data/
│       ├── config/
│       └── train_utils/
│
└── ⚙️ 配置文件
    ├── requirements.txt                        # Python依赖
    ├── pyproject.toml                          # 项目配置
    ├── .gitignore                              # Git忽略
    └── docs/                                   # 其他文档
```

---

## 📖 文档导航

### 🚀 RunPod快速路径（推荐，最快上手）

1. **RUNPOD_PURE_WORKFLOW.md** (5分钟) 🆕 ⭐
   - 纯RunPod工作流
   - 复制粘贴一行命令
   - 15分钟完成实验
   - **最快的开始方式！**

2. **RUNPOD_START_HERE.md** (10分钟)
   - 3步快速开始
   - 详细说明
   - 常见问题

3. 运行第一个实验 (15分钟)
   - 一键自动完成

### 📚 深入学习路径

4. **START_HERE_COMPREHENSIVE.md** (10分钟)
   - 项目总览
   - 量化方案说明
   - 文件结构

5. **COMPREHENSIVE_QUANTIZATION_GUIDE.md** (30分钟)
   - 量化原理详解
   - 8种方案说明
   - 评估指标解释

6. **EXPERIMENT_PARAMETERS_EXPLAINED.md** (按需)
   - 所有参数详解
   - 参数调优指南
   - 场景化配置

### 🔧 运维查询路径

7. **RUNPOD_QUICK_COMMANDS.md** (查询用)
   - 命令速查手册
   - 复制即用

8. **RUNPOD_COMPREHENSIVE_GUIDE.md** (故障排查)
   - 完整RunPod使用指南
   - 详细故障排查
   - 成本优化

### 📊 展示路径（向同伴展示）

9. **NEW_FRAMEWORK_SUMMARY.md** (15分钟)
   - 新旧框架对比
   - 改进亮点总结
   - 如何展示给同伴

10. **UPDATE_V3.0_PURE_RUNPOD.md** (10分钟) 🆕
    - v3.0版本更新说明
    - 纯RunPod工作流介绍
    - 技术改进清单

---

## 🚀 快速开始（纯RunPod）

### ⚡ 超级快速（一行命令）

**在RunPod终端复制粘贴**:

```bash
cd /workspace && git clone https://github.com/yourusername/vggt.git vggt && cd vggt && bash scripts/runpod_full_workflow.sh quick
```

**15分钟后自动完成！**

---

### 📝 分步执行（RunPod终端）

```bash
# 步骤1: 克隆代码
cd /workspace
git clone https://github.com/yourusername/vggt.git vggt

# 步骤2: 运行一键流程
cd vggt
bash scripts/runpod_full_workflow.sh quick

# 步骤3: 查看结果
cat /workspace/results/quick_test_*/comprehensive_report.txt

# 步骤4: 下载结果
cd /workspace
tar -czf results.tar.gz results/
# 在RunPod界面下载 /workspace/results.tar.gz
```

---

## 🎯 核心功能

### 1. 完整的Baseline
- FP32原始模型作为对比基准

### 2. 7种量化方案
- INT8 Per-Tensor Symmetric/Asymmetric
- INT8 Per-Channel Symmetric/Asymmetric
- INT4 Group-wise (32/64/128)

### 3. 8种评估指标
- MAE, MSE, RMSE, PSNR
- Cross Entropy, Cosine Similarity
- Relative Error, SSIM

### 4. 专业输出
- JSON格式数据
- 格式化文本报告
- 可视化图表（6个子图）

---

## 📊 实验输出

运行实验后，你会得到：

```
results/my_test/
├── comprehensive_results.json          # 完整数据
├── comprehensive_report.txt            # 文本报告
└── comprehensive_visualizations.png    # 可视化图表
```

报告包含：
- 8个方案的对比表格
- 详细的评估指标
- 最佳方案推荐

---

## 💾 ETH3D数据集

### 下载命令

```bash
# 自动下载（需要7z工具）
python scripts/download_eth3d.py --output_dir data/eth3d

# 手动下载
# 1. 访问: https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z
# 2. 下载文件（~15GB）
# 3. 解压到 data/eth3d/
```

### 数据结构

```
data/eth3d/
├── courtyard/
│   └── dslr_images_undistorted/
│       ├── DSC_0001.JPG
│       ├── DSC_0002.JPG
│       └── ...
├── delivery_area/
│   └── dslr_images_undistorted/
│       └── ...
└── ...（其他场景）
```

---

## 🔧 常用命令

### 评估实验

```bash
# 基础命令
python scripts/comprehensive_evaluation.py \
    --image_folder <IMAGE_FOLDER> \
    --max_images 10 \
    --output_dir results/test

# 快速测试（5张图）
python scripts/comprehensive_evaluation.py \
    --image_folder data/eth3d/courtyard/dslr_images_undistorted \
    --max_images 5 \
    --output_dir results/quick_test

# 标准测试（10张图）
python scripts/comprehensive_evaluation.py \
    --image_folder data/eth3d/courtyard/dslr_images_undistorted \
    --max_images 10 \
    --output_dir results/standard_test
```

### 下载数据

```bash
# 下载ETH3D
python scripts/download_eth3d.py --output_dir data/eth3d

# 保留压缩包
python scripts/download_eth3d.py --output_dir data/eth3d --keep_archive

# 跳过下载（如果已存在）
python scripts/download_eth3d.py --output_dir data/eth3d --skip_download
```

### 查看结果

```bash
# 文本报告
cat results/test/comprehensive_report.txt

# JSON数据
python -m json.tool results/test/comprehensive_results.json
```

---

## 📝 文件说明

### 核心脚本

| 文件 | 作用 | 何时使用 | 推荐度 |
|------|------|----------|--------|
| **runpod_full_workflow.sh** 🆕 | 一键完整流程 | RunPod运行实验（推荐） | ⭐⭐⭐⭐⭐ |
| **comprehensive_evaluation.py** | 完整评估脚本 | 自定义参数实验 | ⭐⭐⭐⭐ |
| **runpod_setup_comprehensive.sh** | RunPod环境设置 | 单独设置环境时 | ⭐⭐⭐ |
| **download_eth3d.py** | 下载ETH3D数据集 | 手动下载数据 | ⭐⭐⭐ |

### 核心文档

| 文档 | 页数 | 内容 | 读者 | 推荐度 |
|------|------|------|------|--------|
| **RUNPOD_PURE_WORKFLOW.md** 🆕 | 5页 | 纯RunPod快速参考 | 所有人 | ⭐⭐⭐⭐⭐ |
| **RUNPOD_START_HERE.md** | 10页 | RunPod快速开始 | 新手 | ⭐⭐⭐⭐⭐ |
| **START_HERE_COMPREHENSIVE.md** | 5页 | 项目总览 | 了解框架 | ⭐⭐⭐⭐ |
| **COMPREHENSIVE_QUANTIZATION_GUIDE.md** | 25页 | 量化指南 | 深入学习者 | ⭐⭐⭐⭐ |
| **EXPERIMENT_PARAMETERS_EXPLAINED.md** | 30页 | 参数详解 | 高级用户 | ⭐⭐⭐ |
| **RUNPOD_QUICK_COMMANDS.md** | 15页 | 命令速查 | 查询时用 | ⭐⭐⭐⭐ |

---

## ✅ 已删除的文件

以下不必要的文件已被删除：

### 旧文档（已删除）
- CODE_OF_CONDUCT.md
- CONTRIBUTING.md
- DOWNLOAD_AND_VISUALIZE.md
- FIXES_APPLIED.md
- IMPLEMENTATION_SUMMARY.md
- LATEST_FIXES_SUMMARY.md
- QUANTIZATION_COMPARISON_GUIDE.md
- QUANTIZATION_FIXES.md
- QUANTIZATION_QUICK_COMMANDS.md
- QUANTIZATION_README.md
- QUICK_VISUALIZATION_COMMANDS.md
- RUNPOD_COMMANDS.md
- RUNPOD_COMPLETE_WORKFLOW.md
- RUNPOD_DEPLOYMENT.md
- RUNPOD_SAVE_STATE.md
- START_HERE.md（旧版）
- UPDATES_AND_FIXES.md

### 示例文件（已删除）
- demo_colmap.py
- demo_gradio.py
- demo_viser.py
- visual_util.py
- examples/
- requirements_demo.txt

### IDE文件（已删除）
- .idea/

---

## 🎉 总结

### v3.0 - 纯RunPod工作流 🆕

**核心改进**:
- 🆕 一键执行脚本: `runpod_full_workflow.sh`
- 🆕 纯RunPod操作: 无需本地电脑
- 🆕 超级快速: 一行命令15分钟完成
- 🆕 新增文档: `RUNPOD_PURE_WORKFLOW.md`

### 当前项目包含：

✅ **完整的量化框架**
- 1个Baseline (FP32)
- 7种量化方案 (INT8/INT4)
- 8种评估指标
- 专业可视化输出

✅ **一键执行系统** 🆕
- `runpod_full_workflow.sh`: 全自动流程
- 自动环境检查
- 自动依赖安装
- 自动数据下载
- 自动实验运行

✅ **核心文档**（11个）
- 3个RunPod快速开始文档 🆕
- 项目总览和量化指南
- 参数详解和命令速查
- v3.0更新说明 🆕

✅ **核心脚本**（8个）
- 一键完整流程 🆕
- 评估、下载、可视化
- RunPod自动化设置

✅ **清晰的结构**
- 删除了20+个不必要的文件
- 保留了所有核心功能
- 文档简洁易懂
- 纯RunPod工作流 🆕

---

## 📞 获取帮助

### 快速查找

| 需求 | 查看文档 | 时间 |
|------|----------|------|
| 🚀 最快上手 | RUNPOD_PURE_WORKFLOW.md 🆕 | 5分钟 |
| 第一次使用RunPod | RUNPOD_START_HERE.md | 10分钟 |
| 查找RunPod命令 | RUNPOD_QUICK_COMMANDS.md | 查询 |
| 遇到RunPod问题 | RUNPOD_COMPREHENSIVE_GUIDE.md | 按需 |
| 了解项目结构 | START_HERE_COMPREHENSIVE.md | 10分钟 |
| 理解量化原理 | COMPREHENSIVE_QUANTIZATION_GUIDE.md | 30分钟 |
| 调整实验参数 | EXPERIMENT_PARAMETERS_EXPLAINED.md | 按需 |
| 向同伴展示 | NEW_FRAMEWORK_SUMMARY.md | 15分钟 |
| 查看版本更新 | UPDATE_V3.0_PURE_RUNPOD.md 🆕 | 10分钟 |

### 推荐阅读顺序

#### 新手快速上手
1. **RUNPOD_PURE_WORKFLOW.md** (5分钟) 🆕 ⭐
2. 直接运行实验 (15分钟)
3. **START_HERE_COMPREHENSIVE.md** (10分钟)

#### 深入学习
4. **COMPREHENSIVE_QUANTIZATION_GUIDE.md** (30分钟)
5. **EXPERIMENT_PARAMETERS_EXPLAINED.md** (按需)

---

## ⚡ 立即开始

**在RunPod终端复制粘贴一行**:

```bash
cd /workspace && git clone https://github.com/yourusername/vggt.git vggt && cd vggt && bash scripts/runpod_full_workflow.sh quick
```

**15分钟后，拥有完整的量化实验结果！** 🎊

---

**v3.0 - 项目已优化为纯RunPod工作流，更快更简单！** 🚀
