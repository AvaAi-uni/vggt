# 完整量化实验 - 快速开始

**⚡ 5分钟快速上手指南**

---

## 📦 一键运行

### 方法1: 使用测试数据（推荐新手）

```bash
# 进入项目目录
cd C:\Users\Ava Ai\Desktop\8539Project\code\vggt

# 快速测试（5张图像）
python scripts/comprehensive_evaluation.py \
    --image_folder data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir results/quick_test \
    --device cuda
```

**预计时间**: 5-10分钟
**输出**: `results/quick_test/`

---

### 方法2: 标准评估（10张图像）

```bash
python scripts/comprehensive_evaluation.py \
    --image_folder data/eth3d/courtyard/images \
    --max_images 10 \
    --output_dir results/standard_evaluation \
    --device cuda
```

**预计时间**: 10-15分钟
**输出**: `results/standard_evaluation/`

---

### 方法3: 完整评估（50张图像，用于正式实验）

```bash
python scripts/comprehensive_evaluation.py \
    --image_folder data/eth3d/courtyard/images \
    --max_images 50 \
    --output_dir results/full_evaluation \
    --device cuda
```

**预计时间**: 30-60分钟
**输出**: `results/full_evaluation/`

---

## 📊 查看结果

### Windows命令

```bash
# 查看文本报告
type results\quick_test\comprehensive_report.txt

# 打开可视化图表
start results\quick_test\comprehensive_visualizations.png

# 查看JSON数据
python -m json.tool results\quick_test\comprehensive_results.json
```

### Linux/Mac命令

```bash
# 查看文本报告
cat results/quick_test/comprehensive_report.txt

# 打开可视化图表
xdg-open results/quick_test/comprehensive_visualizations.png  # Linux
open results/quick_test/comprehensive_visualizations.png      # Mac

# 查看JSON数据
python -m json.tool results/quick_test/comprehensive_results.json
```

---

## 🎯 实验方案说明

运行脚本后，将自动测试以下8个方案：

| # | 方案名称 | 类型 | 位数 | 特点 |
|---|----------|------|------|------|
| 0 | **Baseline_FP32** | 原始 | 32 | 基准线 |
| 1 | INT8_Per_Tensor_Symmetric | INT8 | 8 | 最快 |
| 2 | INT8_Per_Tensor_Asymmetric | INT8 | 8 | 适应偏斜 |
| 3 | **INT8_Per_Channel_Symmetric** | INT8 | 8 | ⭐ 推荐 |
| 4 | INT8_Per_Channel_Asymmetric | INT8 | 8 | 最高精度 |
| 5 | INT4_Group_128 | INT4 | 4 | 高压缩 |
| 6 | INT4_Group_64 | INT4 | 4 | 平衡 |
| 7 | INT4_Group_32 | INT4 | 4 | 精度优先 |

---

## 📈 评估指标说明

每个方案将输出以下指标：

| 指标 | 说明 | 越小/越大 |
|------|------|-----------|
| **MAE** | 平均绝对误差 | 越小越好 |
| **MSE** | 均方误差 | 越小越好 |
| **RMSE** | 均方根误差 | 越小越好 |
| **PSNR** | 峰值信噪比 | 越大越好 (>40dB优秀) |
| **Cross Entropy** | 交叉熵 | 越小越好 |
| **Cosine Similarity** | 余弦相似度 | 越接近1越好 |
| **模型大小** | MB | 越小越好 |
| **推理时间** | 秒 | 越小越好 |

---

## 🔧 常用参数

### 完整参数列表

```bash
python scripts/comprehensive_evaluation.py \
    --model_name facebook/VGGT-1B \        # 模型名称
    --image_folder PATH \                   # 图像文件夹
    --max_images 10 \                       # 最大图像数
    --output_dir results/eval \             # 输出目录
    --device cuda                           # 设备 (cuda/cpu)
```

### 调试模式（使用更少图像）

```bash
python scripts/comprehensive_evaluation.py \
    --image_folder data/eth3d/courtyard/images \
    --max_images 2 \
    --output_dir results/debug \
    --device cuda
```

**预计时间**: 2-3分钟

---

## 📁 输出文件说明

运行完成后，输出目录包含：

```
results/quick_test/
├── comprehensive_results.json          # JSON格式的完整数据
├── comprehensive_report.txt            # 文本格式的报告
└── comprehensive_visualizations.png    # 可视化图表（6个子图）
```

### 1. comprehensive_results.json

包含所有方案的详细数据：
- 模型大小
- 推理时间
- 压缩率
- 加速比
- 所有评估指标

**用途**: 数据分析、进一步处理

### 2. comprehensive_report.txt

格式化的文本报告，包含：
- 实验概览表格
- 详细指标列表
- 实验总结（最佳方案）

**用途**: 快速查看、写入论文/报告

### 3. comprehensive_visualizations.png

包含6个子图：
1. 模型大小对比（柱状图）
2. 推理时间对比（柱状图）
3. 压缩率对比（柱状图）
4. 加速比对比（柱状图）
5. MAE精度对比（柱状图）
6. 精度vs压缩率权衡（散点图）

**用途**: 直观展示、PPT/海报

---

## 🎨 结果示例

### 文本报告示例

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

方案                           | 大小(MB)   | 压缩率   | 时间(s)    | 加速   | MAE
-------------------------------------------------------------------------------------------------
Baseline_FP32                  |    4000.00 |     1.00x |     0.0500 |   1.00x | 0.000000
INT8_Per_Tensor_Symmetric      |    1000.00 |     4.00x |     0.0400 |   1.25x | 0.001247
INT8_Per_Tensor_Asymmetric     |    1000.00 |     4.00x |     0.0395 |   1.27x | 0.001089
INT8_Per_Channel_Symmetric     |    1010.00 |     3.96x |     0.0385 |   1.30x | 0.000523    ⭐
INT8_Per_Channel_Asymmetric    |    1010.00 |     3.96x |     0.0380 |   1.32x | 0.000498    ⭐⭐
INT4_Group_128                 |     500.00 |     8.00x |     0.0350 |   1.43x | 0.007891
INT4_Group_64                  |     510.00 |     7.84x |     0.0360 |   1.39x | 0.005234
INT4_Group_32                  |     520.00 |     7.69x |     0.0370 |   1.35x | 0.003821

================================================================================
实验总结
================================================================================

最高压缩率: INT4_Group_128 (8.00x)
最快推理: INT4_Group_128 (1.43x)
最高精度: INT8_Per_Channel_Asymmetric (MAE: 0.000498)
```

---

## ❓ 常见问题

### Q: 运行时出现 CUDA out of memory 错误？

**解决方案1**: 减少图像数量
```bash
python scripts/comprehensive_evaluation.py \
    --max_images 3 \
    --output_dir results/small_test
```

**解决方案2**: 使用CPU（较慢）
```bash
python scripts/comprehensive_evaluation.py \
    --device cpu \
    --max_images 5
```

---

### Q: 找不到测试图像？

**检查步骤**:

1. 确认图像文件夹存在：
```bash
# Windows
dir data\eth3d\courtyard\images

# Linux/Mac
ls data/eth3d/courtyard/images
```

2. 如果没有数据，下载ETH3D数据集：
```bash
python scripts/download_eth3d.py --output_dir data/eth3d
```

3. 或使用自己的图像：
```bash
python scripts/comprehensive_evaluation.py \
    --image_folder YOUR_IMAGE_FOLDER \
    --max_images 5
```

---

### Q: 脚本运行很慢？

**正常情况**:
- 5张图像: 5-10分钟
- 10张图像: 10-15分钟
- 50张图像: 30-60分钟

**加速方法**:
1. 使用更强的GPU
2. 减少测试图像数量
3. 只测试关键方案（需要修改代码）

---

### Q: 如何只测试某些量化方案？

**修改方法**:

编辑 `vggt/quantization/comprehensive_quantizer.py`:

```python
def get_all_quantization_configs(device: str = "cuda") -> List[QuantizationConfig]:
    configs = [
        # 只保留你想测试的方案
        QuantizationConfig(
            name="INT8_Per_Channel_Symmetric",
            quant_type="int8_per_channel_sym",
            bits=8,
            symmetric=True,
            per_channel=True,
            device=device
        ),
        # 注释掉其他方案...
    ]
    return configs
```

---

## 📝 实验报告模板

### 推荐结构

```markdown
# VGGT量化实验报告

## 1. 实验目的
研究VGGT模型从FP32到INT8/INT4的量化效果

## 2. 实验设置
- 模型: facebook/VGGT-1B
- 测试图像: 10张 (ETH3D courtyard)
- 设备: NVIDIA RTX 3090
- 日期: 2025-10-16

## 3. 量化方案
测试了8种方案（详见附件表格）

## 4. 实验结果
### 4.1 整体对比
[插入 comprehensive_visualizations.png]

### 4.2 详细数据
[粘贴 comprehensive_report.txt 的表格]

## 5. 结果分析
### 5.1 最佳压缩率
INT4_Group_128 达到8倍压缩...

### 5.2 最佳精度
INT8_Per_Channel_Asymmetric 精度最高...

### 5.3 综合推荐
**生产环境**: INT8_Per_Channel_Symmetric
- 理由: 平衡精度和性能...

## 6. 结论
通过完整的量化实验，我们发现...

## 7. 未来工作
- 量化感知训练(QAT)
- 混合精度量化
- 实际部署测试
```

---

## 🚀 下一步

完成快速实验后，你可以：

1. **深入分析**: 阅读 `COMPREHENSIVE_QUANTIZATION_GUIDE.md`
2. **优化参数**: 调整量化配置
3. **扩展实验**: 测试更多数据集
4. **撰写报告**: 使用上面的模板

---

## 📞 获取帮助

如遇到问题：

1. 检查本文档的"常见问题"部分
2. 阅读完整指南 `COMPREHENSIVE_QUANTIZATION_GUIDE.md`
3. 查看错误日志
4. 联系项目维护者

---

**祝实验顺利！** 🎉

开始你的第一个实验：

```bash
cd C:\Users\Ava Ai\Desktop\8539Project\code\vggt
python scripts/comprehensive_evaluation.py --image_folder data/eth3d/courtyard/images --max_images 5 --output_dir results/my_first_test
```
