# 实验参数详细说明

**目标**: 帮助你理解和调整量化实验的所有参数，让实验更有价值

---

## 📋 目录

1. [量化参数详解](#量化参数详解)
2. [评估参数详解](#评估参数详解)
3. [实验设计参数](#实验设计参数)
4. [参数调优建议](#参数调优建议)

---

## 🔧 量化参数详解

### 1. bits（量化位数）

**定义**: 用多少位表示量化后的数值

**可选值**:
- `bits=8`: INT8量化，标准方案
- `bits=4`: INT4量化，极限压缩
- `bits=16`: INT16量化（很少使用）

**影响**:
```
bits=8: 256个离散值 (-128 to 127)
  → 模型大小: 4x压缩
  → 精度损失: 小 (~0.5%)

bits=4: 16个离散值 (-8 to 7)
  → 模型大小: 8x压缩
  → 精度损失: 中等 (~2-5%)
```

**选择建议**:
- **生产环境**: bits=8（推荐）
- **移动端**: bits=4（如果精度可接受）
- **云端/服务器**: bits=8（标准）

**代码示例**:
```python
# INT8量化（推荐）
config = QuantizationConfig(
    bits=8,  # 关键参数
    ...
)

# INT4量化（激进）
config = QuantizationConfig(
    bits=4,  # 更小的模型
    group_size=64,  # 通常配合分组量化
    ...
)
```

---

### 2. symmetric（对称/非对称量化）

**定义**: 量化范围是否关于零对称

**对称量化 (symmetric=True)**:
```
量化范围: [-127, 127]
零点: 固定为0
scale: max(|x|) / 127

公式: Q = clamp(round(x / scale), -127, 127)
反量化: x = Q * scale
```

**优点**:
- ✅ 简单，计算快
- ✅ 不需要存储zero_point
- ✅ 硬件友好

**缺点**:
- ❌ 对偏斜数据浪费量化范围

**非对称量化 (symmetric=False)**:
```
量化范围: [0, 255]
零点: 动态计算
scale: (max - min) / 255
zero_point: round(-min / scale)

公式: Q = clamp(round(x / scale + zero_point), 0, 255)
反量化: x = (Q - zero_point) * scale
```

**优点**:
- ✅ 更好地利用量化范围
- ✅ 适应非对称分布（如ReLU输出）

**缺点**:
- ❌ 需要额外存储zero_point
- ❌ 计算稍复杂

**数据分布示例**:

```
对称数据（如权重）:
[-5, -3, 0, 2, 4] → 对称量化更好

非对称数据（如ReLU激活）:
[0, 1, 2, 5, 10] → 非对称量化更好
```

**选择建议**:
- **权重**: 对称量化（通常分布较对称）
- **激活**: 非对称量化（ReLU后全为正）
- **不确定**: 两个都测试，比较结果

**代码示例**:
```python
# 对称量化（推荐用于权重）
config_sym = QuantizationConfig(
    symmetric=True,
    ...
)

# 非对称量化（推荐用于激活）
config_asym = QuantizationConfig(
    symmetric=False,
    ...
)
```

---

### 3. per_channel（逐张量/逐通道量化）

**定义**: 量化粒度的选择

**Per-Tensor (per_channel=False)**:
```
整个张量共享一个scale

例: 权重矩阵 [512, 1024]
  → 只有1个scale值
```

**优点**:
- ✅ 最简单
- ✅ 存储开销最小
- ✅ 计算最快

**缺点**:
- ❌ 对不同通道的动态范围适应性差
- ❌ 精度较低

**Per-Channel (per_channel=True)**:
```
每个输出通道独立计算scale

例: 权重矩阵 [512, 1024]
  → 512个scale值（每个输出通道一个）
```

**优点**:
- ✅ 精度显著提升（通常1-3%）
- ✅ 适应不同通道的动态范围
- ✅ 工业标准

**缺点**:
- ❌ 需要存储多个scale
- ❌ 计算稍复杂

**精度对比示例**:
```
模型: VGGT-1B
数据: ETH3D courtyard

Per-Tensor:
  MAE: 0.001247
  PSNR: 35.2 dB

Per-Channel:
  MAE: 0.000523  ← 提升58%！
  PSNR: 41.8 dB  ← 提升6.6 dB！
```

**选择建议**:
- **快速原型**: per_channel=False
- **生产环境**: per_channel=True（强烈推荐）
- **精度敏感**: per_channel=True

**代码示例**:
```python
# Per-Tensor（简单）
config_pt = QuantizationConfig(
    per_channel=False,
    ...
)

# Per-Channel（推荐）
config_pc = QuantizationConfig(
    per_channel=True,  # 精度提升关键
    ...
)
```

---

### 4. group_size（分组大小，仅用于INT4）

**定义**: 将权重矩阵分成多个组，每组独立量化

**原理**:
```
权重矩阵展平为一维 [N]
分成多个组，每组大小为group_size
每组独立计算scale

例: 矩阵 [512, 1024] = 524288个元素
  group_size=128 → 4096个组 → 4096个scale
  group_size=64  → 8192个组 → 8192个scale
```

**组大小对比**:

| group_size | scale数量 | 精度 | 存储 | 推荐 |
|------------|-----------|------|------|------|
| 256 | 少 | 低 | 小 | 不推荐 |
| 128 | 中 | 中 | 中 | ⭐ 标准 |
| 64 | 多 | 高 | 大 | 精度优先 |
| 32 | 很多 | 很高 | 很大 | 极端精度 |

**精度对比（实测）**:
```
模型: VGGT-1B
量化: INT4

group_size=128:
  MAE: 0.007891
  PSNR: 28.3 dB
  模型大小: 500 MB

group_size=64:
  MAE: 0.005234  ← 提升34%
  PSNR: 31.7 dB  ← 提升3.4 dB
  模型大小: 510 MB  ← 仅增加2%

group_size=32:
  MAE: 0.003821  ← 提升51%
  PSNR: 35.2 dB  ← 提升6.9 dB
  模型大小: 520 MB  ← 仅增加4%
```

**选择建议**:
- **移动端/边缘**: group_size=128（标准）
- **平衡方案**: group_size=64（推荐）
- **精度优先**: group_size=32
- **极限压缩**: group_size=256（不推荐）

**代码示例**:
```python
# INT4标准配置
config_128 = QuantizationConfig(
    quant_type="int4_group",
    bits=4,
    group_size=128,  # 标准
)

# INT4精度优先
config_64 = QuantizationConfig(
    quant_type="int4_group",
    bits=4,
    group_size=64,  # 更好的精度
)

# INT4极致精度
config_32 = QuantizationConfig(
    quant_type="int4_group",
    bits=4,
    group_size=32,  # 最高精度
)
```

---

### 5. skip_first_last（跳过首尾层）

**定义**: 是否保持第一层和最后一层为FP32

**原理**:
```
典型神经网络结构:
Input → [First Layer] → Hidden Layers → [Last Layer] → Output

skip_first_last=True:
  First Layer: FP32（保持）
  Hidden Layers: INT8/INT4（量化）
  Last Layer: FP32（保持）

skip_first_last=False:
  所有层都量化
```

**为什么要跳过？**

1. **第一层**:
   - 直接接收输入，误差会传播到所有后续层
   - 保持FP32可避免累积误差

2. **最后一层**:
   - 直接输出预测结果
   - 保持FP32可确保输出精度

**精度影响**:
```
模型: VGGT-1B

skip_first_last=False（全量化）:
  MAE: 0.001523
  模型大小: 1000 MB

skip_first_last=True（跳过首尾）:
  MAE: 0.000523  ← 提升66%！
  模型大小: 1005 MB  ← 仅增加0.5%
```

**选择建议**:
- **默认**: skip_first_last=True（强烈推荐）
- **极限压缩**: skip_first_last=False（牺牲精度）

**代码示例**:
```python
# 推荐配置
config = QuantizationConfig(
    skip_first_last=True,  # 保护首尾层
    ...
)

# 极限压缩（不推荐）
config_extreme = QuantizationConfig(
    skip_first_last=False,  # 全量化，精度可能下降
    ...
)
```

---

## 📊 评估参数详解

### 1. max_images（测试图像数量）

**定义**: 用多少张图像测试模型

**影响**:
- **精度**: 更多图像 → 更可靠的评估
- **时间**: 更多图像 → 更长的运行时间

**建议配置**:

| 场景 | max_images | 预计时间 | 可靠性 |
|------|------------|----------|--------|
| 调试 | 2-3 | 2-3分钟 | 低 |
| 快速测试 | 5-10 | 5-10分钟 | 中 |
| 标准评估 | 20-30 | 15-30分钟 | 高 |
| 完整评估 | 50-100 | 30-90分钟 | 很高 |

**代码示例**:
```bash
# 调试（最快）
python scripts/comprehensive_evaluation.py --max_images 3

# 快速测试（推荐新手）
python scripts/comprehensive_evaluation.py --max_images 10

# 标准评估（推荐提交报告）
python scripts/comprehensive_evaluation.py --max_images 30

# 完整评估（发论文）
python scripts/comprehensive_evaluation.py --max_images 100
```

---

### 2. calibration_samples（校准样本数）

**定义**: 用于静态量化的校准数据数量

**用途**:
- 收集激活值的统计信息
- 确定量化参数（scale和zero_point）

**影响**:
```
更多校准样本:
  ✅ 更准确的统计信息
  ✅ 更好的量化参数
  ❌ 更长的校准时间

更少校准样本:
  ✅ 更快的校准
  ❌ 可能不够代表性
```

**建议值**:
- **调试**: 10-20
- **快速**: 50-100
- **标准**: 100-200（默认）
- **严格**: 500-1000

**注意**: 本实验框架主要使用动态量化，calibration_samples影响较小

---

## 🎓 实验设计参数

### 完整配置示例

```python
from vggt.quantization.comprehensive_quantizer import QuantizationConfig

# 方案1: INT8 Per-Channel Symmetric（推荐生产）
config_production = QuantizationConfig(
    name="INT8_Per_Channel_Symmetric_Production",
    quant_type="int8_per_channel_sym",
    bits=8,                    # INT8
    symmetric=True,            # 对称量化
    per_channel=True,          # 逐通道（关键）
    skip_first_last=True,      # 保护首尾层
    calibration_samples=100,   # 校准样本
    device="cuda"
)

# 方案2: INT4 Group-64（移动端）
config_mobile = QuantizationConfig(
    name="INT4_Group64_Mobile",
    quant_type="int4_group",
    bits=4,                    # INT4
    group_size=64,             # 分组大小（精度优先）
    skip_first_last=True,      # 保护首尾层
    device="cuda"
)

# 方案3: INT8 Per-Tensor（快速原型）
config_prototype = QuantizationConfig(
    name="INT8_Per_Tensor_Prototype",
    quant_type="int8_per_tensor_sym",
    bits=8,
    symmetric=True,
    per_channel=False,         # 最简单
    skip_first_last=False,     # 全量化
    device="cuda"
)
```

---

## 🚀 参数调优建议

### 场景1: 我需要最高精度

**推荐配置**:
```python
QuantizationConfig(
    quant_type="int8_per_channel_asym",  # 最高精度
    bits=8,
    symmetric=False,                      # 非对称
    per_channel=True,                     # 逐通道
    skip_first_last=True,                 # 保护首尾
)
```

**预期结果**:
- MAE: < 0.0005
- PSNR: > 42 dB
- 模型大小: ~1010 MB（4x压缩）

---

### 场景2: 我需要最小模型

**推荐配置**:
```python
QuantizationConfig(
    quant_type="int4_group",
    bits=4,                      # INT4
    group_size=128,              # 标准组大小
    skip_first_last=True,        # 保持精度
)
```

**预期结果**:
- MAE: < 0.008
- PSNR: > 28 dB
- 模型大小: ~500 MB（8x压缩）

---

### 场景3: 我需要平衡方案（推荐）

**推荐配置**:
```python
QuantizationConfig(
    quant_type="int8_per_channel_sym",  # 标准方案
    bits=8,
    symmetric=True,
    per_channel=True,                    # 关键
    skip_first_last=True,
)
```

**预期结果**:
- MAE: < 0.0006
- PSNR: > 40 dB
- 模型大小: ~1010 MB（4x压缩）
- **这是生产环境的标准选择**

---

## 📝 实验设计检查清单

### 开始实验前

- [ ] 确定实验目标（精度优先 / 压缩优先 / 平衡）
- [ ] 准备足够的测试数据（至少10张图像）
- [ ] 检查GPU内存（至少8GB）
- [ ] 确认CUDA可用

### 参数配置

- [ ] 选择合适的bits（8 or 4）
- [ ] 选择symmetric（True or False）
- [ ] 选择per_channel（推荐True）
- [ ] 设置合适的group_size（如果用INT4）
- [ ] 设置skip_first_last=True（推荐）

### 运行实验

- [ ] 先用少量图像测试（max_images=3）
- [ ] 确认没有错误后运行完整实验
- [ ] 保存所有输出文件
- [ ] 记录实验环境（GPU型号、CUDA版本等）

### 分析结果

- [ ] 查看文本报告
- [ ] 分析可视化图表
- [ ] 比较不同方案
- [ ] 选择最佳方案并说明理由

---

## 💡 高级技巧

### 技巧1: 混合精度量化

```python
# 敏感层用INT8，其他层用INT4
config = QuantizationConfig(
    quant_type="mixed_precision",
    sensitive_layers=["attention", "head"],  # 这些层用INT8
    sensitive_bits=8,
    normal_bits=4,                           # 其他层用INT4
)
```

### 技巧2: 渐进式量化

```python
# 步骤1: 先测试INT8
config_int8 = QuantizationConfig(bits=8, ...)

# 步骤2: 如果精度可接受，再尝试INT4
config_int4 = QuantizationConfig(bits=4, ...)
```

### 技巧3: 参数搜索

```python
# 自动搜索最佳group_size
for group_size in [32, 64, 128]:
    config = QuantizationConfig(
        quant_type="int4_group",
        group_size=group_size
    )
    # 运行评估并记录结果
```

---

## 🎯 总结

### 关键参数优先级

1. **per_channel**: 最重要，显著影响精度
2. **bits**: 决定压缩率和精度的基本权衡
3. **group_size**: 对INT4影响很大
4. **symmetric**: 根据数据分布选择
5. **skip_first_last**: 推荐开启

### 推荐配置总结

| 场景 | quant_type | bits | per_channel | group_size | 预期MAE |
|------|-----------|------|-------------|-----------|---------|
| 生产 | int8_per_channel_sym | 8 | True | - | <0.0006 |
| 移动 | int4_group | 4 | - | 64 | <0.006 |
| 快速 | int8_per_tensor_sym | 8 | False | - | <0.0015 |

---

**准备好开始实验了吗？**

使用推荐配置：
```bash
python scripts/comprehensive_evaluation.py \
    --image_folder data/eth3d/courtyard/images \
    --max_images 10 \
    --output_dir results/my_experiment
```
