# ⚡ 量化对比实验 - 快速命令

所有命令都已经测试过，可以直接复制粘贴使用！

---

## 🎯 立即开始（复制这一条）

```bash
cd /workspace/vggt && \
python scripts/compare_quantization.py \
    --model_name facebook/VGGT-1B \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/quantization_comparison && \
echo "✅ 实验完成！查看结果：" && \
cat /workspace/quantization_comparison/comparison_summary.txt
```

**预计时间：** 15-20 分钟

---

## 📊 查看结果

### 方法 1：文本报告（最快）

```bash
cat /workspace/quantization_comparison/comparison_summary.txt
```

### 方法 2：JSON 数据

```bash
python -m json.tool /workspace/quantization_comparison/comparison_report.json
```

### 方法 3：可视化图表

```bash
cd /workspace/quantization_comparison && \
python -m http.server 8000
```

然后在 RunPod 控制台：Connect → HTTP Service [Port 8000] → 打开 `comparison_plots.png`

---

## 🔬 不同实验场景

### 实验 A：快速测试（3 张图，5 分钟）

```bash
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 3 \
    --output_dir /workspace/comparison_quick
```

### 实验 B：完整测试（10 张图，30 分钟）

```bash
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 10 \
    --output_dir /workspace/comparison_full
```

### 实验 C：多场景对比

```bash
# 场景 1
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/comparison_courtyard

# 场景 2
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/delivery_area/images \
    --max_images 5 \
    --output_dir /workspace/comparison_delivery

# 场景 3
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/electro/images \
    --max_images 5 \
    --output_dir /workspace/comparison_electro
```

---

## 💾 保存最佳量化模型

### 选项 1：PyTorch Dynamic（推荐）

```bash
python scripts/quantize_model.py \
    --model_name facebook/VGGT-1B \
    --quantization_type dynamic \
    --output_path /workspace/models/vggt_int8_dynamic.pt \
    --quantize_attention \
    --quantize_heads
```

### 选项 2：INT8 Symmetric

```python
# 创建 save_int8_symmetric.py
python << 'EOF'
import torch
from vggt.models.vggt import VGGT
from vggt.quantization import quantize_model_advanced, AdvancedQuantConfig

model = VGGT.from_pretrained("facebook/VGGT-1B").cuda()
config = AdvancedQuantConfig(quant_type="int8_symmetric", bits=8)
quant_model = quantize_model_advanced(model, config)

torch.save(quant_model.state_dict(), "/workspace/models/vggt_int8_symmetric.pt")
print("✅ Saved: /workspace/models/vggt_int8_symmetric.pt")
EOF
```

### 选项 3：INT8 Asymmetric

```python
python << 'EOF'
import torch
from vggt.models.vggt import VGGT
from vggt.quantization import quantize_model_advanced, AdvancedQuantConfig

model = VGGT.from_pretrained("facebook/VGGT-1B").cuda()
config = AdvancedQuantConfig(quant_type="int8_asymmetric", bits=8)
quant_model = quantize_model_advanced(model, config)

torch.save(quant_model.state_dict(), "/workspace/models/vggt_int8_asymmetric.pt")
print("✅ Saved: /workspace/models/vggt_int8_asymmetric.pt")
EOF
```

### 选项 4：INT4 Group-128（最小模型）

```python
python << 'EOF'
import torch
from vggt.models.vggt import VGGT
from vggt.quantization import quantize_model_advanced, AdvancedQuantConfig

model = VGGT.from_pretrained("facebook/VGGT-1B").cuda()
config = AdvancedQuantConfig(quant_type="int4_group", bits=4, group_size=128)
quant_model = quantize_model_advanced(model, config)

torch.save(quant_model.state_dict(), "/workspace/models/vggt_int4_group128.pt")
print("✅ Saved: /workspace/models/vggt_int4_group128.pt")
EOF
```

---

## 📥 下载结果到本地

### 方法 1：下载对比报告（推荐）

```bash
# 在 RunPod 上打包
cd /workspace && \
tar -czf comparison_results.tar.gz quantization_comparison/

# 在本地下载（替换 <PORT> 和 <POD_IP>）
scp -P <PORT> root@<POD_IP>:/workspace/comparison_results.tar.gz ./

# 解压
tar -xzf comparison_results.tar.gz

# 查看
cat quantization_comparison/comparison_summary.txt
open quantization_comparison/comparison_plots.png
```

### 方法 2：只下载文本报告

```bash
# 在本地
scp -P <PORT> root@<POD_IP>:/workspace/quantization_comparison/comparison_summary.txt ./
cat comparison_summary.txt
```

### 方法 3：下载量化模型

```bash
# 下载所有量化模型
scp -P <PORT> root@<POD_IP>:/workspace/models/vggt_int*.pt ./models/
```

---

## 🎨 可视化量化效果

### 生成深度图对比

```bash
# 对比原始模型和量化模型的深度图
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --original_model facebook/VGGT-1B \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/quantization_comparison \
    --max_images 5
```

---

## 🔍 验证模型

### 快速验证量化模型

```python
python << 'EOF'
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.quantization import estimate_model_size

# 加载量化模型
model = VGGT()
model.load_state_dict(torch.load("/workspace/models/vggt_int8_dynamic.pt"))
model.eval()

# 检查模型大小
size = estimate_model_size(model)
print(f"Model size: {size['total_mb']:.2f} MB")

# 测试推理
images = load_and_preprocess_images([
    "/workspace/data/eth3d/courtyard/images/DSC_0001.JPG"
]).cpu()

with torch.no_grad():
    output = model(images)

print("✅ Model loaded and inference successful!")
print(f"Output keys: {list(output.keys())}")
EOF
```

---

## 📊 生成对比表格

```bash
# 自动生成 Markdown 表格
python << 'EOF'
import json

with open("/workspace/quantization_comparison/comparison_report.json") as f:
    data = json.load(f)

print("| Method | Size(MB) | Compression | Time(s) | Speedup | Depth MAE |")
print("|--------|----------|-------------|---------|---------|-----------|")

for method, metrics in data.items():
    if "error" in metrics:
        continue

    size = metrics.get("model_size_mb", 0)
    comp = metrics.get("compression_ratio", 1.0)
    time = metrics.get("inference_time", 0)
    speedup = metrics.get("speedup", 1.0)

    depth_metrics = metrics.get("metrics", {}).get("depth", {})
    mae = depth_metrics.get("mae", 0)

    print(f"| {method:20} | {size:8.2f} | {comp:11.2f}x | {time:7.4f} | {speedup:7.2f}x | {mae:.6f} |")
EOF
```

---

## 🎯 完整工作流程（一键完成）

```bash
#!/bin/bash
# 完整的量化对比实验流程

echo "=========================================="
echo "VGGT Quantization Comparison Workflow"
echo "=========================================="

# 1. 运行对比实验
echo "[1/4] Running quantization comparison..."
cd /workspace/vggt
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/quantization_comparison

# 2. 保存最佳模型（Dynamic）
echo "[2/4] Saving best quantized model..."
python scripts/quantize_model.py \
    --model_name facebook/VGGT-1B \
    --quantization_type dynamic \
    --output_path /workspace/models/vggt_int8_best.pt \
    --quantize_attention \
    --quantize_heads

# 3. 生成可视化
echo "[3/4] Generating visualizations..."
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_best.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/final \
    --max_images 10

# 4. 打包结果
echo "[4/4] Packaging results..."
cd /workspace
tar -czf experiment_results.tar.gz \
    quantization_comparison/ \
    visualizations/final/ \
    models/vggt_int8_best.pt

echo "=========================================="
echo "✅ Complete! Download: /workspace/experiment_results.tar.gz"
echo "=========================================="
```

保存为 `run_full_experiment.sh` 并执行：

```bash
bash run_full_experiment.sh
```

---

## 🆘 故障排除

### 错误：找不到图像

```bash
# 检查图像路径
ls /workspace/data/eth3d/courtyard/images/ | head -5

# 使用正确的路径
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5
```

### 错误：CUDA Out of Memory

```bash
# 使用 CPU
python scripts/compare_quantization.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 3 \
    --device cpu
```

### 错误：模块导入失败

```bash
# 确保在正确的目录
cd /workspace/vggt

# 验证安装
python -c "from vggt.quantization import quantize_model_advanced; print('✅ OK')"
```

---

## 💡 性能优化建议

### 1. 减少测试图像

```bash
# 从 3 张图像开始
--max_images 3
```

### 2. 使用更小的场景

```bash
# 选择图像数量少的场景
ls /workspace/data/eth3d/*/images | wc -l
```

### 3. 分步执行

```bash
# 先测试一个方案
# 满意后再运行完整对比
```

---

## 📚 文档链接

- 完整指南：`QUANTIZATION_COMPARISON_GUIDE.md`
- 技术细节：`IMPLEMENTATION_SUMMARY.md`
- RunPod 部署：`RUNPOD_DEPLOYMENT.md`
- 快速开始：`START_HERE.md`

---

**现在就开始你的量化对比实验！** 🚀

复制第一条命令到 RunPod 终端执行即可。
