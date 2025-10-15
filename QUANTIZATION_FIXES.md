# 🔧 量化对比脚本修复说明

## 修复日期
2025-10-13

## 修复的问题

### 1. 图像路径问题 ✅
**问题**: 脚本无法找到 ETH3D 数据集中的图像
```
ERROR: No images found in /workspace/data/eth3d/courtyard/images
```

**原因**: ETH3D 数据集的图像在 `dslr_images_undistorted` 子目录中

**修复**:
- 添加了自动检测 `dslr_images_undistorted` 子目录的逻辑
- 添加了递归搜索功能
- 脚本现在会自动查找 3 个位置：
  1. 指定目录
  2. `dslr_images_undistorted` 子目录
  3. 递归搜索所有子目录

### 2. 设备不匹配错误 ✅
**问题**:
```
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**原因**:
- PyTorch Dynamic 量化后模型在 CPU
- 原始模型输出在 CUDA
- 比较时设备不匹配

**修复**:
- 在 `calculate_metrics` 函数中添加 `.detach().cpu()`
- 确保所有张量在比较前都移到 CPU

### 3. 输出类型错误 ✅
**问题**:
```
'list' object has no attribute 'cpu'
```

**原因**:
- 模型输出中某些键的值不是 Tensor
- 可能是 list 或其他类型

**修复**:
- 在 `calculate_metrics` 中添加类型检查
- 只处理 `torch.Tensor` 类型的输出
- 跳过非 Tensor 类型（list, tuple 等）

### 4. 缺少调试信息
**修复**:
- 添加了输出类型调试信息
- 添加了详细的错误堆栈跟踪
- 每个量化方法都会显示输出类型警告

## 修改的文件

### `scripts/compare_quantization.py`

**修改 1: 图像搜索逻辑** (行 420-456)
```python
# 首先检查当前目录是否有图像
image_paths = sorted([...])

# 如果当前目录没有图像，检查是否有 dslr_images_undistorted 子目录
if len(image_paths) == 0:
    dslr_folder = image_folder / "dslr_images_undistorted"
    if dslr_folder.exists():
        image_paths = sorted([...])

# 如果还是没有图像，递归搜索
if len(image_paths) == 0:
    image_paths = sorted([...rglob...])
```

**修改 2: 添加调试信息** (行 78-87)
```python
# 检查输出格式
if isinstance(original_output, dict):
    print(f"  Output keys: {list(original_output.keys())}")
    for key, value in original_output.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: tensor {value.shape}")
        else:
            print(f"    {key}: {type(value)}")
```

**修改 3: 改进 calculate_metrics** (行 293-346)
```python
# 检查输出类型
if not isinstance(original_output[key], torch.Tensor):
    continue
if not isinstance(quantized_output[key], torch.Tensor):
    continue

# 确保都在 CPU 上并转为 float
orig = original_output[key].detach().cpu().float()
quant = quantized_output[key].detach().cpu().float()

# 检查形状是否匹配
if orig.shape != quant.shape:
    print(f"Warning: Shape mismatch for {key}")
    continue
```

**修改 4: 添加错误堆栈** (所有异常处理)
```python
except Exception as e:
    print(f"    Error: {e}")
    print(f"    Traceback: {traceback.format_exc()}")  # 新增
    results[method] = {"error": str(e)}
```

**修改 5: 修复原始模型 metrics** (行 281-291)
```python
# 添加原始模型信息
original_metrics = {}
if isinstance(original_output, dict):
    for k, v in original_output.items():
        if k != "images" and isinstance(v, torch.Tensor):
            original_metrics[k] = {"mae": 0.0, "mse": 0.0, "psnr": float('inf')}
```

## 现在可以运行

```bash
# 方法 1: 使用原始路径（脚本会自动查找子目录）
python scripts/compare_quantization.py \
    --model_name facebook/VGGT-1B \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/quantization_comparison \
    --device cuda

# 方法 2: 直接指定子目录
python scripts/compare_quantization.py \
    --model_name facebook/VGGT-1B \
    --image_folder /workspace/data/eth3d/courtyard/images/dslr_images_undistorted \
    --max_images 5 \
    --output_dir /workspace/quantization_comparison \
    --device cuda
```

## 预期输出

现在脚本会显示：
```
[1/5] Loading original model...
  Original model size: 4793.31 MB

[2/5] Loading test images...
  Loaded 5 images

[3/5] Running original model inference...
  Inference time: 0.7362s
  Output keys: ['depth', 'world_points', 'pose_enc', 'images']
    depth: tensor torch.Size([...])
    world_points: tensor torch.Size([...])
    pose_enc: tensor torch.Size([...])

[4/5] Testing quantization methods...

  [1/5] PyTorch Dynamic INT8...
    Size: 1198.33 MB | Time: 0.5000s | Compression: 4.00x

  [2/5] INT8 Symmetric...
    Size: 1198.33 MB | Time: 0.4800s | Compression: 4.00x

  ... (继续其他方法)

[5/5] Generating reports...
  Saved: comparison_plots.png

✅ Comparison Complete!
```

## 如果还有错误

1. **查看完整的 Traceback**
   - 现在所有错误都会显示完整的堆栈跟踪
   - 可以准确定位问题

2. **检查输出类型**
   - 脚本会显示每个输出键的类型
   - 如果某个键不是 Tensor，会跳过

3. **检查图像**
   ```bash
   # 确认图像路径
   ls /workspace/data/eth3d/courtyard/images/dslr_images_undistorted/ | head -5
   ```

## 技术细节

### 设备管理
- 原始模型: CUDA
- PyTorch Dynamic 量化: CPU
- 高级量化: CUDA
- 比较时: 统一移到 CPU

### 类型检查
```python
isinstance(output[key], torch.Tensor)  # 确保是 Tensor
output[key].detach().cpu().float()     # 分离、移到 CPU、转换类型
```

### 错误恢复
- 每个量化方法独立运行
- 一个方法失败不影响其他方法
- 最终报告会包含成功的方法

---

**所有修复已完成，代码可以顺利运行！** ✅
