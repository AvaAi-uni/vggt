# ⚡ 快速可视化命令参考

**已完成量化？现在立即查看结果！**

---

## 🎨 在 RunPod 上生成可视化（1 分钟）

### 复制这一条命令即可：

```bash
cd /workspace/vggt && \
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/results \
    --max_images 10
```

**完成！** 可视化已生成在 `/workspace/visualizations/results/`

---

## 👀 在 RunPod 上查看 HTML 报告

### 方法 1：使用 HTTP 服务器

```bash
cd /workspace/visualizations/results && \
python -m http.server 8000
```

然后：
1. 在 RunPod 控制台，点击 "Connect" → "HTTP Service [Port 8000]"
2. 在浏览器中打开 `index.html`

### 方法 2：查看文件列表

```bash
ls -lh /workspace/visualizations/results/
```

---

## 📥 下载到本地电脑

### 步骤 1：在 RunPod 上打包

```bash
cd /workspace && \
tar -czf viz_results.tar.gz visualizations/
```

### 步骤 2：在本地电脑上下载

**在你的本地电脑终端中运行：**

```bash
# 替换 <PORT> 和 <POD_IP> 为你的 RunPod 信息
# （从 RunPod 控制台 → Connect → TCP Port Mappings 获取）

scp -P <PORT> root@<POD_IP>:/workspace/viz_results.tar.gz ./
```

### 步骤 3：解压并查看

```bash
# 解压
tar -xzf viz_results.tar.gz

# 打开 HTML 报告
# macOS:
open visualizations/results/index.html

# Linux:
xdg-open visualizations/results/index.html

# Windows:
start visualizations/results/index.html
```

---

## 🔍 对比动态量化 vs 静态量化

```bash
# 生成动态量化可视化
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/dynamic \
    --max_images 10

# 生成静态量化可视化
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_static.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/static \
    --max_images 10

# 查看对比
cat /workspace/visualizations/dynamic/metadata.json
cat /workspace/visualizations/static/metadata.json
```

---

## 🎯 对比原始模型和量化模型

```bash
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --original_model facebook/VGGT-1B \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/comparison \
    --max_images 5
```

这会生成精度对比图和 MAE 指标。

---

## 📊 快速查看元数据

```bash
# 查看推理性能
cat /workspace/visualizations/results/metadata.json

# 或格式化输出
python -m json.tool /workspace/visualizations/results/metadata.json
```

---

## 🖼️ 使用 VGGT 自带的可视化工具

### Gradio Web 界面（推荐）

```bash
cd /workspace/vggt
python demo_gradio.py --share
```

会生成一个公共 URL，在浏览器中打开即可交互式使用。

### Viser 3D 查看器

```bash
python demo_viser.py --image_folder /workspace/data/eth3d/courtyard/images
```

### 导出为 COLMAP 格式

```bash
python demo_colmap.py --scene_dir /workspace/data/eth3d/courtyard
```

---

## 📦 一键完整流程

### 生成、打包、准备下载：

```bash
cd /workspace/vggt && \
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/results \
    --max_images 20 && \
cd /workspace && \
tar -czf viz_results.tar.gz visualizations/ && \
echo "✅ 完成！现在可以下载 /workspace/viz_results.tar.gz"
```

---

## 🆘 问题排查

### 检查文件是否存在

```bash
# 检查模型
ls -lh /workspace/models/vggt_int8_*.pt

# 检查数据
ls /workspace/data/eth3d/courtyard/images/ | head -5

# 检查可视化
ls -lh /workspace/visualizations/results/
```

### 查看完整错误信息

```bash
# 运行时不要使用 -q 参数，看完整输出
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/debug
```

---

## 💡 高级选项

### 只生成深度图

```bash
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/depth_only \
    --skip_points --skip_camera
```

### 处理更多图像

```bash
# 处理 50 张图像（需要更长时间）
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/large \
    --max_images 50
```

### 使用不同场景

```bash
# 查看所有可用场景
ls /workspace/data/eth3d/

# 使用不同场景
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/delivery_area/images \
    --output_dir /workspace/visualizations/delivery_area
```

---

## 🎓 推荐工作流程

1. **快速测试**（5 张图）
   ```bash
   python scripts/visualize_results.py \
       --model_path /workspace/models/vggt_int8_dynamic.pt \
       --image_folder /workspace/data/eth3d/courtyard/images \
       --output_dir /workspace/visualizations/test \
       --max_images 5
   ```

2. **查看 HTML 报告**
   ```bash
   cd /workspace/visualizations/test
   python -m http.server 8000
   ```

3. **如果满意，处理更多图像**
   ```bash
   python scripts/visualize_results.py \
       --model_path /workspace/models/vggt_int8_dynamic.pt \
       --image_folder /workspace/data/eth3d/courtyard/images \
       --output_dir /workspace/visualizations/full \
       --max_images 20
   ```

4. **下载到本地**
   ```bash
   # 在 RunPod 上打包
   tar -czf /workspace/viz_full.tar.gz /workspace/visualizations/full

   # 在本地下载
   scp -P <PORT> root@<POD_IP>:/workspace/viz_full.tar.gz ./
   ```

---

## 📱 本地查看工具推荐

### HTML 报告
- 任何现代浏览器（Chrome, Firefox, Safari, Edge）

### 点云 PLY 文件
- **MeshLab**: https://www.meshlab.net/ (免费)
- **CloudCompare**: https://www.cloudcompare.org/ (免费)
- **Blender**: https://www.blender.org/ (免费)
- **在线查看**: https://3dviewer.net/

### 图像文件
- 任何图像查看器

---

## ⏱️ 预计时间

| 任务 | 时间 |
|------|------|
| 生成可视化（10 张图）| 1-2 分钟 |
| 生成可视化（20 张图）| 3-5 分钟 |
| 打包结果 | 10-30 秒 |
| 下载到本地（取决于网速）| 1-5 分钟 |
| 查看 HTML 报告 | 即时 |

---

## 🎉 完成后你将拥有

- ✅ 深度图可视化（彩色编码）
- ✅ 深度图置信度热力图
- ✅ 3D 点云（3 个视角的 2D 投影）
- ✅ 点云 PLY 文件（可在专业软件中查看）
- ✅ 相机轨迹 3D 可视化
- ✅ 交互式 HTML 报告
- ✅ 性能指标（推理时间、FPS）
- ✅ 模型对比（如果生成了）
- ✅ 完整的元数据 JSON

---

**现在就开始可视化你的结果！** 🚀

选择上面任意一个命令复制执行即可。

---

**详细文档**: [DOWNLOAD_AND_VISUALIZE.md](DOWNLOAD_AND_VISUALIZE.md)
