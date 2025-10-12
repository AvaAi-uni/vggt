# 📥 下载结果和可视化指南

恭喜你完成了量化！本指南将帮你：
1. ✅ 在 RunPod 上查看和可视化结果
2. 📥 下载结果到本地电脑
3. 🎨 生成各种可视化（深度图、点云、相机轨迹等）
4. 📊 生成 HTML 交互式报告

---

## 🎨 第一步：在 RunPod 上生成可视化

### 方法 A：使用可视化脚本（推荐，最全面）

```bash
# 在 RunPod 上运行
cd /workspace/vggt

# 可视化动态量化模型的结果
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/dynamic \
    --max_images 10

# 可视化静态量化模型的结果
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_static.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/static \
    --max_images 10
```

这会生成：
- ✅ 深度图可视化（每张图像）
- ✅ 点云 2D 投影（3个视角）
- ✅ 点云 PLY 文件（可用 MeshLab/CloudCompare 打开）
- ✅ 相机轨迹图
- ✅ HTML 交互式报告
- ✅ 元数据 JSON 文件

### 方法 B：对比原始模型和量化模型

```bash
# 下载原始模型（如果还没有）
python -c "from vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')"

# 对比可视化
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --original_model facebook/VGGT-1B \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/comparison \
    --max_images 5
```

这会额外生成：
- ✅ 原始 vs 量化模型的深度图对比
- ✅ 原始 vs 量化模型的点云对比
- ✅ 精度差异热力图
- ✅ MAE（平均绝对误差）指标

---

## 👀 第二步：在 RunPod 上查看结果

### 查看生成的文件

```bash
# 查看所有可视化文件
ls -lh /workspace/visualizations/dynamic/

# 应该看到：
# - quant_depth_b0_s0.png, quant_depth_b0_s1.png, ...  (深度图)
# - quant_pointcloud_b0.png, quant_pointcloud_b0.ply   (点云)
# - quant_camera_trajectory_b0.png                      (相机轨迹)
# - model_comparison.png                                (模型对比，如果有)
# - index.html                                          (HTML 报告)
# - metadata.json                                       (元数据)
```

### 方法 A：使用 Web 浏览器查看（推荐）

#### 选项 1：使用 RunPod 的 HTTP 服务

```bash
# 启动简单的 HTTP 服务器
cd /workspace/visualizations/dynamic
python -m http.server 8000
```

然后：
1. 在 RunPod 控制台，找到你的 Pod
2. 点击 "Connect" → "HTTP Service [Port 8000]"
3. 在浏览器中打开 `index.html`

#### 选项 2：使用 Jupyter Notebook

如果你的 RunPod 有 Jupyter：

```bash
# 启动 Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
```

然后：
1. 在 RunPod 控制台点击 "Connect" → "Jupyter"
2. 导航到 `/workspace/visualizations/dynamic/`
3. 右键点击 `index.html` → "Open With" → "Browser"

### 方法 B：使用命令行查看（快速预览）

```bash
# 查看元数据
cat /workspace/visualizations/dynamic/metadata.json

# 应该显示类似：
# {
#   "model_path": "/workspace/models/vggt_int8_dynamic.pt",
#   "num_images": 10,
#   "inference_time": 0.856,
#   "avg_time_per_image": 0.086,
#   "depth_mae": 0.0123,  # 如果有对比
#   "points_mae": 0.0456  # 如果有对比
# }
```

---

## 📥 第三步：下载结果到本地电脑

### 方法 A：使用 SCP（推荐）

在**你的本地电脑**上打开终端，运行：

```bash
# 获取 RunPod SSH 信息
# 1. 在 RunPod 控制台，点击你的 Pod
# 2. 点击 "Connect" → "TCP Port Mappings"
# 3. 找到 SSH 端口（通常是 22）和对应的外部端口

# 下载整个可视化目录
scp -r -P <外部端口> root@<POD_IP>:/workspace/visualizations ./vggt_visualizations

# 例如：
# scp -r -P 12345 root@123.456.789.0:/workspace/visualizations ./vggt_visualizations
```

**如果需要 SSH 密钥**：

```bash
scp -r -P <外部端口> -i ~/.ssh/id_ed25519 root@<POD_IP>:/workspace/visualizations ./vggt_visualizations
```

### 方法 B：下载单个文件

```bash
# 只下载 HTML 报告
scp -P <外部端口> root@<POD_IP>:/workspace/visualizations/dynamic/index.html ./

# 只下载点云文件
scp -P <外部端口> root@<POD_IP>:/workspace/visualizations/dynamic/*.ply ./pointclouds/

# 只下载元数据
scp -P <外部端口> root@<POD_IP>:/workspace/visualizations/dynamic/metadata.json ./
```

### 方法 C：打包后下载

在 **RunPod 上**：

```bash
# 打包所有结果
cd /workspace
tar -czf visualizations.tar.gz visualizations/

# 查看文件大小
ls -lh visualizations.tar.gz
```

在**本地电脑**上：

```bash
# 下载打包文件
scp -P <外部端口> root@<POD_IP>:/workspace/visualizations.tar.gz ./

# 解压
tar -xzf visualizations.tar.gz
```

### 方法 D：使用 RunPod 的文件管理器

1. 在 RunPod 控制台，点击 "Connect" → "Jupyter"
2. 导航到 `/workspace/visualizations/`
3. 右键点击文件 → "Download"

---

## 🖼️ 第四步：在本地查看结果

### 查看 HTML 报告（推荐）

```bash
# 方法 1：直接双击打开
# 在文件浏览器中找到 index.html，双击打开

# 方法 2：使用命令行
# macOS
open vggt_visualizations/dynamic/index.html

# Linux
xdg-open vggt_visualizations/dynamic/index.html

# Windows
start vggt_visualizations/dynamic/index.html
```

HTML 报告包含：
- 📊 性能指标汇总
- 🖼️ 所有深度图的网格展示
- ☁️ 点云可视化
- 📹 相机轨迹
- 🔍 模型对比（如果有）

### 查看点云文件

推荐使用以下软件打开 `.ply` 文件：

#### Windows/Mac/Linux：
1. **MeshLab**（免费）
   - 下载: https://www.meshlab.net/
   - 打开 MeshLab → File → Import Mesh → 选择 `.ply` 文件

2. **CloudCompare**（免费）
   - 下载: https://www.cloudcompare.org/
   - 打开 CloudCompare → File → Open → 选择 `.ply` 文件

3. **Blender**（免费）
   - 下载: https://www.blender.org/
   - File → Import → Stanford (.ply)

#### 在线查看：
- https://3dviewer.net/
  - 上传 `.ply` 文件即可在线查看

### 查看图像文件

所有 `.png` 文件可以用任何图像查看器打开：
- Windows: Photos, Paint
- macOS: Preview, Photos
- Linux: Eye of GNOME, gwenview

---

## 🎨 更多可视化选项

### 1. 只生成特定类型的可视化

```bash
# 只生成深度图
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/depth_only \
    --skip_points \
    --skip_camera

# 只生成点云
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/pointcloud_only \
    --skip_depth \
    --skip_camera
```

### 2. 使用 VGGT 自带的演示

```bash
# Gradio Web 界面（交互式）
python demo_gradio.py --share

# 会生成一个公共 URL，例如：
# Running on public URL: https://xxxxx.gradio.live
# 在浏览器中打开这个 URL

# Viser 3D 查看器
python demo_viser.py --image_folder /workspace/data/eth3d/courtyard/images

# COLMAP 格式导出
python demo_colmap.py --scene_dir /workspace/data/eth3d/courtyard
```

### 3. 自定义可视化脚本

创建 `custom_viz.py`：

```python
import torch
import matplotlib.pyplot as plt
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 加载模型
model = VGGT()
model.load_state_dict(torch.load('/workspace/models/vggt_int8_dynamic.pt'))
model.cuda().eval()

# 加载图像
images = load_and_preprocess_images([
    '/workspace/data/eth3d/courtyard/images/DSC_0001.JPG',
]).cuda()

# 推理
with torch.no_grad():
    pred = model(images)

# 可视化深度
depth = pred['depth'][0, 0, :, :, 0].cpu().numpy()
plt.figure(figsize=(12, 8))
plt.imshow(depth, cmap='turbo')
plt.colorbar()
plt.title('Depth Map')
plt.savefig('/workspace/my_depth.png', dpi=300)
print('Saved to /workspace/my_depth.png')
```

运行：

```bash
python custom_viz.py
```

---

## 📊 生成对比报告

对比动态量化和静态量化：

```bash
# 创建对比脚本
cat > /workspace/compare_quantization.py << 'EOF'
import torch
import json
from pathlib import Path

# 读取元数据
dynamic_meta = json.load(open('/workspace/visualizations/dynamic/metadata.json'))
static_meta = json.load(open('/workspace/visualizations/static/metadata.json'))

print("=" * 60)
print("Quantization Methods Comparison")
print("=" * 60)
print()

print("Dynamic Quantization:")
print(f"  Inference Time: {dynamic_meta['inference_time']:.3f}s")
print(f"  Avg Time/Image: {dynamic_meta['avg_time_per_image']:.3f}s")
print()

print("Static Quantization:")
print(f"  Inference Time: {static_meta['inference_time']:.3f}s")
print(f"  Avg Time/Image: {static_meta['avg_time_per_image']:.3f}s")
print()

speedup = float(dynamic_meta['inference_time']) / float(static_meta['inference_time'])
print(f"Static is {speedup:.2f}x faster than Dynamic")
EOF

python /workspace/compare_quantization.py
```

---

## 🔍 高级可视化

### 1. 生成视频（需要 ffmpeg）

```bash
# 安装 ffmpeg
apt-get install -y ffmpeg

# 将深度图序列生成视频
cd /workspace/visualizations/dynamic
ffmpeg -framerate 5 -pattern_type glob -i 'quant_depth_*.png' \
    -c:v libx264 -pix_fmt yuv420p depth_video.mp4

# 下载视频
# 在本地：scp -P <PORT> root@<POD_IP>:/workspace/visualizations/dynamic/depth_video.mp4 ./
```

### 2. 3D 点云动画

使用 Blender：

```python
# 在 Blender 中打开 Python 控制台，运行：
import bpy

# 导入点云
bpy.ops.import_mesh.ply(filepath="/path/to/pointcloud.ply")

# 添加相机轨迹
# 设置动画
# 渲染视频
```

### 3. 交互式 3D 查看（Plotly）

```bash
pip install plotly

# 创建交互式 3D 可视化
python << 'EOF'
import numpy as np
import plotly.graph_objects as go

# 加载点云
# （假设你已经有了点云数据）
points = np.load('/workspace/visualizations/dynamic/points.npy')

fig = go.Figure(data=[go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(size=1)
)])

fig.write_html('/workspace/visualizations/interactive_3d.html')
print('Saved to /workspace/visualizations/interactive_3d.html')
EOF
```

---

## 📋 完整工作流程示例

```bash
# ========== 在 RunPod 上 ==========

# 1. 生成可视化
cd /workspace/vggt
python scripts/visualize_results.py \
    --model_path /workspace/models/vggt_int8_dynamic.pt \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --output_dir /workspace/visualizations/dynamic \
    --max_images 20

# 2. 查看结果
ls -lh /workspace/visualizations/dynamic/
cat /workspace/visualizations/dynamic/metadata.json

# 3. 启动 HTTP 服务器（可选，用于在浏览器中预览）
cd /workspace/visualizations/dynamic
python -m http.server 8000 &

# 4. 打包结果
cd /workspace
tar -czf viz_results.tar.gz visualizations/

# 5. 查看文件大小
ls -lh viz_results.tar.gz


# ========== 在本地电脑上 ==========

# 1. 下载结果
scp -P <PORT> root@<POD_IP>:/workspace/viz_results.tar.gz ./

# 2. 解压
tar -xzf viz_results.tar.gz

# 3. 打开 HTML 报告
open visualizations/dynamic/index.html  # macOS
# 或
xdg-open visualizations/dynamic/index.html  # Linux
# 或
start visualizations/dynamic/index.html  # Windows

# 4. 查看点云
# 使用 MeshLab 或 CloudCompare 打开 .ply 文件
```

---

## 🆘 常见问题

### Q: SCP 下载速度很慢怎么办？

A: 使用压缩：

```bash
# 在 RunPod 上
tar -czf viz_small.tar.gz visualizations/dynamic/*.html visualizations/dynamic/*.json visualizations/dynamic/*.png

# 只下载必要文件（不包括大的 PLY 文件）
```

### Q: 无法连接到 RunPod 的 SSH？

A: 确保：
1. Pod 正在运行
2. 使用正确的端口（从 RunPod 控制台获取）
3. 检查防火墙设置

### Q: HTML 报告中图片不显示？

A: 确保：
1. HTML 文件和图片在同一目录
2. 使用浏览器打开（不是文本编辑器）

### Q: 点云文件太大无法下载？

A: 降采样：

```python
import numpy as np

# 读取点云
with open('pointcloud.ply', 'r') as f:
    lines = f.readlines()

# 只保留前 100000 个点
header_end = next(i for i, line in enumerate(lines) if 'end_header' in line)
new_lines = lines[:header_end+1] + lines[header_end+1:header_end+100001]

# 更新头部的顶点数量
new_lines[2] = f'element vertex 100000\n'

# 保存
with open('pointcloud_small.ply', 'w') as f:
    f.writelines(new_lines)
```

---

## ✅ 检查清单

下载完成后，你应该有：

- [ ] HTML 报告 (`index.html`)
- [ ] 深度图图像 (`quant_depth_*.png`)
- [ ] 点云图像 (`quant_pointcloud_*.png`)
- [ ] 点云 PLY 文件 (`quant_pointcloud_*.ply`)
- [ ] 相机轨迹图 (`quant_camera_*.png`)
- [ ] 元数据文件 (`metadata.json`)
- [ ] 模型对比图（如果生成了）(`model_comparison.png`)

---

## 📞 获取帮助

如果遇到问题：
1. 检查 `/workspace/visualizations/` 目录是否存在
2. 确认 RunPod Pod 正在运行
3. 查看 `metadata.json` 确认可视化已完成
4. 检查 SSH 连接设置

---

**准备好查看你的结果了吗？**

从第一步开始，生成你的可视化！

---

**最后更新**: 2025-10-13
**维护者**: Your Team
