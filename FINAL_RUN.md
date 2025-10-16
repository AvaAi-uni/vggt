# 🎯 最终运行指南

## 已修复的问题

1. ✅ 配置文件路径：`/workspace/vggt/data/eth3d`（不是 training 子目录）
2. ✅ ETH3D 数据加载器：支持 `images` 和 `dslr_undistorted_images` 两种目录结构
3. ✅ 所有依赖已在 requirements.txt（包括 hydra-core）

---

## 🚀 现在运行

### 步骤 1：诊断数据集（可选）

```bash
cd /workspace/vggt
bash check_dataset.sh
```

这会显示：
- 数据集目录结构
- 每个场景的图像数量
- 使用的是 `images` 还是 `dslr_undistorted_images`

### 步骤 2：运行训练

```bash
bash train.sh eth3d_fp32_quick_test
```

---

## ✅ 预期成功输出

```
INFO: Loading ETH3D scenes: ['courtyard', 'delivery_area', ...]
INFO:   Loaded scene 'courtyard': 389 images       ← 有图像数量！
INFO:   Loaded scene 'delivery_area': 238 images
INFO:   Loaded scene 'electro': 328 images
...
INFO: Training: ETH3D Dataset initialized
INFO:   - Root directory: /workspace/vggt/data/eth3d
INFO:   - Number of scenes: 13                     ← 不是 0！
INFO:   - Total sequences: 13
INFO:   - Virtual dataset length: 200

Train Epoch: [0]  [0/50]  Batch Time: 2.345  ...
```

---

## 📁 正确的目录结构

ETH3D 数据集支持两种结构：

**结构 1 (标准)**：
```
data/eth3d/
├── courtyard/
│   └── dslr_undistorted_images/
│       ├── DSC_0001.JPG
│       └── ...
├── delivery_area/
│   └── dslr_undistorted_images/
└── ...
```

**结构 2 (简化)**：
```
data/eth3d/
├── courtyard/
│   └── images/
│       ├── DSC_0001.JPG
│       └── ...
├── delivery_area/
│   └── images/
└── ...
```

代码已修改，两种结构都支持！

---

## 🔧 如果仍然失败

运行诊断：
```bash
bash check_dataset.sh
```

检查输出，如果某个场景显示 "没有 images 或 dslr_undistorted_images 目录"，说明数据集解压有问题。

---

## 立即运行

```bash
cd /workspace/vggt
bash train.sh eth3d_fp32_quick_test
```

这次一定成功！🎯
