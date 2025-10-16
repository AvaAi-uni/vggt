# 🚀 立即开始

## 在 RunPod 上运行（唯一需要的命令）

```bash
cd /workspace/vggt
bash runpod_start.sh
```

**就这么简单！** 脚本会自动：
1. 安装依赖（NumPy 1.x, wcmatch, vggt包）
2. 下载 ETH3D 数据集（~1.5 GB）
3. 验证数据集完整性
4. 启动快速测试训练（5-10分钟）

---

## 如果快速测试成功

运行完整实验：

```bash
# FP32 Baseline（建立基准）
bash train.sh eth3d_fp32_baseline

# INT8 量化实验
bash train.sh eth3d_int8_per_tensor
bash train.sh eth3d_int8_per_channel

# INT4 量化实验
bash train.sh eth3d_int4_group128
bash train.sh eth3d_int4_group64
bash train.sh eth3d_int4_group32
```

---

## 预期输出（成功）

```
[1/4] 安装依赖...
✓ 依赖已安装

[2/4] 下载 ETH3D 数据集...
下载中 (~1.5 GB)...
multi_view_training_dslr_undistorted.7z  100%
解压中...
✓ 数据集已下载

[3/4] 验证数据集...
找到 3500 张图像
✓ 数据集验证完成

[4/4] 启动快速测试...

INFO: Training: ETH3D Dataset initialized
INFO:   - Number of scenes: 13
INFO:   - Total sequences: 13
INFO:   - Virtual dataset length: 200

Train Epoch: [0]  [0/50]  Batch Time: 2.345  ...
```

**关键指标**：
- ✅ Number of scenes: 13 (不是 0)
- ✅ 训练开始，显示 Batch Time

---

## 常见问题

### Q: 找到 0 张图像怎么办？

脚本会自动尝试修复。如果仍然失败，手动检查：

```bash
# 查看解压后的目录结构
ls -la data/eth3d/

# 如果看到 multi_view_training_dslr_undistorted 目录
mv data/eth3d/multi_view_training_dslr_undistorted data/eth3d/training

# 验证
find data/eth3d/training -name "*.JPG" | wc -l
```

### Q: ModuleNotFoundError: No module named 'hydra'

运行：
```bash
pip install -r requirements.txt
pip install -e .
```

### Q: 下载太慢怎么办？
脚本支持断点续传：
```bash
bash runpod_start.sh
```

### Q: 如何监控训练？
```bash
# 查看日志
tail -f logs/eth3d_fp32_quick_test/train.log

# TensorBoard
tensorboard --logdir logs --port 6006 --bind_all
```

---

## 文件说明

保留的文件：
- `runpod_start.sh` - 一键启动脚本
- `train.sh` - 训练启动脚本
- `README.md` - 简要说明
- `START_HERE.md` - 本文件

已删除的文件：
- 所有旧的文档（20+ 个 .md 文件）
- 所有旧的脚本（.bat, 旧的 .sh）
- 只保留必要的核心文件

---

## 立即运行

```bash
cd /workspace/vggt
bash runpod_start.sh
```

🎯 就这么简单！
