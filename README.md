# VGGT - ETH3D Training with Quantization

## RunPod 快速启动

```bash
cd /workspace/vggt
bash runpod_start.sh
```

### 如果遇到错误（找到0张图像 或 缺少hydra模块）

```bash
bash fix_now.sh
```

然后再运行训练：
```bash
bash train.sh eth3d_fp32_quick_test
```

## 训练命令

```bash
# 快速测试（5-10分钟）
bash train.sh eth3d_fp32_quick_test

# FP32 Baseline（4-6小时）
bash train.sh eth3d_fp32_baseline

# INT8 量化
bash train.sh eth3d_int8_per_tensor
bash train.sh eth3d_int8_per_channel

# INT4 量化
bash train.sh eth3d_int4_group128
bash train.sh eth3d_int4_group64
bash train.sh eth3d_int4_group32
```

## 监控训练

```bash
# 查看日志
tail -f logs/eth3d_fp32_quick_test/train.log

# TensorBoard
tensorboard --logdir logs --port 6006 --bind_all
```

## 验证

```bash
# 检查数据集
ls data/eth3d/training/

# 统计图像
find data/eth3d/training -name "*.JPG" | wc -l

# 检查环境
python -c "import vggt; print('OK')"
```
