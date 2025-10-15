# RunPod 纯工作流 - 快速参考

**🎯 目标**: 在RunPod上从零到结果，所有操作都在RunPod终端完成

**⏱️ 总时间**: 15-30分钟
**💰 总费用**: $0.10 - $0.40

---

## 📋 前提条件

1. 已在 https://runpod.io 创建账户
2. 代码已上传到Git仓库（GitHub/GitLab）或准备好上传ZIP

---

## 🚀 完整流程（复制粘贴即可）

### 方法1: 使用Git（推荐）

**在RunPod终端复制粘贴这一整段：**

```bash
################################################################################
# RunPod 完整量化实验 - Git版本
#
# 使用前修改: 把 yourusername/vggt 改成你的仓库地址
################################################################################

# 步骤1: 克隆代码
cd /workspace
git clone https://github.com/yourusername/vggt.git vggt

# 步骤2: 运行快速测试（5张图像，约10分钟）
cd vggt
bash scripts/runpod_full_workflow.sh quick

# 步骤3: 查看结果
echo ""
echo "========================================="
echo "实验完成！查看结果："
echo "========================================="
cat /workspace/results/quick_test_*/comprehensive_report.txt | head -60

# 步骤4: 创建下载包
cd /workspace
tar -czf results.tar.gz results/
echo ""
echo "结果已打包: /workspace/results.tar.gz"
echo "在RunPod界面点击Files下载此文件"
```

---

### 方法2: 使用上传的ZIP文件

**假设你已在RunPod界面上传了 `vggt.zip` 到 `/workspace/`**

```bash
################################################################################
# RunPod 完整量化实验 - ZIP版本
################################################################################

# 步骤1: 解压代码
cd /workspace
unzip vggt.zip
cd vggt

# 步骤2: 运行快速测试
bash scripts/runpod_full_workflow.sh quick

# 步骤3: 查看结果
cat /workspace/results/quick_test_*/comprehensive_report.txt | head -60

# 步骤4: 创建下载包
cd /workspace
tar -czf results.tar.gz results/
echo ""
echo "结果已打包: /workspace/results.tar.gz"
```

---

## 📊 测试选项

运行 `runpod_full_workflow.sh` 时可以选择：

| 参数 | 图像数 | 时间 | 费用 | 适用场景 |
|------|--------|------|------|----------|
| `quick` | 5张 | ~10分钟 | ~$0.10 | 首次测试 |
| `standard` | 10张 | ~15分钟 | ~$0.13 | 作业提交 |
| `full` | 50张 | ~60分钟 | ~$0.40 | 论文发表 |

**示例**:
```bash
bash scripts/runpod_full_workflow.sh standard
```

---

## 🎯 单行超级命令

**复制这一行到RunPod终端（修改你的仓库地址）：**

```bash
cd /workspace && git clone https://github.com/yourusername/vggt.git vggt && cd vggt && bash scripts/runpod_full_workflow.sh quick && cat /workspace/results/quick_test_*/comprehensive_report.txt | head -60
```

**15分钟后自动完成并显示结果！**

---

## 📦 结果文件位置

实验完成后，结果保存在：

```
/workspace/results/quick_test_YYYYMMDD_HHMMSS/
├── comprehensive_results.json          # 完整数据
├── comprehensive_report.txt            # 文本报告
└── comprehensive_visualizations.png    # 可视化图表
```

---

## 💾 下载结果到本地

### 方法1: 通过RunPod Web界面（推荐）

**在RunPod终端：**
```bash
cd /workspace
tar -czf results.tar.gz results/
```

**在RunPod Web界面：**
1. 点击 "Files"
2. 找到 `/workspace/results.tar.gz`
3. 点击下载

### 方法2: 通过浏览器查看图表

如果Pod有HTTP端口，可以直接在浏览器查看PNG：
```
http://<POD_IP>:<PORT>/workspace/results/quick_test_*/comprehensive_visualizations.png
```

---

## 🔄 运行多个实验

### 不同图像数量

```bash
cd /workspace/vggt

# 快速测试
bash scripts/runpod_full_workflow.sh quick

# 标准测试
bash scripts/runpod_full_workflow.sh standard

# 完整测试
bash scripts/runpod_full_workflow.sh full
```

### 不同场景

```bash
cd /workspace/vggt

# 运行多个ETH3D场景
for scene in courtyard delivery_area facade; do
    python scripts/comprehensive_evaluation.py \
        --image_folder /workspace/data/eth3d/$scene/dslr_images_undistorted \
        --max_images 10 \
        --output_dir /workspace/results/${scene}_test
done
```

---

## ⚙️ 高级选项

### 使用tmux防止断开

```bash
# 创建tmux会话
tmux new -s experiment

# 在tmux中运行
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh full

# 分离: Ctrl+B, 然后按 D
# 重新连接: tmux attach -t experiment
```

### 后台运行

```bash
cd /workspace/vggt
nohup bash scripts/runpod_full_workflow.sh full > /workspace/experiment.log 2>&1 &

# 查看日志
tail -f /workspace/experiment.log

# 查看进程
ps aux | grep runpod_full_workflow
```

### 自定义参数

```bash
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/dslr_images_undistorted \
    --max_images 20 \
    --output_dir /workspace/results/custom_20img \
    --device cuda
```

---

## ❓ 故障排查

### 问题1: Git clone失败

```bash
# 检查网络
ping -c 3 github.com

# 使用HTTPS而不是SSH
git clone https://github.com/yourusername/vggt.git vggt
```

### 问题2: CUDA内存不足

```bash
# 使用更少图像
cd /workspace/vggt
bash scripts/runpod_full_workflow.sh quick  # 只用5张图

# 或手动指定
python scripts/comprehensive_evaluation.py \
    --max_images 3 \
    --output_dir /workspace/results/tiny_test
```

### 问题3: 找不到数据

```bash
# 重新下载ETH3D
cd /workspace/vggt
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d

# 检查数据
find /workspace/data -name "*.JPG" | head -10
```

### 问题4: 依赖问题

```bash
# 重新安装依赖
pip install --force-reinstall torch torchvision numpy matplotlib tqdm
```

---

## 🎯 检查清单

运行实验前：
- [ ] Pod已创建（推荐RTX 4090）
- [ ] 代码已上传到Git或已上传ZIP
- [ ] 已在RunPod终端登录

运行实验中：
- [ ] 使用tmux或nohup防止断开
- [ ] 定期检查GPU使用: `nvidia-smi`
- [ ] 监控磁盘空间: `df -h /workspace`

实验完成后：
- [ ] 查看文本报告: `cat /workspace/results/*/comprehensive_report.txt`
- [ ] 创建下载包: `tar -czf results.tar.gz results/`
- [ ] 停止Pod避免计费

---

## 📚 相关文档

- **[RUNPOD_START_HERE.md](RUNPOD_START_HERE.md)** - 详细的快速开始指南
- **[RUNPOD_QUICK_COMMANDS.md](RUNPOD_QUICK_COMMANDS.md)** - 命令速查手册
- **[RUNPOD_COMPREHENSIVE_GUIDE.md](RUNPOD_COMPREHENSIVE_GUIDE.md)** - 完整RunPod指南

---

## 🎉 快速回顾

**3步完成实验：**

```bash
# 1. 获取代码
cd /workspace && git clone https://github.com/yourusername/vggt.git vggt

# 2. 运行实验
cd vggt && bash scripts/runpod_full_workflow.sh quick

# 3. 查看结果
cat /workspace/results/quick_test_*/comprehensive_report.txt
```

**下载结果：**
```bash
cd /workspace && tar -czf results.tar.gz results/
# 然后在RunPod界面下载
```

**停止计费：**
在RunPod界面点击 "Stop"

---

**就是这么简单！祝实验顺利！** 🚀
