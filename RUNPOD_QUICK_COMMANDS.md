# RunPod 快速命令参考

**快速查找你需要的命令！**

---

## 🎯 最常用命令（复制即用）

### 1. 快速测试（5张图像，5-10分钟）

```bash
bash /workspace/run_quick_test.sh
```

### 2. 标准测试（10张图像，10-15分钟）

```bash
bash /workspace/run_standard_test.sh
```

### 3. 完整测试（50张图像，30-60分钟）

```bash
bash /workspace/run_full_test.sh
```

---

## 📦 完整工作流程（复制粘贴整段）

```bash
# ============================================================================
# RunPod 完整量化实验工作流程
# 复制粘贴这整段到RunPod终端，一键完成所有操作
# ============================================================================

# 步骤1: 进入项目目录
cd /workspace/vggt

# 步骤2: 运行环境设置（首次使用）
bash scripts/runpod_setup_comprehensive.sh

# 步骤3: 运行快速测试
bash /workspace/run_quick_test.sh

# 步骤4: 查看结果
cat /workspace/results/quick_test/comprehensive_report.txt

echo "=============================================================================="
echo "✅ 实验完成！"
echo "结果保存在: /workspace/results/quick_test/"
echo "=============================================================================="
```

---

## 🔧 环境设置命令

### 首次设置（一键完成）

```bash
cd /workspace/vggt
bash scripts/runpod_setup_comprehensive.sh
```

### 手动设置（如果自动设置失败）

```bash
# 创建目录
mkdir -p /workspace/data /workspace/results /workspace/models

# 安装依赖
cd /workspace/vggt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib Pillow scipy tqdm huggingface_hub

# 验证CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 下载测试数据

```bash
cd /workspace/vggt
python scripts/download_eth3d.py --output_dir /workspace/data/eth3d
```

---

## 🚀 运行实验命令

### 方法1: 使用快捷脚本（推荐）

```bash
# 快速测试（5张图）
bash /workspace/run_quick_test.sh

# 标准测试（10张图）
bash /workspace/run_standard_test.sh

# 完整测试（50张图）
bash /workspace/run_full_test.sh
```

### 方法2: 手动运行

```bash
cd /workspace/vggt

# 基础命令
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 10 \
    --output_dir /workspace/results/my_test \
    --device cuda

# 使用自定义图像
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/my_images \
    --max_images 20 \
    --output_dir /workspace/results/custom_test \
    --device cuda

# CPU模式（如果CUDA不可用）
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 5 \
    --output_dir /workspace/results/cpu_test \
    --device cpu
```

### 方法3: 后台运行（推荐长时间任务）

```bash
cd /workspace/vggt

# 使用nohup后台运行
nohup python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 50 \
    --output_dir /workspace/results/background_test \
    --device cuda \
    > /workspace/results/run.log 2>&1 &

# 查看进程
ps aux | grep comprehensive_evaluation

# 实时查看日志
tail -f /workspace/results/run.log

# 停止后台任务（如需要）
pkill -f comprehensive_evaluation
```

### 方法4: 使用tmux（推荐，防止断开）

```bash
# 创建tmux会话
tmux new -s quantization

# 在tmux中运行实验
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 30 \
    --output_dir /workspace/results/tmux_test \
    --device cuda

# 分离tmux会话: 按 Ctrl+B, 然后按 D
# 此时可以关闭浏览器，实验继续运行

# 重新连接到会话
tmux attach -t quantization

# 查看所有会话
tmux ls

# 结束会话
tmux kill-session -t quantization
```

---

## 📊 查看结果命令

### 文本报告

```bash
# 查看完整报告
cat /workspace/results/quick_test/comprehensive_report.txt

# 查看前50行
head -50 /workspace/results/quick_test/comprehensive_report.txt

# 使用less浏览（可翻页）
less /workspace/results/quick_test/comprehensive_report.txt
# 按 q 退出
```

### JSON数据

```bash
# 格式化查看JSON
python -m json.tool /workspace/results/quick_test/comprehensive_results.json | less

# 查看特定部分（使用jq，如果安装了）
jq '.Baseline_FP32' /workspace/results/quick_test/comprehensive_results.json
jq '.INT8_Per_Channel_Symmetric' /workspace/results/quick_test/comprehensive_results.json
```

### 列出所有输出文件

```bash
# 列出结果目录
ls -lh /workspace/results/quick_test/

# 递归列出所有结果
tree /workspace/results/
# 或
find /workspace/results -type f -name "*.txt" -o -name "*.json" -o -name "*.png"
```

---

## 💾 下载结果命令

### 从RunPod下载到本地

在**本地电脑**运行（不是在RunPod终端）：

```bash
# 下载单个文件夹（推荐）
scp -r -P <POD_SSH_PORT> \
    root@<POD_IP>:/workspace/results/quick_test \
    ~/Desktop/results/

# 下载所有结果
scp -r -P <POD_SSH_PORT> \
    root@<POD_IP>:/workspace/results/ \
    ~/Desktop/runpod_results/

# 使用rsync（更快，支持断点续传）
rsync -avz -e "ssh -p <POD_SSH_PORT>" \
    root@<POD_IP>:/workspace/results/ \
    ~/Desktop/runpod_results/

# Windows（使用Git Bash或WSL）
scp -r -P <PORT> root@<IP>:/workspace/results/quick_test C:/Users/YourName/Desktop/
```

### 压缩后下载（更快）

在**RunPod终端**运行：

```bash
# 压缩结果
cd /workspace
tar -czf results.tar.gz results/

# 然后在本地下载
# scp -P <PORT> root@<IP>:/workspace/results.tar.gz ~/Desktop/

# 本地解压
# tar -xzf results.tar.gz
```

---

## 🔍 监控和调试命令

### 监控GPU

```bash
# 实时监控
watch -n 1 nvidia-smi

# 单次查看
nvidia-smi

# 查看GPU使用率
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
```

### 监控Python进程

```bash
# 查找Python进程
ps aux | grep python

# 查看特定进程的详细信息
top -p <PID>

# 查看进程CPU和内存使用
htop  # 如果安装了
```

### 检查磁盘空间

```bash
# 查看磁盘使用
df -h /workspace

# 查看目录大小
du -sh /workspace/*
du -sh /workspace/data/
du -sh /workspace/results/

# 查找大文件
find /workspace -type f -size +100M -exec ls -lh {} \;
```

### 查看日志

```bash
# 实时查看日志
tail -f /workspace/results/run.log

# 查看最后100行
tail -100 /workspace/results/run.log

# 搜索特定内容
grep "ERROR" /workspace/results/run.log
grep "完成" /workspace/results/run.log
```

---

## 🧹 清理命令

### 清理缓存

```bash
# 清理Python缓存
find /workspace/vggt -type d -name "__pycache__" -exec rm -rf {} +

# 清理临时文件
rm -rf /tmp/*

# 清理PyTorch缓存
python << EOF
import torch
torch.cuda.empty_cache()
print("GPU缓存已清理")
EOF
```

### 删除旧结果

```bash
# 删除特定实验结果
rm -rf /workspace/results/old_test/

# 删除所有结果（小心！）
# rm -rf /workspace/results/*
```

---

## 🛠️ 故障排查命令

### CUDA问题

```bash
# 检查CUDA是否可用
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA版本: {torch.version.cuda}')"
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"

# 清理GPU内存
python << EOF
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU缓存已清理")
EOF
```

### 依赖问题

```bash
# 检查已安装的包
pip list | grep torch
pip list | grep numpy
pip list | grep matplotlib

# 重新安装
pip install --force-reinstall torch torchvision
pip install --force-reinstall numpy matplotlib
```

### 图像找不到

```bash
# 搜索所有图像
find /workspace/data -name "*.jpg" | head -20
find /workspace/data -name "*.png" | head -20

# 统计图像数量
find /workspace/data/eth3d -name "*.jpg" -o -name "*.png" | wc -l

# 使用找到的路径运行
python scripts/comprehensive_evaluation.py \
    --image_folder <FOUND_PATH> \
    --max_images 5 \
    --output_dir /workspace/results/test
```

---

## 📝 批量实验命令

### 运行多个实验

```bash
# 创建批量实验脚本
cat > /workspace/run_batch_experiments.sh << 'EOF'
#!/bin/bash
cd /workspace/vggt

# 实验1: 5张图
python scripts/comprehensive_evaluation.py \
    --max_images 5 \
    --output_dir /workspace/results/exp_5img \
    --image_folder /workspace/data/eth3d/courtyard/images

# 实验2: 10张图
python scripts/comprehensive_evaluation.py \
    --max_images 10 \
    --output_dir /workspace/results/exp_10img \
    --image_folder /workspace/data/eth3d/courtyard/images

# 实验3: 20张图
python scripts/comprehensive_evaluation.py \
    --max_images 20 \
    --output_dir /workspace/results/exp_20img \
    --image_folder /workspace/data/eth3d/courtyard/images

echo "所有实验完成！"
EOF

# 添加执行权限
chmod +x /workspace/run_batch_experiments.sh

# 运行
bash /workspace/run_batch_experiments.sh
```

### 并行运行（小心GPU内存）

```bash
# 并行运行多个实验（确保GPU内存足够）
cd /workspace/vggt

python scripts/comprehensive_evaluation.py \
    --max_images 5 \
    --output_dir /workspace/results/parallel1 \
    --image_folder /workspace/data/eth3d/courtyard/images &

python scripts/comprehensive_evaluation.py \
    --max_images 5 \
    --output_dir /workspace/results/parallel2 \
    --image_folder /workspace/data/eth3d/delivery_area/images &

# 等待所有后台任务完成
wait

echo "所有并行实验完成！"
```

---

## 📂 文件上传命令

### 使用SCP上传（从本地）

```bash
# Windows (Git Bash)
cd "C:\Users\Ava Ai\Desktop\8539Project\code"
scp -r -P <PORT> ./vggt root@<IP>:/workspace/

# Linux/Mac
cd ~/projects/8539Project/code
scp -r -P <PORT> ./vggt root@<IP>:/workspace/

# 上传单个文件
scp -P <PORT> myfile.py root@<IP>:/workspace/vggt/scripts/
```

### 使用rsync上传（更快）

```bash
# Linux/Mac/WSL
rsync -avz -e "ssh -p <PORT>" \
    ./vggt/ \
    root@<IP>:/workspace/vggt/

# 只上传Python文件
rsync -avz --include="*.py" --include="*/" --exclude="*" \
    -e "ssh -p <PORT>" \
    ./vggt/ \
    root@<IP>:/workspace/vggt/
```

---

## 🎓 高级技巧

### 自动保存实验记录

```bash
# 创建自动记录脚本
cat > /workspace/run_and_log.sh << 'EOF'
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/workspace/logs"
mkdir -p $LOG_DIR

cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 10 \
    --output_dir /workspace/results/exp_$TIMESTAMP \
    --device cuda \
    2>&1 | tee $LOG_DIR/run_$TIMESTAMP.log

echo "实验完成，日志保存在: $LOG_DIR/run_$TIMESTAMP.log"
EOF

chmod +x /workspace/run_and_log.sh
bash /workspace/run_and_log.sh
```

### 定时运行（cron）

```bash
# 编辑crontab
crontab -e

# 添加定时任务（每天凌晨2点运行）
0 2 * * * cd /workspace/vggt && bash /workspace/run_standard_test.sh >> /workspace/logs/cron.log 2>&1
```

### 邮件通知（需要配置SMTP）

```bash
# 实验完成后发送邮件
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 10 \
    --output_dir /workspace/results/test && \
    echo "实验完成" | mail -s "RunPod实验完成" your@email.com
```

---

## ⚡ 一行命令速查

| 任务 | 命令 |
|------|------|
| 快速测试 | `bash /workspace/run_quick_test.sh` |
| 查看结果 | `cat /workspace/results/quick_test/comprehensive_report.txt` |
| 监控GPU | `watch -n 1 nvidia-smi` |
| 后台运行 | `nohup python script.py &` |
| 查看进程 | `ps aux \| grep python` |
| 停止进程 | `pkill -f comprehensive` |
| 下载结果 | `scp -r -P <PORT> root@<IP>:/workspace/results ~/Desktop/` |
| 清理缓存 | `rm -rf __pycache__ /tmp/*` |
| 查看磁盘 | `df -h /workspace` |
| 压缩结果 | `tar -czf results.tar.gz results/` |

---

## 📞 获取帮助

### 查看脚本帮助

```bash
# 查看评估脚本帮助
python scripts/comprehensive_evaluation.py --help

# 查看下载脚本帮助
python scripts/download_eth3d.py --help
```

### 阅读文档

```bash
# 查看快速开始
cat /workspace/vggt/START_HERE_COMPREHENSIVE.md | less

# 查看完整指南
cat /workspace/vggt/COMPREHENSIVE_QUANTIZATION_GUIDE.md | less

# 查看RunPod指南
cat /workspace/vggt/RUNPOD_COMPREHENSIVE_GUIDE.md | less
```

---

## ✅ 常用命令组合

### 完整工作流（复制整段）

```bash
# ============================================================================
# 完整 RunPod 工作流（从零到结果）
# ============================================================================

# 1. 环境设置
cd /workspace/vggt && bash scripts/runpod_setup_comprehensive.sh

# 2. 运行快速测试
bash /workspace/run_quick_test.sh

# 3. 查看结果
cat /workspace/results/quick_test/comprehensive_report.txt

# 4. 压缩结果准备下载
cd /workspace && tar -czf results.tar.gz results/

echo "完成！结果已压缩到 /workspace/results.tar.gz"
echo "在本地运行: scp -P <PORT> root@<IP>:/workspace/results.tar.gz ~/"
```

### 调试模式（详细输出）

```bash
cd /workspace/vggt
python -u scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 3 \
    --output_dir /workspace/results/debug \
    --device cuda \
    2>&1 | tee /workspace/debug.log
```

---

## 🎉 复制即用模板

### 基础实验

```bash
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 10 \
    --output_dir /workspace/results/my_experiment \
    --device cuda
```

### 自定义实验

```bash
cd /workspace/vggt
python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/<YOUR_FOLDER> \
    --max_images <NUMBER> \
    --output_dir /workspace/results/<EXP_NAME> \
    --device cuda
```

### 后台运行模板

```bash
cd /workspace/vggt
nohup python scripts/comprehensive_evaluation.py \
    --image_folder /workspace/data/eth3d/courtyard/images \
    --max_images 50 \
    --output_dir /workspace/results/background_test \
    --device cuda \
    > /workspace/results/run.log 2>&1 &

echo "后台运行已启动，PID: $!"
echo "查看日志: tail -f /workspace/results/run.log"
```

---

**保存这个文件作为快速参考！** 📑

需要更多帮助？查看:
- `RUNPOD_COMPREHENSIVE_GUIDE.md` - 完整RunPod指南
- `START_HERE_COMPREHENSIVE.md` - 项目总览
- `COMPREHENSIVE_QUANTIZATION_GUIDE.md` - 量化指南
