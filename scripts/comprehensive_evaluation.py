#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
完整的量化评估脚本

这个脚本提供了一个全面的量化实验评估框架，包括：
1. Baseline: FP32原始模型
2. 7种量化方案的对比
3. 8种评估指标: MAE, MSE, RMSE, PSNR, Cross Entropy, Cosine Similarity, 模型大小, 推理时间

输出：
- 详细的JSON报告
- 格式化的文本报告
- 可视化图表（对比图、热力图等）

使用方法:
python scripts/comprehensive_evaluation.py \
    --model_name facebook/VGGT-1B \
    --image_folder data/eth3d/courtyard/images \
    --output_dir results/comprehensive_evaluation \
    --max_images 10

作者: Quantization Research Team
日期: 2025-10-16
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Any
import traceback

matplotlib.use('Agg')

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.quantization.comprehensive_quantizer import (
    quantize_model_comprehensive,
    get_all_quantization_configs,
    estimate_model_size,
    measure_inference_time
)


# ============================================================================
# 评估指标计算
# ============================================================================

def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()


def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Squared Error"""
    return torch.mean((pred - target) ** 2).item()


def calculate_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error"""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio

    PSNR = 10 * log10(MAX^2 / MSE)

    常用范围：
    - >40 dB: 优秀
    - 30-40 dB: 良好
    - 20-30 dB: 可接受
    - <20 dB: 较差
    """
    mse = calculate_mse(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def calculate_cross_entropy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Cross Entropy Loss

    CE = -sum(target * log(pred + eps))

    注意: 这里假设pred和target已经是概率分布或logits
    对于回归任务，这个指标可能不适用
    """
    # 如果是分类logits，转换为概率
    if pred.dim() > 1 and pred.shape[-1] > 1:
        pred_prob = F.softmax(pred, dim=-1)
        target_prob = F.softmax(target, dim=-1)
        ce = -torch.sum(target_prob * torch.log(pred_prob + eps))
        return ce.item() / pred.numel()
    else:
        # 对于回归任务，使用MSE作为替代
        return calculate_mse(pred, target)


def calculate_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Cosine Similarity

    CosSim = (A · B) / (||A|| * ||B||)

    范围: [-1, 1]
    - 1: 完全相同
    - 0: 正交
    - -1: 完全相反
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    cos_sim = F.cosine_similarity(pred_flat.unsqueeze(0), target_flat.unsqueeze(0))
    return cos_sim.item()


def calculate_relative_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Relative Error

    RE = mean(|pred - target| / (|target| + eps))
    """
    relative_error = torch.abs(pred - target) / (torch.abs(target) + eps)
    return torch.mean(relative_error).item()


def calculate_ssim_simplified(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Simplified SSIM (Structural Similarity Index)

    这是SSIM的简化版本，用于快速评估
    完整的SSIM需要考虑局部窗口，这里只计算全局统计量
    """
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    pred_var = torch.var(pred)
    target_var = torch.var(target)
    cov = torch.mean((pred - pred_mean) * (target - target_mean))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = ((2 * pred_mean * target_mean + c1) * (2 * cov + c2)) / \
           ((pred_mean ** 2 + target_mean ** 2 + c1) * (pred_var + target_var + c2))

    return ssim.item() if isinstance(ssim, torch.Tensor) else ssim


def calculate_comprehensive_metrics(
    original_output: Dict[str, torch.Tensor],
    quantized_output: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, float]]:
    """
    计算所有评估指标

    Args:
        original_output: 原始模型输出
        quantized_output: 量化模型输出

    Returns:
        包含所有输出的所有指标的嵌套字典
    """
    all_metrics = {}

    # 遍历所有输出
    for key in original_output.keys():
        if key == "images":
            continue

        if not isinstance(original_output[key], torch.Tensor):
            continue
        if not isinstance(quantized_output[key], torch.Tensor):
            continue

        # 确保在同一设备上
        orig = original_output[key].detach().cpu().float()
        quant = quantized_output[key].detach().cpu().float()

        # 检查形状
        if orig.shape != quant.shape:
            print(f"⚠️  形状不匹配 {key}: {orig.shape} vs {quant.shape}, 跳过")
            continue

        # 计算所有指标
        metrics = {
            "mae": calculate_mae(quant, orig),
            "mse": calculate_mse(quant, orig),
            "rmse": calculate_rmse(quant, orig),
            "psnr": calculate_psnr(quant, orig, max_val=torch.max(torch.abs(orig)).item()),
            "cross_entropy": calculate_cross_entropy(quant, orig),
            "cosine_similarity": calculate_cosine_similarity(quant, orig),
            "relative_error": calculate_relative_error(quant, orig),
            "ssim": calculate_ssim_simplified(quant, orig),
        }

        all_metrics[key] = metrics

    return all_metrics


# ============================================================================
# 主评估函数
# ============================================================================

def run_comprehensive_evaluation(
    model_name: str,
    test_images: List[str],
    output_dir: Path,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    运行完整的量化评估实验

    Args:
        model_name: 模型名称
        test_images: 测试图像路径列表
        output_dir: 输出目录
        device: 设备

    Returns:
        包含所有实验结果的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("完整量化评估实验")
    print("=" * 100)
    print(f"模型: {model_name}")
    print(f"测试图像数: {len(test_images)}")
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")
    print("=" * 100)
    print()

    # ========================================================================
    # 步骤 1: 加载原始模型（Baseline）
    # ========================================================================
    print("[步骤 1/5] 加载原始FP32模型（Baseline）...")
    print("-" * 100)

    original_model = VGGT.from_pretrained(model_name)
    original_model = original_model.to(device)
    original_model.eval()

    original_size = estimate_model_size(original_model)
    print(f"✓ 模型大小: {original_size['total_mb']:.2f} MB")
    print(f"  - 参数: {original_size['params_mb']:.2f} MB")
    print(f"  - 缓冲区: {original_size['buffers_mb']:.2f} MB")
    print()

    # ========================================================================
    # 步骤 2: 加载测试数据
    # ========================================================================
    print("[步骤 2/5] 加载测试图像...")
    print("-" * 100)

    images = load_and_preprocess_images(test_images).to(device)
    print(f"✓ 加载了 {len(test_images)} 张图像")
    print(f"  图像张量形状: {images.shape}")
    print()

    # ========================================================================
    # 步骤 3: Baseline推理
    # ========================================================================
    print("[步骤 3/5] Baseline (FP32) 推理...")
    print("-" * 100)

    with torch.no_grad():
        start_time = time.time()
        original_output = original_model(images)
        baseline_time = time.time() - start_time

    print(f"✓ 推理时间: {baseline_time:.4f}s")

    # 显示输出信息
    if isinstance(original_output, dict):
        print(f"✓ 输出键: {list(original_output.keys())}")
        for key, value in original_output.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape} [{value.dtype}]")
    print()

    # 测量更精确的推理时间
    baseline_time_stats = measure_inference_time(
        original_model, images, warmup=5, iterations=20, device=device
    )
    print(f"✓ 精确推理时间统计:")
    print(f"  - 平均: {baseline_time_stats['mean']:.4f}s")
    print(f"  - 标准差: {baseline_time_stats['std']:.4f}s")
    print(f"  - 最小: {baseline_time_stats['min']:.4f}s")
    print(f"  - 最大: {baseline_time_stats['max']:.4f}s")
    print()

    # ========================================================================
    # 步骤 4: 测试所有量化方案
    # ========================================================================
    print("[步骤 4/5] 测试所有量化方案...")
    print("=" * 100)

    results = {}

    # 添加baseline结果
    results["Baseline_FP32"] = {
        "model_size_mb": original_size['total_mb'],
        "inference_time": baseline_time_stats['mean'],
        "inference_time_std": baseline_time_stats['std'],
        "compression_ratio": 1.0,
        "speedup": 1.0,
        "metrics": {},  # Baseline没有误差
        "config": "FP32 原始模型，无量化"
    }

    # 获取所有量化配置
    quant_configs = get_all_quantization_configs(device=device)

    for idx, config in enumerate(quant_configs, 1):
        print(f"\n[{idx}/{len(quant_configs)}] 测试: {config.name}")
        print("-" * 100)
        print(f"  量化类型: {config.quant_type}")
        print(f"  位数: {config.bits} bits")
        if config.quant_type == "int4_group":
            print(f"  分组大小: {config.group_size}")

        try:
            # 量化模型
            print("  [1/4] 量化模型...")
            quant_model = quantize_model_comprehensive(original_model, config)
            quant_model = quant_model.to(device)
            quant_model.eval()

            # 计算模型大小
            quant_size = estimate_model_size(quant_model)
            compression_ratio = original_size['total_mb'] / quant_size['total_mb']
            print(f"  ✓ 量化后大小: {quant_size['total_mb']:.2f} MB (压缩率: {compression_ratio:.2f}x)")

            # 推理
            print("  [2/4] 推理测试...")
            with torch.no_grad():
                quant_output = quant_model(images)

            # 测量推理时间
            quant_time_stats = measure_inference_time(
                quant_model, images, warmup=5, iterations=20, device=device
            )
            speedup = baseline_time_stats['mean'] / quant_time_stats['mean']
            print(f"  ✓ 推理时间: {quant_time_stats['mean']:.4f}s (加速: {speedup:.2f}x)")

            # 计算精度指标
            print("  [3/4] 计算精度指标...")
            metrics = calculate_comprehensive_metrics(original_output, quant_output)

            # 显示关键指标
            if metrics:
                first_key = list(metrics.keys())[0]
                print(f"  ✓ 关键指标 ({first_key}):")
                print(f"    - MAE: {metrics[first_key]['mae']:.6f}")
                print(f"    - PSNR: {metrics[first_key]['psnr']:.2f} dB")
                print(f"    - Cosine Similarity: {metrics[first_key]['cosine_similarity']:.6f}")

            # 保存结果
            print("  [4/4] 保存结果...")
            results[config.name] = {
                "model_size_mb": quant_size['total_mb'],
                "inference_time": quant_time_stats['mean'],
                "inference_time_std": quant_time_stats['std'],
                "compression_ratio": compression_ratio,
                "speedup": speedup,
                "metrics": metrics,
                "config": {
                    "quant_type": config.quant_type,
                    "bits": config.bits,
                    "symmetric": config.symmetric,
                    "per_channel": config.per_channel,
                    "group_size": config.group_size if hasattr(config, 'group_size') else None,
                }
            }

            print(f"  ✓ {config.name} 完成!")

        except Exception as e:
            print(f"  ✗ 错误: {e}")
            print(f"  详细错误: {traceback.format_exc()}")
            results[config.name] = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    print("\n" + "=" * 100)

    # ========================================================================
    # 步骤 5: 生成报告
    # ========================================================================
    print("\n[步骤 5/5] 生成评估报告...")
    print("-" * 100)

    # 保存JSON报告
    json_path = output_dir / "comprehensive_results.json"
    save_json_report(results, json_path)
    print(f"✓ JSON报告: {json_path}")

    # 保存文本报告
    txt_path = output_dir / "comprehensive_report.txt"
    save_text_report(results, txt_path, test_images, model_name)
    print(f"✓ 文本报告: {txt_path}")

    # 生成可视化
    viz_path = output_dir / "comprehensive_visualizations.png"
    generate_visualizations(results, viz_path)
    print(f"✓ 可视化图表: {viz_path}")

    print()
    print("=" * 100)
    print("✅ 评估完成!")
    print("=" * 100)
    print()

    return results


# ============================================================================
# 报告生成
# ============================================================================

def save_json_report(results: Dict[str, Any], output_path: Path):
    """保存JSON报告"""
    # 转换为可序列化的格式
    serializable_results = {}
    for method, data in results.items():
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                serializable_data[key] = value
            elif isinstance(value, dict):
                serializable_data[key] = value
            else:
                serializable_data[key] = str(value)
        serializable_results[method] = serializable_data

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


def save_text_report(
    results: Dict[str, Any],
    output_path: Path,
    test_images: List[str],
    model_name: str
):
    """保存格式化的文本报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        # 标题
        f.write("=" * 100 + "\n")
        f.write("完整量化评估报告\n")
        f.write("=" * 100 + "\n\n")

        # 实验信息
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"测试图像数: {len(test_images)}\n")
        f.write(f"测试配置数: {len(results)}\n\n")

        # 概览表格
        f.write("=" * 100 + "\n")
        f.write("实验结果概览\n")
        f.write("=" * 100 + "\n\n")

        # 表头
        f.write(f"{'方案':<30} | {'大小(MB)':<10} | {'压缩率':<8} | {'时间(s)':<10} | {'加速':<8} | {'MAE':<12}\n")
        f.write("-" * 100 + "\n")

        # 表格内容
        for method, data in results.items():
            if "error" in data:
                f.write(f"{method:<30} | ERROR\n")
                continue

            size = data.get('model_size_mb', 0)
            comp = data.get('compression_ratio', 1.0)
            time_val = data.get('inference_time', 0)
            speedup = data.get('speedup', 1.0)

            # 获取第一个输出的MAE
            metrics = data.get('metrics', {})
            mae = "N/A"
            if metrics:
                first_output = list(metrics.values())[0]
                mae = f"{first_output.get('mae', 0):.6f}"

            f.write(f"{method:<30} | {size:<10.2f} | {comp:<8.2f}x | {time_val:<10.4f} | {speedup:<8.2f}x | {mae:<12}\n")

        f.write("\n")

        # 详细指标
        f.write("=" * 100 + "\n")
        f.write("详细评估指标\n")
        f.write("=" * 100 + "\n\n")

        for method, data in results.items():
            if "error" in data:
                f.write(f"{method}:\n")
                f.write(f"  错误: {data['error']}\n\n")
                continue

            f.write(f"{method}:\n")
            f.write(f"  模型大小: {data.get('model_size_mb', 0):.2f} MB\n")
            f.write(f"  压缩率: {data.get('compression_ratio', 1.0):.2f}x\n")
            f.write(f"  推理时间: {data.get('inference_time', 0):.4f}s (±{data.get('inference_time_std', 0):.4f}s)\n")
            f.write(f"  加速比: {data.get('speedup', 1.0):.2f}x\n")

            metrics = data.get('metrics', {})
            if metrics:
                f.write(f"  精度指标:\n")
                for output_name, output_metrics in metrics.items():
                    f.write(f"    {output_name}:\n")
                    for metric_name, metric_value in output_metrics.items():
                        if isinstance(metric_value, float):
                            if metric_name == "psnr":
                                f.write(f"      {metric_name}: {metric_value:.2f} dB\n")
                            else:
                                f.write(f"      {metric_name}: {metric_value:.6f}\n")

            f.write("\n")

        # 总结
        f.write("=" * 100 + "\n")
        f.write("实验总结\n")
        f.write("=" * 100 + "\n\n")

        # 找出最佳配置
        valid_results = {k: v for k, v in results.items() if "error" not in v and k != "Baseline_FP32"}

        if valid_results:
            # 最高压缩率
            best_comp = max(valid_results.items(), key=lambda x: x[1].get('compression_ratio', 0))
            f.write(f"最高压缩率: {best_comp[0]} ({best_comp[1]['compression_ratio']:.2f}x)\n")

            # 最快速度
            best_speed = max(valid_results.items(), key=lambda x: x[1].get('speedup', 0))
            f.write(f"最快推理: {best_speed[0]} ({best_speed[1]['speedup']:.2f}x)\n")

            # 最小MAE（最高精度）
            def get_first_mae(item):
                metrics = item[1].get('metrics', {})
                if metrics:
                    first = list(metrics.values())[0]
                    return first.get('mae', float('inf'))
                return float('inf')

            best_acc = min(valid_results.items(), key=get_first_mae)
            mae_val = get_first_mae(best_acc)
            f.write(f"最高精度: {best_acc[0]} (MAE: {mae_val:.6f})\n")

        f.write("\n")


def generate_visualizations(results: Dict[str, Any], output_path: Path):
    """生成可视化图表"""
    # 过滤有效结果
    valid_results = {k: v for k, v in results.items() if "error" not in v}

    if len(valid_results) == 0:
        print("没有有效结果，跳过可视化")
        return

    methods = list(valid_results.keys())

    # 提取数据
    sizes = [valid_results[m]['model_size_mb'] for m in methods]
    times = [valid_results[m]['inference_time'] for m in methods]
    compressions = [valid_results[m]['compression_ratio'] for m in methods]
    speedups = [valid_results[m]['speedup'] for m in methods]

    # 提取MAE
    maes = []
    for m in methods:
        metrics = valid_results[m].get('metrics', {})
        if metrics:
            first = list(metrics.values())[0]
            maes.append(first.get('mae', 0))
        else:
            maes.append(0)

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('完整量化评估结果', fontsize=16, fontweight='bold')

    # 1. 模型大小
    axes[0, 0].bar(range(len(methods)), sizes, color='skyblue')
    axes[0, 0].set_title('模型大小对比', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('大小 (MB)')
    axes[0, 0].set_xticks(range(len(methods)))
    axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. 推理时间
    axes[0, 1].bar(range(len(methods)), times, color='lightcoral')
    axes[0, 1].set_title('推理时间对比', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('时间 (秒)')
    axes[0, 1].set_xticks(range(len(methods)))
    axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. 压缩率
    axes[0, 2].bar(range(len(methods)), compressions, color='lightgreen')
    axes[0, 2].set_title('压缩率对比', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('压缩率 (倍)')
    axes[0, 2].set_xticks(range(len(methods)))
    axes[0, 2].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 2].grid(axis='y', alpha=0.3)

    # 4. 加速比
    axes[1, 0].bar(range(len(methods)), speedups, color='gold')
    axes[1, 0].set_title('推理加速比', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('加速比 (倍)')
    axes[1, 0].set_xticks(range(len(methods)))
    axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 5. MAE
    axes[1, 1].bar(range(len(methods)), maes, color='plum')
    axes[1, 1].set_title('平均绝对误差 (MAE)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_xticks(range(len(methods)))
    axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    # 6. 精度 vs 压缩率散点图
    axes[1, 2].scatter(compressions, maes, s=100, alpha=0.6, c=range(len(methods)), cmap='viridis')
    axes[1, 2].set_title('精度 vs 压缩率权衡', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('压缩率 (倍)')
    axes[1, 2].set_ylabel('MAE')
    axes[1, 2].grid(alpha=0.3)

    # 添加标签
    for i, method in enumerate(methods):
        axes[1, 2].annotate(method, (compressions[i], maes[i]),
                           fontsize=8, ha='right', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="完整的量化评估实验",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/VGGT-1B",
        help="模型名称"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="测试图像文件夹路径"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=10,
        help="最大测试图像数"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comprehensive_evaluation",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu)"
    )

    args = parser.parse_args()

    # 查找测试图像
    image_folder = Path(args.image_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    # 搜索图像
    image_paths = sorted([
        str(p) for p in image_folder.rglob("*")
        if p.is_file() and p.suffix in image_extensions
    ])[:args.max_images]

    if len(image_paths) == 0:
        print(f"❌ 错误: 在 {image_folder} 中未找到图像")
        return

    print(f"找到 {len(image_paths)} 张测试图像")

    # 运行评估
    results = run_comprehensive_evaluation(
        model_name=args.model_name,
        test_images=image_paths,
        output_dir=Path(args.output_dir),
        device=args.device
    )

    print(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
