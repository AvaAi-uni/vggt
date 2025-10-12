#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
量化方案对比脚本

对比以下量化方案：
1. INT8 对称量化 (Symmetric)
2. INT8 非对称量化 (Asymmetric)
3. INT4 分组量化 (Group-wise, 不同组大小)
4. PyTorch 动态量化

生成完整的精度对比报告和可视化
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.quantization import quantize_model, QuantizationConfig, estimate_model_size
from vggt.quantization.advanced_quantizer import (
    quantize_model_advanced,
    AdvancedQuantConfig,
    compare_quantization_methods
)


def run_quantization_comparison(
    model_name: str,
    test_images: list,
    output_dir: Path,
    device: str = "cuda"
):
    """运行完整的量化对比"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VGGT Quantization Methods Comparison")
    print("=" * 80)
    print()

    # 加载原始模型
    print("[1/5] Loading original model...")
    model = VGGT.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    original_size = estimate_model_size(model)
    print(f"  Original model size: {original_size['total_mb']:.2f} MB")

    # 加载测试图像
    print("\n[2/5] Loading test images...")
    images = load_and_preprocess_images(test_images).to(device)
    print(f"  Loaded {len(test_images)} images")

    # 原始模型推理
    print("\n[3/5] Running original model inference...")
    start_time = time.time()
    with torch.no_grad():
        original_output = model(images)
    original_time = time.time() - start_time
    print(f"  Inference time: {original_time:.4f}s")

    # 测试不同量化方案
    print("\n[4/5] Testing quantization methods...")
    results = {}

    # 1. PyTorch 动态量化
    print("\n  [1/5] PyTorch Dynamic INT8...")
    try:
        config = QuantizationConfig(
            quantization_type="dynamic",
            dtype=torch.qint8
        )
        quant_model = quantize_model(model, config)
        quant_model.eval()

        start_time = time.time()
        with torch.no_grad():
            quant_output = quant_model(images.cpu())
        quant_time = time.time() - start_time

        quant_size = estimate_model_size(quant_model)

        # 计算精度指标
        metrics = calculate_metrics(original_output, quant_output)

        results["PyTorch_Dynamic_INT8"] = {
            "model_size_mb": quant_size['total_mb'],
            "inference_time": quant_time,
            "compression_ratio": original_size['total_mb'] / quant_size['total_mb'],
            "speedup": original_time / quant_time,
            "metrics": metrics
        }
        print(f"    Size: {quant_size['total_mb']:.2f} MB | Time: {quant_time:.4f}s | Compression: {results['PyTorch_Dynamic_INT8']['compression_ratio']:.2f}x")
    except Exception as e:
        print(f"    Error: {e}")
        results["PyTorch_Dynamic_INT8"] = {"error": str(e)}

    # 2. INT8 对称量化
    print("\n  [2/5] INT8 Symmetric...")
    try:
        config = AdvancedQuantConfig(
            quant_type="int8_symmetric",
            bits=8,
            device=device
        )
        quant_model = quantize_model_advanced(model, config)
        quant_model = quant_model.to(device)
        quant_model.eval()

        start_time = time.time()
        with torch.no_grad():
            quant_output = quant_model(images)
        quant_time = time.time() - start_time

        quant_size = estimate_model_size(quant_model)

        metrics = calculate_metrics(original_output, quant_output)

        results["INT8_Symmetric"] = {
            "model_size_mb": quant_size['total_mb'],
            "inference_time": quant_time,
            "compression_ratio": original_size['total_mb'] / quant_size['total_mb'],
            "speedup": original_time / quant_time,
            "metrics": metrics
        }
        print(f"    Size: {quant_size['total_mb']:.2f} MB | Time: {quant_time:.4f}s | Compression: {results['INT8_Symmetric']['compression_ratio']:.2f}x")
    except Exception as e:
        print(f"    Error: {e}")
        results["INT8_Symmetric"] = {"error": str(e)}

    # 3. INT8 非对称量化
    print("\n  [3/5] INT8 Asymmetric...")
    try:
        config = AdvancedQuantConfig(
            quant_type="int8_asymmetric",
            bits=8,
            device=device
        )
        quant_model = quantize_model_advanced(model, config)
        quant_model = quant_model.to(device)
        quant_model.eval()

        start_time = time.time()
        with torch.no_grad():
            quant_output = quant_model(images)
        quant_time = time.time() - start_time

        quant_size = estimate_model_size(quant_model)

        metrics = calculate_metrics(original_output, quant_output)

        results["INT8_Asymmetric"] = {
            "model_size_mb": quant_size['total_mb'],
            "inference_time": quant_time,
            "compression_ratio": original_size['total_mb'] / quant_size['total_mb'],
            "speedup": original_time / quant_time,
            "metrics": metrics
        }
        print(f"    Size: {quant_size['total_mb']:.2f} MB | Time: {quant_time:.4f}s | Compression: {results['INT8_Asymmetric']['compression_ratio']:.2f}x")
    except Exception as e:
        print(f"    Error: {e}")
        results["INT8_Asymmetric"] = {"error": str(e)}

    # 4. INT4 分组量化 (Group=128)
    print("\n  [4/5] INT4 Group-128...")
    try:
        config = AdvancedQuantConfig(
            quant_type="int4_group",
            bits=4,
            group_size=128,
            device=device
        )
        quant_model = quantize_model_advanced(model, config)
        quant_model = quant_model.to(device)
        quant_model.eval()

        start_time = time.time()
        with torch.no_grad():
            quant_output = quant_model(images)
        quant_time = time.time() - start_time

        quant_size = estimate_model_size(quant_model)

        metrics = calculate_metrics(original_output, quant_output)

        results["INT4_Group128"] = {
            "model_size_mb": quant_size['total_mb'],
            "inference_time": quant_time,
            "compression_ratio": original_size['total_mb'] / quant_size['total_mb'],
            "speedup": original_time / quant_time,
            "metrics": metrics
        }
        print(f"    Size: {quant_size['total_mb']:.2f} MB | Time: {quant_time:.4f}s | Compression: {results['INT4_Group128']['compression_ratio']:.2f}x")
    except Exception as e:
        print(f"    Error: {e}")
        results["INT4_Group128"] = {"error": str(e)}

    # 5. INT4 分组量化 (Group=64)
    print("\n  [5/5] INT4 Group-64...")
    try:
        config = AdvancedQuantConfig(
            quant_type="int4_group",
            bits=4,
            group_size=64,
            device=device
        )
        quant_model = quantize_model_advanced(model, config)
        quant_model = quant_model.to(device)
        quant_model.eval()

        start_time = time.time()
        with torch.no_grad():
            quant_output = quant_model(images)
        quant_time = time.time() - start_time

        quant_size = estimate_model_size(quant_model)

        metrics = calculate_metrics(original_output, quant_output)

        results["INT4_Group64"] = {
            "model_size_mb": quant_size['total_mb'],
            "inference_time": quant_time,
            "compression_ratio": original_size['total_mb'] / quant_size['total_mb'],
            "speedup": original_time / quant_time,
            "metrics": metrics
        }
        print(f"    Size: {quant_size['total_mb']:.2f} MB | Time: {quant_time:.4f}s | Compression: {results['INT4_Group64']['compression_ratio']:.2f}x")
    except Exception as e:
        print(f"    Error: {e}")
        results["INT4_Group64"] = {"error": str(e)}

    # 添加原始模型信息
    results["Original_FP32"] = {
        "model_size_mb": original_size['total_mb'],
        "inference_time": original_time,
        "compression_ratio": 1.0,
        "speedup": 1.0,
        "metrics": {k: {"mae": 0.0, "mse": 0.0, "psnr": float('inf')} for k in original_output.keys() if k != "images"}
    }

    # 生成报告
    print("\n[5/5] Generating reports...")
    generate_comparison_report(results, output_dir)
    generate_visualizations(results, output_dir)

    print("\n" + "=" * 80)
    print("✅ Comparison Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - comparison_report.json")
    print(f"  - comparison_summary.txt")
    print(f"  - comparison_plots.png")
    print()

    return results


def calculate_metrics(original_output, quantized_output):
    """计算精度指标"""
    metrics = {}

    if isinstance(original_output, dict):
        for key in original_output.keys():
            if key == "images":
                continue

            orig = original_output[key].cpu().float()
            quant = quantized_output[key].cpu().float()

            mse = torch.mean((orig - quant) ** 2).item()
            mae = torch.mean(torch.abs(orig - quant)).item()
            max_diff = torch.max(torch.abs(orig - quant)).item()

            # PSNR
            if mse > 0:
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            else:
                psnr = float('inf')

            # SSIM (simplified)
            orig_mean = torch.mean(orig)
            quant_mean = torch.mean(quant)
            orig_var = torch.var(orig)
            quant_var = torch.var(quant)
            cov = torch.mean((orig - orig_mean) * (quant - quant_mean))

            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            ssim = ((2 * orig_mean * quant_mean + c1) * (2 * cov + c2)) / \
                   ((orig_mean ** 2 + quant_mean ** 2 + c1) * (orig_var + quant_var + c2))

            metrics[key] = {
                "mse": mse,
                "mae": mae,
                "max_diff": max_diff,
                "psnr": psnr,
                "ssim": ssim.item() if isinstance(ssim, torch.Tensor) else ssim
            }

    return metrics


def generate_comparison_report(results, output_dir):
    """生成对比报告"""

    # JSON 报告
    with open(output_dir / "comparison_report.json", 'w') as f:
        # 转换不可序列化的对象
        serializable_results = {}
        for method, data in results.items():
            serializable_data = {}
            for key, value in data.items():
                if key == "config":
                    continue
                elif isinstance(value, (int, float, str, bool, type(None))):
                    serializable_data[key] = value
                elif isinstance(value, dict):
                    serializable_data[key] = value
                else:
                    serializable_data[key] = str(value)
            serializable_results[method] = serializable_data

        json.dump(serializable_results, f, indent=2)

    # 文本报告
    with open(output_dir / "comparison_summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VGGT Quantization Methods Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 表格
        f.write("Method                  | Size(MB) | Compression | Time(s) | Speedup | Depth MAE\n")
        f.write("-" * 80 + "\n")

        for method, data in results.items():
            if "error" in data:
                f.write(f"{method:23} | ERROR: {data['error']}\n")
                continue

            size = data.get('model_size_mb', 0)
            comp = data.get('compression_ratio', 1.0)
            time_val = data.get('inference_time', 0)
            speedup = data.get('speedup', 1.0)

            # 获取深度 MAE
            metrics = data.get('metrics', {})
            depth_mae = metrics.get('depth', {}).get('mae', 0.0)

            f.write(f"{method:23} | {size:8.2f} | {comp:11.2f}x | {time_val:7.4f} | {speedup:7.2f}x | {depth_mae:.6f}\n")

        f.write("\n" + "=" * 80 + "\n\n")

        # 详细指标
        f.write("Detailed Metrics:\n\n")
        for method, data in results.items():
            if "error" in data:
                continue

            f.write(f"{method}:\n")
            f.write(f"  Model Size: {data.get('model_size_mb', 0):.2f} MB\n")
            f.write(f"  Compression Ratio: {data.get('compression_ratio', 1.0):.2f}x\n")
            f.write(f"  Inference Time: {data.get('inference_time', 0):.4f}s\n")
            f.write(f"  Speedup: {data.get('speedup', 1.0):.2f}x\n")

            metrics = data.get('metrics', {})
            for output_name, output_metrics in metrics.items():
                f.write(f"  {output_name}:\n")
                for metric_name, metric_value in output_metrics.items():
                    f.write(f"    {metric_name}: {metric_value:.6f}\n")
            f.write("\n")


def generate_visualizations(results, output_dir):
    """生成可视化图表"""

    # 过滤掉有错误的结果
    valid_results = {k: v for k, v in results.items() if "error" not in v}

    if len(valid_results) == 0:
        print("No valid results to visualize")
        return

    methods = list(valid_results.keys())
    sizes = [valid_results[m]['model_size_mb'] for m in methods]
    times = [valid_results[m]['inference_time'] for m in methods]
    compressions = [valid_results[m]['compression_ratio'] for m in methods]

    # 获取深度 MAE
    depth_maes = []
    for m in methods:
        metrics = valid_results[m].get('metrics', {})
        depth_mae = metrics.get('depth', {}).get('mae', 0.0)
        depth_maes.append(depth_mae)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 模型大小对比
    axes[0, 0].bar(methods, sizes, color='skyblue')
    axes[0, 0].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Size (MB)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. 推理时间对比
    axes[0, 1].bar(methods, times, color='lightcoral')
    axes[0, 1].set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. 压缩率对比
    axes[1, 0].bar(methods, compressions, color='lightgreen')
    axes[1, 0].set_title('Compression Ratio', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Compression Ratio (x)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. 精度对比 (深度 MAE)
    axes[1, 1].bar(methods, depth_maes, color='gold')
    axes[1, 1].set_title('Depth Prediction Accuracy (MAE)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_plots.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: comparison_plots.png")


def main():
    parser = argparse.ArgumentParser(description="Compare quantization methods for VGGT")

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/VGGT-1B",
        help="Model name"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to test images"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=5,
        help="Maximum number of images for testing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/quantization_comparison",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device"
    )

    args = parser.parse_args()

    # 获取测试图像
    image_folder = Path(args.image_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    # 首先检查当前目录是否有图像
    image_paths = sorted([
        str(p) for p in image_folder.iterdir()
        if p.is_file() and p.suffix in image_extensions
    ])

    # 如果当前目录没有图像，检查是否有 dslr_images_undistorted 子目录
    if len(image_paths) == 0:
        dslr_folder = image_folder / "dslr_images_undistorted"
        if dslr_folder.exists() and dslr_folder.is_dir():
            print(f"No images in {image_folder}, checking {dslr_folder}...")
            image_paths = sorted([
                str(p) for p in dslr_folder.iterdir()
                if p.is_file() and p.suffix in image_extensions
            ])

    # 如果还是没有图像，递归搜索
    if len(image_paths) == 0:
        print(f"Searching for images recursively in {image_folder}...")
        image_paths = sorted([
            str(p) for p in image_folder.rglob("*")
            if p.is_file() and p.suffix in image_extensions
        ])

    # 限制图像数量
    image_paths = image_paths[:args.max_images]

    if len(image_paths) == 0:
        print(f"ERROR: No images found in {image_folder}")
        print(f"Searched for extensions: {image_extensions}")
        print(f"Directory contents:")
        for item in image_folder.iterdir():
            print(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
        return

    print(f"Found {len(image_paths)} test images")
    print(f"First image: {image_paths[0]}")

    # 运行对比
    results = run_quantization_comparison(
        model_name=args.model_name,
        test_images=image_paths,
        output_dir=Path(args.output_dir),
        device=args.device
    )


if __name__ == "__main__":
    main()
