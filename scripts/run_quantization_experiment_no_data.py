#!/usr/bin/env python3
"""
VGGT Quantization Experiment - No Dataset Required
完整的量化实验脚本 - 使用合成数据，无需真实数据集

这个脚本实现了多维度的量化对比实验：
1. 量化精度维度: INT8 vs INT4
2. 量化粒度维度: Per-Tensor vs Per-Channel vs Group-wise
3. 量化对称性维度: Symmetric vs Asymmetric
4. 性能维度: 模型大小、推理速度、精度损失

适用于课程项目研究，可在RunPod等云平台直接运行
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from vggt.quantization.comprehensive_quantizer import (
    quantize_model_comprehensive,
    get_all_quantization_configs,
    estimate_model_size,
    measure_inference_time,
    QuantizationConfig
)


class SyntheticDataGenerator:
    """合成数据生成器 - 模拟VGGT输入"""

    def __init__(self, batch_size=1, num_frames=5, img_size=518):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.img_size = img_size

    def generate_images(self, device="cuda"):
        """生成合成图像序列"""
        # 生成随机图像，范围 [0, 1]
        images = torch.rand(
            self.batch_size,
            self.num_frames,
            3,
            self.img_size,
            self.img_size,
            device=device
        )
        return images

    def generate_query_points(self, num_points=100, device="cuda"):
        """生成查询点用于track"""
        # 随机生成像素坐标
        query_points = torch.rand(
            self.batch_size,
            num_points,
            2,
            device=device
        ) * self.img_size
        return query_points


class QuantizationExperiment:
    """量化实验管理器"""

    def __init__(self, model_name="facebook/VGGT-1B", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.results = {}

    def load_model(self):
        """加载原始模型"""
        print("\n" + "="*80)
        print("Loading VGGT Model")
        print("="*80)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")

        try:
            model = VGGT.from_pretrained(self.model_name)
            model = model.to(self.device)
            model.eval()
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("\nTrying to create model from scratch...")
            # If pretrained model fails, create from scratch
            model = VGGT(img_size=518, patch_size=14, embed_dim=1024)
            model = model.to(self.device)
            model.eval()
            print("✓ Model created from scratch")
            return model

    def benchmark_baseline(self, model, test_images):
        """Baseline性能测试"""
        print("\n" + "="*80)
        print("Baseline Performance (FP32)")
        print("="*80)

        # 模型大小
        size_info = estimate_model_size(model)
        print(f"Model Size: {size_info['total_mb']:.2f} MB")
        print(f"  Parameters: {size_info['params_mb']:.2f} MB")
        print(f"  Buffers: {size_info['buffers_mb']:.2f} MB")

        # 推理时间
        time_info = measure_inference_time(
            model, test_images,
            warmup=3, iterations=10,
            device=self.device
        )
        print(f"Inference Time: {time_info['mean']:.4f}s ± {time_info['std']:.4f}s")

        # 获取baseline输出
        with torch.no_grad():
            baseline_output = model(test_images)

        self.baseline_output = baseline_output
        self.baseline_size = size_info['total_mb']
        self.baseline_time = time_info['mean']

        return {
            "size_mb": size_info['total_mb'],
            "inference_time": time_info['mean'],
            "inference_std": time_info['std'],
            "compression_ratio": 1.0,
            "speedup": 1.0
        }

    def evaluate_quantized_model(self, quant_model, config, test_images):
        """评估量化模型"""
        print(f"\n{'='*80}")
        print(f"Evaluating: {config.name}")
        print(f"{'='*80}")

        results = {}

        try:
            quant_model = quant_model.to(self.device)
            quant_model.eval()

            # 1. 模型大小
            size_info = estimate_model_size(quant_model)
            compression_ratio = self.baseline_size / size_info['total_mb']
            print(f"Model Size: {size_info['total_mb']:.2f} MB (Compression: {compression_ratio:.2f}x)")

            # 2. 推理时间
            time_info = measure_inference_time(
                quant_model, test_images,
                warmup=3, iterations=10,
                device=self.device
            )
            speedup = self.baseline_time / time_info['mean']
            print(f"Inference Time: {time_info['mean']:.4f}s (Speedup: {speedup:.2f}x)")

            # 3. 精度评估
            with torch.no_grad():
                quant_output = quant_model(test_images)

            accuracy_metrics = self.calculate_accuracy_metrics(
                self.baseline_output, quant_output
            )

            print(f"Accuracy Metrics:")
            for output_name, metrics in accuracy_metrics.items():
                print(f"  {output_name}:")
                print(f"    MAE: {metrics['mae']:.6f}")
                print(f"    MSE: {metrics['mse']:.6f}")
                print(f"    PSNR: {metrics['psnr']:.2f} dB")

            results = {
                "config_name": config.name,
                "quant_type": config.quant_type,
                "bits": config.bits,
                "size_mb": size_info['total_mb'],
                "compression_ratio": compression_ratio,
                "inference_time_mean": time_info['mean'],
                "inference_time_std": time_info['std'],
                "speedup": speedup,
                "accuracy_metrics": accuracy_metrics,
                "status": "success"
            }

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results = {
                "config_name": config.name,
                "status": "failed",
                "error": str(e)
            }

        return results

    def calculate_accuracy_metrics(self, baseline_output, quant_output):
        """计算精度指标"""
        metrics = {}

        if not isinstance(baseline_output, dict) or not isinstance(quant_output, dict):
            return metrics

        for key in baseline_output.keys():
            if key == "images" or key.endswith("_list"):
                continue

            if not isinstance(baseline_output[key], torch.Tensor):
                continue
            if not isinstance(quant_output[key], torch.Tensor):
                continue

            baseline_tensor = baseline_output[key].detach().cpu().float()
            quant_tensor = quant_output[key].detach().cpu().float()

            if baseline_tensor.shape != quant_tensor.shape:
                continue

            # MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(baseline_tensor - quant_tensor)).item()

            # MSE (Mean Squared Error)
            mse = torch.mean((baseline_tensor - quant_tensor) ** 2).item()

            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(mse)

            # PSNR (Peak Signal-to-Noise Ratio)
            if mse > 0:
                max_val = torch.max(torch.abs(baseline_tensor)).item()
                psnr = 10 * np.log10((max_val ** 2) / mse)
            else:
                psnr = float('inf')

            # Cosine Similarity
            baseline_flat = baseline_tensor.flatten()
            quant_flat = quant_tensor.flatten()
            cos_sim = torch.nn.functional.cosine_similarity(
                baseline_flat.unsqueeze(0),
                quant_flat.unsqueeze(0)
            ).item()

            metrics[key] = {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "psnr": psnr,
                "cosine_similarity": cos_sim
            }

        return metrics

    def run_comprehensive_experiment(self, test_images):
        """运行完整的量化实验"""
        print("\n" + "="*80)
        print("VGGT Comprehensive Quantization Experiment")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Load model
        original_model = self.load_model()

        # 2. Baseline benchmark
        baseline_results = self.benchmark_baseline(original_model, test_images)
        self.results["Baseline_FP32"] = baseline_results

        # 3. Get quantization configurations
        configs = get_all_quantization_configs(device=self.device)

        print(f"\nTotal quantization schemes to test: {len(configs)}")

        # 4. Test each quantization scheme
        for idx, config in enumerate(configs, 1):
            print(f"\n[{idx}/{len(configs)}] Testing: {config.name}")

            try:
                # Quantize model
                print("  Quantizing model...")
                quant_model = quantize_model_comprehensive(original_model, config)

                # Evaluate
                results = self.evaluate_quantized_model(quant_model, config, test_images)
                self.results[config.name] = results

                # Clean up
                del quant_model
                torch.cuda.empty_cache() if self.device == "cuda" else None

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                self.results[config.name] = {
                    "config_name": config.name,
                    "status": "failed",
                    "error": str(e)
                }

        return self.results

    def save_results(self, output_dir):
        """保存实验结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print("Saving Results")
        print(f"{'='*80}")

        # 1. Save JSON
        json_path = output_dir / "quantization_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved: {json_path}")

        # 2. Save summary report
        report_path = output_dir / "quantization_report.txt"
        self.generate_text_report(report_path)
        print(f"✓ Saved: {report_path}")

        # 3. Save visualizations
        viz_path = output_dir / "quantization_plots.png"
        self.generate_visualizations(viz_path)
        print(f"✓ Saved: {viz_path}")

        # 4. Save detailed metrics CSV
        csv_path = output_dir / "quantization_metrics.csv"
        self.save_csv(csv_path)
        print(f"✓ Saved: {csv_path}")

        print(f"\nAll results saved to: {output_dir}")

    def generate_text_report(self, output_path):
        """生成文本报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("VGGT Quantization Experiment Report\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Device: {self.device}\n\n")

            # Summary table
            f.write("="*80 + "\n")
            f.write("Summary Table\n")
            f.write("="*80 + "\n\n")
            f.write(f"{'Method':<30} {'Size(MB)':<12} {'Compress':<10} {'Time(s)':<10} {'Speedup':<10}\n")
            f.write("-"*80 + "\n")

            for method, data in self.results.items():
                if data.get("status") == "failed":
                    f.write(f"{method:<30} FAILED\n")
                    continue

                size = data.get('size_mb', 0)
                comp = data.get('compression_ratio', 1.0)
                time_val = data.get('inference_time_mean', 0)
                speedup = data.get('speedup', 1.0)

                f.write(f"{method:<30} {size:<12.2f} {comp:<10.2f}x {time_val:<10.4f} {speedup:<10.2f}x\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Detailed Metrics\n")
            f.write("="*80 + "\n\n")

            for method, data in self.results.items():
                if data.get("status") == "failed":
                    continue

                f.write(f"\n{method}:\n")
                f.write(f"  Configuration:\n")
                f.write(f"    Quantization Type: {data.get('quant_type', 'N/A')}\n")
                f.write(f"    Bits: {data.get('bits', 'N/A')}\n")
                f.write(f"  Performance:\n")
                f.write(f"    Model Size: {data.get('size_mb', 0):.2f} MB\n")
                f.write(f"    Compression Ratio: {data.get('compression_ratio', 1.0):.2f}x\n")
                f.write(f"    Inference Time: {data.get('inference_time_mean', 0):.4f}s\n")
                f.write(f"    Speedup: {data.get('speedup', 1.0):.2f}x\n")

                if 'accuracy_metrics' in data:
                    f.write(f"  Accuracy Metrics:\n")
                    for output_name, metrics in data['accuracy_metrics'].items():
                        f.write(f"    {output_name}:\n")
                        for metric_name, value in metrics.items():
                            f.write(f"      {metric_name}: {value:.6f}\n")

    def generate_visualizations(self, output_path):
        """生成可视化图表"""
        # Filter valid results
        valid_results = {
            k: v for k, v in self.results.items()
            if v.get("status") != "failed"
        }

        if len(valid_results) < 2:
            print("Not enough valid results for visualization")
            return

        methods = list(valid_results.keys())
        sizes = [valid_results[m].get('size_mb', 0) for m in methods]
        times = [valid_results[m].get('inference_time_mean', 0) for m in methods]
        compressions = [valid_results[m].get('compression_ratio', 1.0) for m in methods]
        speedups = [valid_results[m].get('speedup', 1.0) for m in methods]

        # Extract accuracy metrics (use depth as example)
        maes = []
        for m in methods:
            metrics = valid_results[m].get('accuracy_metrics', {})
            depth_metrics = metrics.get('depth', {})
            maes.append(depth_metrics.get('mae', 0))

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Model Size
        axes[0, 0].bar(range(len(methods)), sizes, color='skyblue')
        axes[0, 0].set_title('Model Size (MB)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Size (MB)')
        axes[0, 0].set_xticks(range(len(methods)))
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Compression Ratio
        axes[0, 1].bar(range(len(methods)), compressions, color='lightgreen')
        axes[0, 1].set_title('Compression Ratio', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Compression (x)')
        axes[0, 1].set_xticks(range(len(methods)))
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline')

        # 3. Inference Time
        axes[0, 2].bar(range(len(methods)), times, color='lightcoral')
        axes[0, 2].set_title('Inference Time (s)', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].set_xticks(range(len(methods)))
        axes[0, 2].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 2].grid(axis='y', alpha=0.3)

        # 4. Speedup
        axes[1, 0].bar(range(len(methods)), speedups, color='gold')
        axes[1, 0].set_title('Inference Speedup', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Speedup (x)')
        axes[1, 0].set_xticks(range(len(methods)))
        axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline')

        # 5. Accuracy (MAE)
        axes[1, 1].bar(range(len(methods)), maes, color='plum')
        axes[1, 1].set_title('Depth Prediction MAE (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_xticks(range(len(methods)))
        axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 1].grid(axis='y', alpha=0.3)

        # 6. Size vs Accuracy tradeoff
        axes[1, 2].scatter(compressions, maes, s=100, alpha=0.6, c=range(len(methods)), cmap='viridis')
        for i, method in enumerate(methods):
            axes[1, 2].annotate(method, (compressions[i], maes[i]),
                               fontsize=8, rotation=45, ha='right')
        axes[1, 2].set_title('Compression vs Accuracy Tradeoff', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Compression Ratio (x)')
        axes[1, 2].set_ylabel('MAE (Lower is Better)')
        axes[1, 2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_csv(self, output_path):
        """保存CSV格式的详细指标"""
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'Method', 'Status', 'Quant_Type', 'Bits',
                'Size_MB', 'Compression_Ratio',
                'Inference_Time_Mean', 'Inference_Time_Std', 'Speedup',
                'Depth_MAE', 'Depth_MSE', 'Depth_PSNR',
                'World_Points_MAE', 'World_Points_MSE', 'World_Points_PSNR'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for method, data in self.results.items():
                row = {
                    'Method': method,
                    'Status': data.get('status', 'success'),
                    'Quant_Type': data.get('quant_type', ''),
                    'Bits': data.get('bits', ''),
                    'Size_MB': data.get('size_mb', 0),
                    'Compression_Ratio': data.get('compression_ratio', 1.0),
                    'Inference_Time_Mean': data.get('inference_time_mean', 0),
                    'Inference_Time_Std': data.get('inference_time_std', 0),
                    'Speedup': data.get('speedup', 1.0),
                }

                if 'accuracy_metrics' in data:
                    metrics = data['accuracy_metrics']
                    if 'depth' in metrics:
                        row['Depth_MAE'] = metrics['depth'].get('mae', 0)
                        row['Depth_MSE'] = metrics['depth'].get('mse', 0)
                        row['Depth_PSNR'] = metrics['depth'].get('psnr', 0)
                    if 'world_points' in metrics:
                        row['World_Points_MAE'] = metrics['world_points'].get('mae', 0)
                        row['World_Points_MSE'] = metrics['world_points'].get('mse', 0)
                        row['World_Points_PSNR'] = metrics['world_points'].get('psnr', 0)

                writer.writerow(row)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="VGGT Quantization Experiment - No Dataset Required"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/VGGT-1B",
        help="Model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/quantization_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="Number of frames in sequence"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=518,
        help="Image size"
    )

    args = parser.parse_args()

    print("="*80)
    print("VGGT Quantization Experiment (No Dataset Required)")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Frames: {args.num_frames}")
    print(f"Image Size: {args.img_size}")
    print("="*80)

    # Generate synthetic data
    print("\nGenerating synthetic test data...")
    data_gen = SyntheticDataGenerator(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        img_size=args.img_size
    )
    test_images = data_gen.generate_images(device=args.device)
    print(f"✓ Generated images: {test_images.shape}")

    # Run experiment
    experiment = QuantizationExperiment(
        model_name=args.model_name,
        device=args.device
    )

    results = experiment.run_comprehensive_experiment(test_images)

    # Save results
    experiment.save_results(args.output_dir)

    print("\n" + "="*80)
    print("Experiment Completed Successfully!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nFiles generated:")
    print("  - quantization_results.json    (Raw data)")
    print("  - quantization_report.txt      (Summary report)")
    print("  - quantization_plots.png       (Visualizations)")
    print("  - quantization_metrics.csv     (Detailed metrics)")
    print("\n")


if __name__ == "__main__":
    main()
