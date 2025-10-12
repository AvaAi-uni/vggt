#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
可视化量化模型的推理结果

功能：
- 深度图可视化
- 点云可视化
- 相机轨迹可视化
- 对比原始模型和量化模型
- 生成 HTML 报告

用法:
    python scripts/visualize_results.py \
        --model_path /workspace/models/vggt_int8_dynamic.pt \
        --image_folder /workspace/data/eth3d/courtyard/images \
        --output_dir /workspace/visualizations \
        --max_images 10
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from PIL import Image
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def visualize_depth_maps(depth, depth_conf, images, output_dir, prefix=""):
    """可视化深度图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    B, S, H, W, _ = depth.shape

    for b in range(B):
        for s in range(S):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 原始图像
            img = images[b, s].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            axes[0].imshow(img)
            axes[0].set_title(f'Original Image (Batch {b}, Frame {s})')
            axes[0].axis('off')

            # 深度图
            depth_map = depth[b, s, :, :, 0].cpu().numpy()
            im1 = axes[1].imshow(depth_map, cmap='turbo')
            axes[1].set_title(f'Depth Map (min: {depth_map.min():.2f}, max: {depth_map.max():.2f})')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])

            # 置信度
            conf_map = depth_conf[b, s].cpu().numpy()
            im2 = axes[2].imshow(conf_map, cmap='viridis', vmin=0, vmax=1)
            axes[2].set_title(f'Confidence (mean: {conf_map.mean():.3f})')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2])

            plt.tight_layout()
            output_path = output_dir / f'{prefix}depth_b{b}_s{s}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {output_path}")


def visualize_point_cloud(world_points, world_points_conf, images, output_dir, prefix="", max_points=50000):
    """可视化点云（2D投影）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    B, S, H, W, _ = world_points.shape

    for b in range(B):
        # 合并所有帧的点云
        all_points = []
        all_colors = []
        all_conf = []

        for s in range(S):
            points = world_points[b, s].cpu().numpy().reshape(-1, 3)
            conf = world_points_conf[b, s].cpu().numpy().reshape(-1)

            # 获取颜色
            img = images[b, s].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)

            # 过滤置信度低的点
            mask = conf > 0.5
            points = points[mask]
            colors = img[mask]
            conf_filtered = conf[mask]

            all_points.append(points)
            all_colors.append(colors)
            all_conf.append(conf_filtered)

        # 合并
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_conf = np.concatenate(all_conf, axis=0)

        # 随机采样以减少点数
        if len(all_points) > max_points:
            indices = np.random.choice(len(all_points), max_points, replace=False)
            all_points = all_points[indices]
            all_colors = all_colors[indices]
            all_conf = all_conf[indices]

        # 创建 3D 投影图
        fig = plt.figure(figsize=(15, 5))

        # XY 平面
        ax1 = fig.add_subplot(131)
        scatter1 = ax1.scatter(all_points[:, 0], all_points[:, 1],
                              c=all_colors, s=1, alpha=0.5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Top View (XY)')
        ax1.set_aspect('equal')

        # XZ 平面
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(all_points[:, 0], all_points[:, 2],
                              c=all_colors, s=1, alpha=0.5)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('Side View (XZ)')
        ax2.set_aspect('equal')

        # YZ 平面
        ax3 = fig.add_subplot(133)
        scatter3 = ax3.scatter(all_points[:, 1], all_points[:, 2],
                              c=all_colors, s=1, alpha=0.5)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_title('Front View (YZ)')
        ax3.set_aspect('equal')

        plt.tight_layout()
        output_path = output_dir / f'{prefix}pointcloud_b{b}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")

        # 保存点云数据（PLY 格式）
        ply_path = output_dir / f'{prefix}pointcloud_b{b}.ply'
        save_ply(all_points, all_colors, ply_path)
        print(f"  Saved PLY: {ply_path}")


def save_ply(points, colors, output_path):
    """保存点云为 PLY 格式"""
    colors = (colors * 255).astype(np.uint8)

    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} ")
            f.write(f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")


def visualize_camera_poses(extrinsic, intrinsic, output_dir, prefix=""):
    """可视化相机位姿"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    B, S, _, _ = extrinsic.shape

    for b in range(B):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 提取相机中心
        camera_centers = []
        for s in range(S):
            # 从 extrinsic 中提取旋转和平移
            R = extrinsic[b, s, :3, :3].cpu().numpy()
            t = extrinsic[b, s, :3, 3].cpu().numpy()

            # 相机中心：C = -R^T * t
            C = -R.T @ t
            camera_centers.append(C)

        camera_centers = np.array(camera_centers)

        # 绘制相机轨迹
        ax.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
                'b-', linewidth=2, label='Camera Trajectory')

        # 绘制相机位置
        ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
                  c='r', s=50, label='Camera Positions')

        # 标注起始点和终点
        ax.scatter(camera_centers[0, 0], camera_centers[0, 1], camera_centers[0, 2],
                  c='g', s=200, marker='*', label='Start')
        ax.scatter(camera_centers[-1, 0], camera_centers[-1, 1], camera_centers[-1, 2],
                  c='orange', s=200, marker='*', label='End')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Camera Trajectory (Batch {b}, {S} frames)')
        ax.legend()

        plt.tight_layout()
        output_path = output_dir / f'{prefix}camera_trajectory_b{b}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")


def compare_models(original_model, quantized_model, test_images, output_dir, device="cuda"):
    """对比原始模型和量化模型"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Comparing original vs quantized models...")

    # 原始模型推理
    original_model.eval()
    with torch.no_grad():
        orig_pred = original_model(test_images)

    # 量化模型推理
    quantized_model.eval()
    quantized_model = quantized_model.to("cpu")
    test_images_cpu = test_images.cpu()
    with torch.no_grad():
        quant_pred = quantized_model(test_images_cpu)

    # 计算差异
    depth_diff = torch.abs(orig_pred["depth"].cpu() - quant_pred["depth"]).mean().item()
    points_diff = torch.abs(orig_pred["world_points"].cpu() - quant_pred["world_points"]).mean().item()

    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 原始模型深度
    orig_depth = orig_pred["depth"][0, 0, :, :, 0].cpu().numpy()
    axes[0, 0].imshow(orig_depth, cmap='turbo')
    axes[0, 0].set_title('Original Model - Depth')
    axes[0, 0].axis('off')

    # 量化模型深度
    quant_depth = quant_pred["depth"][0, 0, :, :, 0].cpu().numpy()
    axes[0, 1].imshow(quant_depth, cmap='turbo')
    axes[0, 1].set_title('Quantized Model - Depth')
    axes[0, 1].axis('off')

    # 深度差异
    depth_diff_map = np.abs(orig_depth - quant_depth)
    im = axes[0, 2].imshow(depth_diff_map, cmap='hot')
    axes[0, 2].set_title(f'Depth Difference (MAE: {depth_diff:.4f})')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])

    # 原始模型点云 (X坐标)
    orig_points = orig_pred["world_points"][0, 0, :, :, 0].cpu().numpy()
    axes[1, 0].imshow(orig_points, cmap='viridis')
    axes[1, 0].set_title('Original Model - Points (X)')
    axes[1, 0].axis('off')

    # 量化模型点云 (X坐标)
    quant_points = quant_pred["world_points"][0, 0, :, :, 0].cpu().numpy()
    axes[1, 1].imshow(quant_points, cmap='viridis')
    axes[1, 1].set_title('Quantized Model - Points (X)')
    axes[1, 1].axis('off')

    # 点云差异
    points_diff_map = np.abs(orig_points - quant_points)
    im2 = axes[1, 2].imshow(points_diff_map, cmap='hot')
    axes[1, 2].set_title(f'Points Difference (MAE: {points_diff:.4f})')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2])

    plt.tight_layout()
    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved comparison: {output_path}")

    return {
        "depth_mae": depth_diff,
        "points_mae": points_diff,
    }


def generate_html_report(output_dir, metrics, image_files):
    """生成 HTML 报告"""
    output_dir = Path(output_dir)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VGGT Quantization Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            .metrics {{
                background-color: #e8f5e9;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .metric-item {{
                margin: 10px 0;
                font-size: 16px;
            }}
            .metric-label {{
                font-weight: bold;
                color: #2e7d32;
            }}
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .image-item {{
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow: hidden;
                background-color: white;
            }}
            .image-item img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            .image-caption {{
                padding: 10px;
                background-color: #f9f9f9;
                font-size: 14px;
                text-align: center;
            }}
            .timestamp {{
                color: #888;
                font-size: 12px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 VGGT INT8 Quantization - Visualization Results</h1>

            <div class="metrics">
                <h2>📊 Performance Metrics</h2>
                <div class="metric-item">
                    <span class="metric-label">Model Type:</span>
                    {metrics.get('model_type', 'INT8 Dynamic Quantization')}
                </div>
                <div class="metric-item">
                    <span class="metric-label">Number of Images:</span>
                    {metrics.get('num_images', 'N/A')}
                </div>
                <div class="metric-item">
                    <span class="metric-label">Inference Time:</span>
                    {metrics.get('inference_time', 'N/A')} seconds
                </div>
                <div class="metric-item">
                    <span class="metric-label">Average Time per Image:</span>
                    {metrics.get('avg_time', 'N/A')} seconds
                </div>
            </div>

            <h2>🖼️ Depth Maps</h2>
            <div class="image-grid">
    """

    # 添加深度图
    depth_images = sorted(output_dir.glob("*depth_*.png"))
    for img_path in depth_images[:12]:  # 最多显示 12 张
        rel_path = img_path.name
        html_content += f"""
                <div class="image-item">
                    <img src="{rel_path}" alt="Depth visualization">
                    <div class="image-caption">{img_path.stem}</div>
                </div>
        """

    html_content += """
            </div>

            <h2>☁️ Point Clouds</h2>
            <div class="image-grid">
    """

    # 添加点云图
    pc_images = sorted(output_dir.glob("*pointcloud_*.png"))
    for img_path in pc_images:
        rel_path = img_path.name
        html_content += f"""
                <div class="image-item">
                    <img src="{rel_path}" alt="Point cloud visualization">
                    <div class="image-caption">{img_path.stem}</div>
                </div>
        """

    html_content += """
            </div>

            <h2>📹 Camera Trajectory</h2>
            <div class="image-grid">
    """

    # 添加相机轨迹
    cam_images = sorted(output_dir.glob("*camera_*.png"))
    for img_path in cam_images:
        rel_path = img_path.name
        html_content += f"""
                <div class="image-item">
                    <img src="{rel_path}" alt="Camera trajectory">
                    <div class="image-caption">{img_path.stem}</div>
                </div>
        """

    html_content += """
            </div>

            <h2>🔍 Model Comparison</h2>
            <div class="image-grid">
    """

    # 添加模型对比
    comp_images = sorted(output_dir.glob("*comparison*.png"))
    for img_path in comp_images:
        rel_path = img_path.name
        html_content += f"""
                <div class="image-item">
                    <img src="{rel_path}" alt="Model comparison">
                    <div class="image-caption">{img_path.stem}</div>
                </div>
        """

    html_content += f"""
            </div>

            <div class="timestamp">
                Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """

    html_path = output_dir / "index.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✅ HTML report generated: {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Visualize VGGT quantization results")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to quantized model")
    parser.add_argument("--original_model", type=str, default=None, help="Path to original model for comparison")

    # Input arguments
    parser.add_argument("--image_folder", type=str, required=True, help="Path to images folder")
    parser.add_argument("--max_images", type=int, default=10, help="Maximum number of images to process")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="/workspace/visualizations", help="Output directory")

    # Visualization options
    parser.add_argument("--skip_depth", action="store_true", help="Skip depth visualization")
    parser.add_argument("--skip_points", action="store_true", help="Skip point cloud visualization")
    parser.add_argument("--skip_camera", action="store_true", help="Skip camera visualization")
    parser.add_argument("--skip_comparison", action="store_true", help="Skip model comparison")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VGGT Quantization Results Visualization")
    print("=" * 60)
    print()

    # 加载图像
    print("Loading images...")
    image_folder = Path(args.image_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_paths = sorted([
        str(p) for p in image_folder.iterdir()
        if p.suffix in image_extensions
    ])[:args.max_images]

    print(f"  Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("ERROR: No images found!")
        return

    images = load_and_preprocess_images(image_paths).to(args.device)

    # 加载量化模型
    print("\nLoading quantized model...")
    model = VGGT()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(args.device)
    model.eval()
    print("  Model loaded")

    # 运行推理
    print("\nRunning inference...")
    import time
    start_time = time.time()

    with torch.no_grad():
        predictions = model(images)

    inference_time = time.time() - start_time
    print(f"  Inference completed in {inference_time:.3f}s")
    print(f"  Average time per image: {inference_time/len(image_paths):.3f}s")

    # 提取结果
    depth = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    world_points = predictions["world_points"]
    world_points_conf = predictions["world_points_conf"]
    pose_enc = predictions["pose_enc"]

    # 获取相机参数
    H, W = images.shape[-2:]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (H, W))

    # 可视化
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    if not args.skip_depth:
        print("\n[1/4] Visualizing depth maps...")
        visualize_depth_maps(depth, depth_conf, images, output_dir, prefix="quant_")

    if not args.skip_points:
        print("\n[2/4] Visualizing point clouds...")
        visualize_point_cloud(world_points, world_points_conf, images, output_dir, prefix="quant_")

    if not args.skip_camera:
        print("\n[3/4] Visualizing camera poses...")
        visualize_camera_poses(extrinsic, intrinsic, output_dir, prefix="quant_")

    # 模型对比
    comparison_metrics = {}
    if args.original_model and not args.skip_comparison:
        print("\n[4/4] Comparing with original model...")
        original_model = VGGT.from_pretrained(args.original_model).to(args.device)
        comparison_metrics = compare_models(original_model, model, images, output_dir, args.device)
        print(f"  Depth MAE: {comparison_metrics['depth_mae']:.6f}")
        print(f"  Points MAE: {comparison_metrics['points_mae']:.6f}")

    # 生成 HTML 报告
    print("\n" + "=" * 60)
    print("Generating HTML Report")
    print("=" * 60)

    metrics = {
        "model_type": "INT8 Quantized Model",
        "num_images": len(image_paths),
        "inference_time": f"{inference_time:.3f}",
        "avg_time": f"{inference_time/len(image_paths):.3f}",
        **comparison_metrics,
    }

    html_path = generate_html_report(output_dir, metrics, image_paths)

    # 保存元数据
    metadata = {
        "model_path": args.model_path,
        "image_folder": args.image_folder,
        "num_images": len(image_paths),
        "inference_time": inference_time,
        "avg_time_per_image": inference_time / len(image_paths),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
        **comparison_metrics,
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ Visualization Complete!")
    print("=" * 60)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"🌐 Open HTML report: {html_path}")
    print(f"📊 Metadata: {output_dir / 'metadata.json'}")
    print()
    print("To download results to your local machine:")
    print(f"  scp -r -P <PORT> root@<POD_IP>:{output_dir} ./local_results/")
    print()


if __name__ == "__main__":
    main()
