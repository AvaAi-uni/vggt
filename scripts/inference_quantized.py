#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
示例脚本：使用量化后的 VGGT 模型进行推理

用法:
    python scripts/inference_quantized.py \
        --model_path models/vggt_int8_dynamic.pt \
        --image_folder data/eth3d/courtyard/images \
        --output_dir outputs/courtyard
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_quantized_model(model_path: str, device: str = "cuda") -> VGGT:
    """
    加载量化后的模型

    Args:
        model_path: 量化模型路径
        device: 运行设备

    Returns:
        加载好的模型
    """
    logger.info(f"Loading quantized model from: {model_path}")

    # 创建模型架构
    model = VGGT()

    # 加载量化权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 移到设备
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def process_images(
    model: VGGT,
    image_paths: list,
    device: str = "cuda",
) -> dict:
    """
    处理一组图像

    Args:
        model: VGGT 模型
        image_paths: 图像路径列表
        device: 运行设备

    Returns:
        预测结果字典
    """
    logger.info(f"Processing {len(image_paths)} images...")

    # 加载和预处理图像
    images = load_and_preprocess_images(image_paths).to(device)

    # 推理
    start_time = time.time()
    with torch.no_grad():
        predictions = model(images)
    inference_time = time.time() - start_time

    logger.info(f"Inference completed in {inference_time:.3f}s ({inference_time/len(image_paths):.3f}s per image)")

    return predictions, inference_time


def save_results(
    predictions: dict,
    output_dir: Path,
    image_paths: list,
    save_depth: bool = True,
    save_points: bool = True,
    save_cameras: bool = True,
):
    """
    保存预测结果

    Args:
        predictions: 模型预测结果
        output_dir: 输出目录
        image_paths: 输入图像路径
        save_depth: 是否保存深度图
        save_points: 是否保存点云
        save_cameras: 是否保存相机参数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to: {output_dir}")

    # 提取预测结果
    pose_enc = predictions["pose_enc"].cpu().numpy()
    depth = predictions["depth"].cpu().numpy()
    world_points = predictions["world_points"].cpu().numpy()
    depth_conf = predictions["depth_conf"].cpu().numpy()
    world_points_conf = predictions["world_points_conf"].cpu().numpy()

    B, S = depth.shape[:2]

    # 保存相机参数
    if save_cameras:
        # 获取相机内外参
        H, W = predictions["images"].shape[-2:]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            (H, W)
        )

        cameras_file = output_dir / "cameras.npz"
        np.savez(
            cameras_file,
            pose_enc=pose_enc,
            extrinsic=extrinsic.cpu().numpy(),
            intrinsic=intrinsic.cpu().numpy(),
            image_paths=[str(p) for p in image_paths],
        )
        logger.info(f"Saved camera parameters to: {cameras_file}")

    # 保存深度图
    if save_depth:
        depth_dir = output_dir / "depth"
        depth_dir.mkdir(exist_ok=True)

        for b in range(B):
            for s in range(S):
                depth_map = depth[b, s, ..., 0]
                confidence = depth_conf[b, s]

                # 保存为 .npy
                depth_file = depth_dir / f"depth_b{b}_s{s}.npy"
                np.save(depth_file, depth_map)

                # 保存置信度
                conf_file = depth_dir / f"depth_conf_b{b}_s{s}.npy"
                np.save(conf_file, confidence)

        logger.info(f"Saved depth maps to: {depth_dir}")

    # 保存点云
    if save_points:
        points_dir = output_dir / "points"
        points_dir.mkdir(exist_ok=True)

        for b in range(B):
            for s in range(S):
                pts3d = world_points[b, s]
                confidence = world_points_conf[b, s]

                # 保存点云
                points_file = points_dir / f"points_b{b}_s{s}.npy"
                np.save(points_file, pts3d)

                # 保存置信度
                conf_file = points_dir / f"points_conf_b{b}_s{s}.npy"
                np.save(conf_file, confidence)

        logger.info(f"Saved point clouds to: {points_dir}")

    # 保存汇总信息
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("VGGT Inference Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of images: {len(image_paths)}\n")
        f.write(f"Batch size: {B}\n")
        f.write(f"Sequence length: {S}\n\n")
        f.write("Output shapes:\n")
        f.write(f"  - Pose encoding: {pose_enc.shape}\n")
        f.write(f"  - Depth maps: {depth.shape}\n")
        f.write(f"  - World points: {world_points.shape}\n\n")
        f.write("Image paths:\n")
        for i, path in enumerate(image_paths):
            f.write(f"  {i+1}. {path}\n")

    logger.info(f"Saved summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with quantized VGGT model"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to quantized model checkpoint"
    )

    # Input arguments
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--image_paths",
        type=str,
        nargs="+",
        default=None,
        help="List of image paths"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save results"
    )
    parser.add_argument(
        "--save_depth",
        action="store_true",
        default=True,
        help="Save depth maps"
    )
    parser.add_argument(
        "--save_points",
        action="store_true",
        default=True,
        help="Save point clouds"
    )
    parser.add_argument(
        "--save_cameras",
        action="store_true",
        default=True,
        help="Save camera parameters"
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )

    args = parser.parse_args()

    # 验证输入
    if args.image_folder is None and args.image_paths is None:
        raise ValueError("Either --image_folder or --image_paths must be provided")

    # 获取图像路径
    if args.image_folder:
        image_folder = Path(args.image_folder)
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_paths = sorted([
            str(p) for p in image_folder.iterdir()
            if p.suffix in image_extensions
        ])
        logger.info(f"Found {len(image_paths)} images in {image_folder}")
    else:
        image_paths = args.image_paths

    # 限制图像数量
    if args.max_images:
        image_paths = image_paths[:args.max_images]
        logger.info(f"Limited to {len(image_paths)} images")

    if len(image_paths) == 0:
        raise ValueError("No images found")

    # 加载模型
    device = torch.device(args.device)
    model = load_quantized_model(args.model_path, device)

    # 运行推理
    predictions, inference_time = process_images(model, image_paths, device)

    # 保存结果
    save_results(
        predictions,
        args.output_dir,
        image_paths,
        save_depth=args.save_depth,
        save_points=args.save_points,
        save_cameras=args.save_cameras,
    )

    # 打印统计信息
    logger.info("\n" + "=" * 50)
    logger.info("Inference Statistics:")
    logger.info(f"  Total images: {len(image_paths)}")
    logger.info(f"  Total time: {inference_time:.3f}s")
    logger.info(f"  Time per image: {inference_time/len(image_paths):.3f}s")
    logger.info(f"  FPS: {len(image_paths)/inference_time:.2f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
