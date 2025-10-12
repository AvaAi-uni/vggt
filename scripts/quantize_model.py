#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to quantize VGGT model from FP32 to INT8.

This script demonstrates how to:
1. Load a pretrained VGGT model
2. Prepare calibration data (if using static quantization)
3. Quantize the model to INT8
4. Compare accuracy and memory usage
5. Save the quantized model

Usage:
    # Dynamic quantization (no calibration needed)
    python scripts/quantize_model.py --model_name facebook/VGGT-1B \
        --quantization_type dynamic \
        --output_path models/vggt_int8.pt

    # Static quantization (requires calibration data)
    python scripts/quantize_model.py --model_name facebook/VGGT-1B \
        --quantization_type static \
        --calibration_data path/to/images \
        --calibration_samples 100 \
        --output_path models/vggt_int8_static.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.quantization import (
    quantize_model,
    QuantizationConfig,
    estimate_model_size,
    compare_model_outputs,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageFolderDataset(Dataset):
    """Simple dataset for loading images from a folder."""

    def __init__(self, image_folder, transform=None, max_images=None):
        self.image_folder = Path(image_folder)
        self.transform = transform

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.image_paths = [
            p for p in self.image_folder.rglob('*')
            if p.suffix.lower() in image_extensions
        ]

        if max_images:
            self.image_paths = self.image_paths[:max_images]

        logger.info(f"Found {len(self.image_paths)} images in {image_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def create_calibration_dataloader(
    image_folder: str,
    batch_size: int = 1,
    num_workers: int = 4,
    img_size: int = 518,
    max_images: int = None,
) -> DataLoader:
    """
    Create a DataLoader for calibration.

    Args:
        image_folder: Path to folder containing calibration images
        batch_size: Batch size for calibration
        num_workers: Number of worker processes
        img_size: Target image size
        max_images: Maximum number of images to use

    Returns:
        DataLoader for calibration
    """
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolderDataset(image_folder, transform=transform, max_images=max_images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Quantize VGGT model to INT8")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/VGGT-1B",
        help="Model name or path to load"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/vggt_int8.pt",
        help="Path to save quantized model"
    )

    # Quantization arguments
    parser.add_argument(
        "--quantization_type",
        type=str,
        default="dynamic",
        choices=["dynamic", "static", "qat"],
        help="Type of quantization"
    )
    parser.add_argument(
        "--observer_type",
        type=str,
        default="minmax",
        choices=["minmax", "histogram", "per_channel"],
        help="Type of observer for calibration"
    )
    parser.add_argument(
        "--quantize_attention",
        action="store_true",
        default=True,
        help="Quantize attention layers"
    )
    parser.add_argument(
        "--quantize_heads",
        action="store_true",
        default=True,
        help="Quantize prediction heads"
    )

    # Calibration arguments
    parser.add_argument(
        "--calibration_data",
        type=str,
        default=None,
        help="Path to calibration data (required for static quantization)"
    )
    parser.add_argument(
        "--calibration_samples",
        type=int,
        default=100,
        help="Number of samples for calibration"
    )
    parser.add_argument(
        "--calibration_batch_size",
        type=int,
        default=1,
        help="Batch size for calibration"
    )

    # Evaluation arguments
    parser.add_argument(
        "--compare_outputs",
        action="store_true",
        help="Compare outputs between original and quantized models"
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default=None,
        help="Path to test image for comparison"
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for quantization"
    )

    args = parser.parse_args()

    # Load original model
    logger.info(f"Loading model: {args.model_name}")
    device = torch.device(args.device)
    model = VGGT.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    # Estimate original model size
    logger.info("Original model size:")
    original_size = estimate_model_size(model)
    for key, value in original_size.items():
        logger.info(f"  {key}: {value:.2f} MB")

    # Create quantization config
    config = QuantizationConfig(
        quantization_type=args.quantization_type,
        calibration_samples=args.calibration_samples,
        observer_type=args.observer_type,
        quantize_attention=args.quantize_attention,
        quantize_heads=args.quantize_heads,
    )

    # Prepare calibration data if needed
    calibration_loader = None
    if args.quantization_type == "static":
        if args.calibration_data is None:
            raise ValueError("--calibration_data is required for static quantization")

        logger.info("Creating calibration dataloader...")
        calibration_loader = create_calibration_dataloader(
            args.calibration_data,
            batch_size=args.calibration_batch_size,
            max_images=args.calibration_samples,
        )

    # Quantize model
    logger.info("Starting quantization...")
    quantized_model = quantize_model(model, config, calibration_loader)

    # Estimate quantized model size
    logger.info("Quantized model size:")
    quantized_size = estimate_model_size(quantized_model)
    for key, value in quantized_size.items():
        logger.info(f"  {key}: {value:.2f} MB")

    # Calculate compression ratio
    compression_ratio = original_size["total_mb"] / quantized_size["total_mb"]
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    logger.info(f"Memory saved: {original_size['total_mb'] - quantized_size['total_mb']:.2f} MB")

    # Compare outputs if requested
    if args.compare_outputs:
        if args.test_image is None:
            logger.warning("--test_image not provided, skipping output comparison")
        else:
            logger.info("Comparing outputs...")
            test_images = load_and_preprocess_images([args.test_image]).to(device)

            metrics = compare_model_outputs(
                model,
                quantized_model,
                test_images,
                device=args.device,
            )

            logger.info("Output comparison metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.6f}")

    # Save quantized model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving quantized model to: {output_path}")
    torch.save(quantized_model.state_dict(), output_path)

    # Also save quantization config
    config_path = output_path.parent / f"{output_path.stem}_config.txt"
    with open(config_path, "w") as f:
        f.write(f"Quantization Configuration:\n")
        f.write(f"  Type: {config.quantization_type}\n")
        f.write(f"  Observer: {config.observer_type}\n")
        f.write(f"  Calibration samples: {config.calibration_samples}\n")
        f.write(f"  Quantize attention: {config.quantize_attention}\n")
        f.write(f"  Quantize heads: {config.quantize_heads}\n")
        f.write(f"\nModel Size:\n")
        f.write(f"  Original: {original_size['total_mb']:.2f} MB\n")
        f.write(f"  Quantized: {quantized_size['total_mb']:.2f} MB\n")
        f.write(f"  Compression ratio: {compression_ratio:.2f}x\n")

    logger.info(f"Quantization config saved to: {config_path}")
    logger.info("Quantization completed successfully!")


if __name__ == "__main__":
    main()
