#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to download and extract ETH3D dataset.

The ETH3D dataset is a multi-view stereo benchmark dataset with high-quality
ground truth 3D reconstructions. This script downloads the DSLR undistorted
training data which is suitable for calibrating and testing VGGT.

Dataset info: https://www.eth3d.net/datasets
Direct link: https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z

Usage:
    python scripts/download_eth3d.py --output_dir data/eth3d
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import urllib.request
import subprocess
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset URL
ETH3D_URL = "https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z"
DATASET_NAME = "multi_view_training_dslr_undistorted.7z"


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """
    Download a file with progress indication.

    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        chunk_size: Size of chunks to download
    """
    logger.info(f"Downloading from: {url}")
    logger.info(f"Saving to: {output_path}")

    # Create progress callback
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)

        if block_num % 100 == 0:  # Update every 100 blocks
            logger.info(f"Downloaded: {mb_downloaded:.2f}/{mb_total:.2f} MB ({percent:.1f}%)")

    try:
        urllib.request.urlretrieve(url, output_path, progress_callback)
        logger.info("Download completed!")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def extract_7z(archive_path: Path, output_dir: Path):
    """
    Extract 7z archive.

    Args:
        archive_path: Path to .7z archive
        output_dir: Directory to extract to

    Note:
        This requires 7z command-line tool to be installed:
        - Windows: Install 7-Zip from https://www.7-zip.org/
        - Linux: sudo apt-get install p7zip-full
        - macOS: brew install p7zip
    """
    logger.info(f"Extracting {archive_path}...")

    # Check if 7z is available
    seven_z_cmd = None
    for cmd in ['7z', '7za', '7zr']:
        if shutil.which(cmd):
            seven_z_cmd = cmd
            break

    if seven_z_cmd is None:
        logger.error("7z command not found!")
        logger.error("Please install 7-Zip:")
        logger.error("  Windows: https://www.7-zip.org/")
        logger.error("  Linux: sudo apt-get install p7zip-full")
        logger.error("  macOS: brew install p7zip")
        logger.error("\nAlternatively, manually extract the file and place contents in the output directory.")
        raise RuntimeError("7z command not found")

    try:
        # Extract archive
        cmd = [seven_z_cmd, 'x', str(archive_path), f'-o{output_dir}', '-y']
        subprocess.run(cmd, check=True)
        logger.info("Extraction completed!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Extraction failed: {e}")
        raise


def organize_dataset(data_dir: Path):
    """
    Organize the extracted dataset for easy use.

    Args:
        data_dir: Directory containing extracted data
    """
    logger.info("Organizing dataset...")

    # ETH3D structure: each scene has an 'images' folder with undistorted images
    scenes = [d for d in data_dir.iterdir() if d.is_dir()]

    logger.info(f"Found {len(scenes)} scenes:")
    for scene in scenes:
        logger.info(f"  - {scene.name}")

        # Check if images folder exists
        images_folder = scene / "images"
        if images_folder.exists():
            num_images = len(list(images_folder.glob("*.JPG"))) + len(list(images_folder.glob("*.jpg")))
            logger.info(f"    Images: {num_images}")
        else:
            logger.warning(f"    No images folder found!")

    logger.info("Dataset organization complete!")


def create_dataset_info(output_dir: Path):
    """
    Create a README with dataset information.

    Args:
        output_dir: Output directory
    """
    readme_path = output_dir / "README.txt"

    with open(readme_path, "w") as f:
        f.write("ETH3D Multi-View Training Dataset (DSLR Undistorted)\n")
        f.write("=" * 60 + "\n\n")
        f.write("This dataset contains high-quality multi-view images for 3D reconstruction.\n\n")
        f.write("Dataset structure:\n")
        f.write("  - Each subdirectory represents a scene\n")
        f.write("  - Each scene contains an 'images' folder with undistorted DSLR images\n")
        f.write("  - Images are in JPG format\n\n")
        f.write("Usage with VGGT:\n")
        f.write("  1. For calibration: Use images from multiple scenes\n")
        f.write("  2. For testing: Use any scene's images folder\n\n")
        f.write("Dataset URL: https://www.eth3d.net/\n")
        f.write("Citation: If you use this dataset, please cite the ETH3D paper.\n")

    logger.info(f"Dataset info saved to: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract ETH3D dataset for VGGT"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/eth3d",
        help="Directory to save dataset"
    )
    parser.add_argument(
        "--keep_archive",
        action="store_true",
        help="Keep the downloaded .7z archive after extraction"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download if archive already exists"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / DATASET_NAME

    # Download dataset
    if args.skip_download and archive_path.exists():
        logger.info(f"Archive already exists: {archive_path}")
        logger.info("Skipping download (--skip_download flag is set)")
    else:
        download_file(ETH3D_URL, archive_path)

    # Extract dataset
    extract_7z(archive_path, output_dir)

    # Organize dataset
    organize_dataset(output_dir)

    # Create dataset info
    create_dataset_info(output_dir)

    # Remove archive if requested
    if not args.keep_archive:
        logger.info("Removing archive...")
        archive_path.unlink()
        logger.info("Archive removed")

    logger.info(f"\nDataset successfully downloaded and extracted to: {output_dir}")
    logger.info("\nYou can now use this data for:")
    logger.info("  1. Calibration during static quantization")
    logger.info("  2. Testing VGGT reconstructions")
    logger.info("  3. Fine-tuning the model")


if __name__ == "__main__":
    main()
