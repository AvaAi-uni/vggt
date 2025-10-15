#!/usr/bin/env python3
"""
ETH3D 数据集下载和解压脚本

支持在 RunPod 和本地 Windows 环境中运行
自动下载 ETH3D Multi-View Training DSLR Undistorted 数据集

数据集信息: https://www.eth3d.net/datasets
下载链接: https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z

使用方法：
    python scripts/download_eth3d.py --output_dir data/eth3d
    python scripts/download_eth3d.py --output_dir /workspace/data/eth3d  # RunPod
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import urllib.request
import subprocess
import shutil

# 尝试导入 tqdm（如果可用）
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset URL
ETH3D_URL = "https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z"
DATASET_NAME = "multi_view_training_dslr_undistorted.7z"


class DownloadProgressBar:
    """下载进度条（支持 tqdm 和纯文本模式）"""
    def __init__(self, total_size=None, desc="下载中"):
        self.total_size = total_size
        self.desc = desc
        self.downloaded = 0

        if HAS_TQDM and total_size:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
        else:
            self.pbar = None
            logger.info(f"{desc}: 总大小 {total_size/(1024*1024):.2f} MB" if total_size else desc)

    def update(self, block_num, block_size, total_size):
        if self.total_size is None and total_size > 0:
            self.total_size = total_size
            if HAS_TQDM:
                self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=self.desc)

        downloaded = block_num * block_size
        increment = downloaded - self.downloaded
        self.downloaded = downloaded

        if self.pbar:
            self.pbar.update(increment)
        elif block_num % 100 == 0 and self.total_size:  # 每100个块更新一次日志
            percent = min(100, downloaded * 100 / self.total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = self.total_size / (1024 * 1024)
            logger.info(f"已下载: {mb_downloaded:.2f}/{mb_total:.2f} MB ({percent:.1f}%)")

    def close(self):
        if self.pbar:
            self.pbar.close()


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """
    下载文件并显示进度

    Args:
        url: 下载 URL
        output_path: 保存路径
        chunk_size: 块大小
    """
    logger.info("=" * 80)
    logger.info(f"开始下载: {url}")
    logger.info(f"保存到: {output_path}")
    logger.info("=" * 80)

    try:
        progress = DownloadProgressBar(desc=DATASET_NAME)
        urllib.request.urlretrieve(url, output_path, progress.update)
        progress.close()
        logger.info("\n✓ 下载完成！")
        return True
    except Exception as e:
        logger.error(f"✗ 下载失败: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def extract_7z(archive_path: Path, output_dir: Path):
    """
    解压 7z 压缩包

    Args:
        archive_path: 7z 文件路径
        output_dir: 解压目标目录

    注意:
        需要安装 7-Zip 工具:
        - Windows: https://www.7-zip.org/
        - Linux: sudo apt-get install p7zip-full
        - macOS: brew install p7zip
        或者使用 Python 库: pip install py7zr
    """
    logger.info("=" * 80)
    logger.info(f"开始解压: {archive_path}")
    logger.info(f"目标目录: {output_dir}")
    logger.info("=" * 80)

    # 检查是否有 7z 命令
    seven_z_cmd = None
    for cmd in ['7z', '7za', '7zr']:
        if shutil.which(cmd):
            seven_z_cmd = cmd
            logger.info(f"找到 7z 工具: {cmd}")
            break

    if seven_z_cmd:
        try:
            # 使用 7z 命令解压
            cmd = [seven_z_cmd, 'x', str(archive_path), f'-o{output_dir}', '-y']
            subprocess.run(cmd, check=True)
            logger.info("\n✓ 解压完成！")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ 解压失败: {e}")
            return False
    else:
        # 尝试使用 py7zr 库
        logger.warning("未找到 7z 命令，尝试使用 py7zr 库...")
        try:
            import py7zr
            logger.info("使用 py7zr 库解压...")
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                z.extractall(path=output_dir)
            logger.info("\n✓ 解压完成！")
            return True
        except ImportError:
            logger.error("✗ 未安装 py7zr 库")
            logger.error("\n请安装 7-Zip 工具或 py7zr 库:")
            logger.error("  1. 安装 7-Zip 工具:")
            logger.error("     Windows: https://www.7-zip.org/")
            logger.error("     Linux: sudo apt-get install p7zip-full")
            logger.error("     RunPod: apt-get update && apt-get install -y p7zip-full")
            logger.error("  2. 或者安装 Python 库:")
            logger.error("     pip install py7zr")
            return False
        except Exception as e:
            logger.error(f"✗ 解压失败: {e}")
            return False


def organize_dataset(data_dir: Path):
    """
    组织和验证数据集

    Args:
        data_dir: 数据集目录
    """
    logger.info("=" * 80)
    logger.info("验证数据集...")
    logger.info("=" * 80)

    # ETH3D 结构: 每个场景有一个 'images' 文件夹
    scenes = [d for d in data_dir.iterdir() if d.is_dir() and d.name != "__MACOSX"]

    if len(scenes) == 0:
        logger.error("✗ 未找到任何场景目录")
        return False

    logger.info(f"找到 {len(scenes)} 个场景:")
    total_images = 0

    for scene in scenes:
        # 检查 images 文件夹
        images_folder = scene / "images"
        if images_folder.exists():
            num_images = len(list(images_folder.glob("*.JPG"))) + \
                        len(list(images_folder.glob("*.jpg"))) + \
                        len(list(images_folder.glob("*.png")))
            total_images += num_images
            logger.info(f"  ✓ {scene.name}: {num_images} 张图像")
        else:
            logger.warning(f"  ✗ {scene.name}: 未找到 images 文件夹")

    logger.info(f"\n数据集总计: {total_images} 张图像")
    logger.info("✓ 数据集验证完成！")
    return True


def create_dataset_info(output_dir: Path):
    """
    创建数据集说明文件

    Args:
        output_dir: 输出目录
    """
    readme_path = output_dir / "README.txt"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("ETH3D Multi-View Training Dataset (DSLR Undistorted)\n")
        f.write("=" * 80 + "\n\n")
        f.write("这是一个高质量的多视图立体数据集，用于 3D 重建任务。\n\n")
        f.write("数据集结构:\n")
        f.write("  - 每个子目录代表一个场景\n")
        f.write("  - 每个场景包含一个 'images' 文件夹，内有未失真的 DSLR 图像\n")
        f.write("  - 图像格式为 JPG\n\n")
        f.write("在 VGGT 中使用:\n")
        f.write("  1. 训练: 使用多个场景的图像进行训练\n")
        f.write("  2. 验证: 使用其他场景的图像进行验证\n")
        f.write("  3. 测试: 评估模型性能\n\n")
        f.write("数据集 URL: https://www.eth3d.net/\n")
        f.write("引用: 如果使用本数据集，请引用 ETH3D 论文。\n")

    logger.info(f"数据集说明已保存: {readme_path}")


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

    # 步骤1: 下载
    if args.skip_download and archive_path.exists():
        logger.info(f"压缩包已存在: {archive_path}")
        logger.info("跳过下载 (--skip_download)")
    else:
        success = download_file(ETH3D_URL, archive_path)
        if not success:
            logger.error("下载失败，退出")
            sys.exit(1)

    # 步骤2: 解压
    success = extract_7z(archive_path, output_dir)
    if not success:
        logger.error("解压失败，退出")
        sys.exit(1)

    # 步骤3: 验证数据集
    success = organize_dataset(output_dir)
    if not success:
        logger.error("数据集验证失败")
        sys.exit(1)

    # 步骤4: 创建说明文件
    create_dataset_info(output_dir)

    # 步骤5: 删除压缩包（可选）
    if not args.keep_archive:
        logger.info("\n删除压缩包...")
        archive_path.unlink()
        logger.info("✓ 压缩包已删除")

    # 完成
    logger.info("\n" + "=" * 80)
    logger.info("✓ ETH3D 数据集准备完成！")
    logger.info("=" * 80)
    logger.info(f"\n数据集路径: {output_dir.absolute()}")
    logger.info("\n下一步:")
    logger.info("  1. 查看配置文件: training/config/eth3d_*.yaml")
    logger.info("  2. 开始训练: python training/launch.py --config eth3d_fp32_baseline")
    logger.info("  3. 或使用一键脚本: python scripts/run_quantization_experiments.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
