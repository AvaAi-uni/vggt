# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ETH3D Dataset Loader for VGGT Training

ETH3D是一个高质量的多视图立体数据集，包含了DSLR相机拍摄的未失真图像。
本数据集加载器设计用于支持VGGT模型的训练和验证。

数据集特点：
- 高分辨率DSLR图像
- 精确的相机标定参数
- 多个室内外场景
- 适合Structure from Motion和多视图重建任务

参考文献:
Schöps, T., Schönberger, J. L., Galliani, S., Sattler, T., Schindler, K.,
Pollefeys, M., & Geiger, A. (2017). A multi-view stereo benchmark with
high-resolution images and multi-camera videos. CVPR 2017.
"""

import os
import os.path as osp
import logging
import json
import glob
from typing import Dict, List, Optional
import numpy as np
import cv2

from data.dataset_util import *
from data.base_dataset import BaseDataset


class ETH3DDataset(BaseDataset):
    """
    ETH3D数据集加载器

    支持多种训练模式：
    1. 单场景训练：专注于一个特定场景
    2. 多场景训练：从所有场景中采样
    3. 混合训练：结合多个数据集

    数据格式：
    ETH3D_DIR/
    ├── scene1/
    │   ├── images/
    │   │   ├── DSC_0001.JPG
    │   │   ├── DSC_0002.JPG
    │   │   └── ...
    │   ├── cameras.txt (optional, COLMAP格式)
    │   └── dslr_calibration_jpg.txt (包含相机内参)
    └── scene2/
        └── ...
    """

    def __init__(
        self,
        common_conf,
        ETH3D_DIR: str = None,
        split: str = "train",
        scenes: Optional[List[str]] = None,
        min_num_images: int = 5,
        max_num_images: int = 50,
        len_train: int = 10000,
        len_test: int = 1000,
        use_cache: bool = True,
    ):
        """
        初始化ETH3D数据集

        Args:
            common_conf: 通用配置对象，包含图像大小、增强等设置
            ETH3D_DIR: ETH3D数据集根目录
            split: 数据集分割 ("train" 或 "test")
            scenes: 要使用的场景列表。如果为None，使用所有场景
            min_num_images: 每个序列的最小图像数量
            max_num_images: 每个序列的最大图像数量
            len_train: 训练数据集长度（虚拟长度，用于epoch计算）
            len_test: 测试数据集长度
            use_cache: 是否使用缓存加速数据加载
        """
        super().__init__(common_conf=common_conf)

        if ETH3D_DIR is None:
            raise ValueError("ETH3D_DIR must be specified")

        if not osp.exists(ETH3D_DIR):
            raise ValueError(f"ETH3D_DIR does not exist: {ETH3D_DIR}")

        self.ETH3D_DIR = ETH3D_DIR
        self.split = split
        self.min_num_images = min_num_images
        self.max_num_images = max_num_images
        self.use_cache = use_cache

        # 从common_conf获取训练配置
        self.debug = getattr(common_conf, 'debug', False)
        self.training = getattr(common_conf, 'training', split == "train")
        self.get_nearby = getattr(common_conf, 'get_nearby', True)
        self.load_depth = getattr(common_conf, 'load_depth', False)
        self.inside_random = getattr(common_conf, 'inside_random', False)
        self.allow_duplicate_img = getattr(common_conf, 'allow_duplicate_img', True)

        # 设置数据集长度
        if split == "train":
            self.len_train = len_train
        else:
            self.len_train = len_test

        # 初始化数据存储
        self.data_store = {}
        self.sequence_list = []

        # 加载场景数据
        self._load_scenes(scenes)

        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ETH3D Dataset initialized")
        logging.info(f"  - Root directory: {ETH3D_DIR}")
        logging.info(f"  - Number of scenes: {len(self.data_store)}")
        logging.info(f"  - Total sequences: {self.sequence_list_len}")
        logging.info(f"  - Virtual dataset length: {len(self)}")

    def _load_scenes(self, scenes: Optional[List[str]] = None):
        """
        加载所有场景的元数据

        Args:
            scenes: 要加载的场景列表。如果为None，加载所有场景
        """
        # 获取所有场景目录
        if scenes is None:
            scene_dirs = [
                d for d in os.listdir(self.ETH3D_DIR)
                if osp.isdir(osp.join(self.ETH3D_DIR, d)) and d not in ['training', 'test', '__pycache__']
            ]
        else:
            scene_dirs = scenes

        if self.debug:
            scene_dirs = scene_dirs[:1]  # 调试模式只使用第一个场景

        logging.info(f"Loading ETH3D scenes: {scene_dirs}")

        for scene_name in scene_dirs:
            scene_path = osp.join(self.ETH3D_DIR, scene_name)

            if not osp.exists(scene_path):
                logging.warning(f"Scene directory not found: {scene_path}")
                continue

            # 加载场景图像列表
            # ETH3D 可能使用 'images' 或 'dslr_undistorted_images' 目录
            images_dir = osp.join(scene_path, "images")
            if not osp.exists(images_dir):
                images_dir = osp.join(scene_path, "dslr_undistorted_images")

            if not osp.exists(images_dir):
                logging.warning(f"Images directory not found for scene {scene_name}")
                logging.warning(f"  Checked: {scene_path}/images and {scene_path}/dslr_undistorted_images")
                continue

            # 获取所有图像文件
            image_files = sorted(glob.glob(osp.join(images_dir, "*.JPG")))
            image_files += sorted(glob.glob(osp.join(images_dir, "*.jpg")))
            image_files += sorted(glob.glob(osp.join(images_dir, "*.png")))

            if len(image_files) < self.min_num_images:
                logging.warning(
                    f"Scene {scene_name} has only {len(image_files)} images, "
                    f"less than minimum {self.min_num_images}"
                )
                continue

            # 加载相机标定参数
            calibration = self._load_calibration(scene_path)

            # 构建序列元数据
            sequence_metadata = []
            for img_path in image_files:
                img_name = osp.basename(img_path)

                # 获取该图像的相机参数
                intrinsics = calibration.get('intrinsics', self._get_default_intrinsics())
                extrinsics = calibration.get('extrinsics', {}).get(img_name, None)

                metadata = {
                    'filepath': osp.relpath(img_path, self.ETH3D_DIR),
                    'scene_name': scene_name,
                    'image_name': img_name,
                    'intrinsics': intrinsics,
                    'extrinsics': extrinsics,
                }
                sequence_metadata.append(metadata)

            # 存储序列数据
            self.data_store[scene_name] = sequence_metadata
            self.sequence_list.append(scene_name)

            logging.info(f"  Loaded scene '{scene_name}': {len(sequence_metadata)} images")

    def _load_calibration(self, scene_path: str) -> Dict:
        """
        加载场景的相机标定参数

        Args:
            scene_path: 场景目录路径

        Returns:
            包含内参和外参的字典
        """
        calibration = {}

        # 尝试加载dslr_calibration_jpg.txt文件
        calib_file = osp.join(scene_path, "dslr_calibration_jpg.txt")
        if osp.exists(calib_file):
            try:
                intrinsics = self._parse_eth3d_calibration(calib_file)
                calibration['intrinsics'] = intrinsics
                logging.debug(f"Loaded calibration from {calib_file}")
            except Exception as e:
                logging.warning(f"Failed to parse calibration file {calib_file}: {e}")

        # 可以扩展支持COLMAP格式的cameras.txt
        # 这里暂时使用简单的内参，外参从SfM中获取

        return calibration

    def _parse_eth3d_calibration(self, calib_file: str) -> np.ndarray:
        """
        解析ETH3D标定文件

        ETH3D的dslr_calibration_jpg.txt格式：
        fx fy cx cy

        Args:
            calib_file: 标定文件路径

        Returns:
            3x3内参矩阵
        """
        with open(calib_file, 'r') as f:
            lines = f.readlines()

        # 解析第一行：fx fy cx cy
        params = lines[0].strip().split()
        fx, fy, cx, cy = map(float, params[:4])

        # 构建内参矩阵
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return intrinsics

    def _get_default_intrinsics(self) -> np.ndarray:
        """
        获取默认内参矩阵（如果没有标定文件）

        Returns:
            3x3内参矩阵
        """
        # 使用常见DSLR相机的典型参数
        # 这里假设图像分辨率约为6000x4000像素
        fx = fy = 4000.0  # 焦距（像素）
        cx = 3000.0       # 主点x坐标
        cy = 2000.0       # 主点y坐标

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return intrinsics

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        获取一个batch的数据

        Args:
            seq_index: 序列索引
            img_per_seq: 每个序列的图像数量
            seq_name: 序列名称
            ids: 指定的图像ID列表
            aspect_ratio: 目标纵横比

        Returns:
            包含图像、深度、相机参数等的字典
        """
        if self.inside_random:
            seq_index = np.random.randint(0, self.sequence_list_len)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]

        # 采样图像ID
        if ids is None:
            num_images = min(len(metadata), self.max_num_images)
            if img_per_seq is not None:
                num_images = min(num_images, img_per_seq)

            # 随机采样或使用nearby采样
            if self.get_nearby:
                # 从序列中选择连续或接近的帧
                start_idx = np.random.randint(0, max(1, len(metadata) - num_images + 1))
                ids = list(range(start_idx, start_idx + num_images))
            else:
                # 完全随机采样
                ids = np.random.choice(
                    len(metadata),
                    num_images,
                    replace=self.allow_duplicate_img
                )

        annos = [metadata[i] for i in ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        # 初始化数据列表
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for anno in annos:
            filepath = anno['filepath']
            image_path = osp.join(self.ETH3D_DIR, filepath)

            # 读取图像
            image = read_image_cv2(image_path)

            # ETH3D通常不提供深度图，使用零深度图
            # 在实际训练中，可以使用预计算的深度或在线估计
            depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            original_size = np.array(image.shape[:2])

            # 获取相机参数
            intri_opencv = anno['intrinsics']

            # 如果没有外参，使用单位矩阵
            if anno['extrinsics'] is not None:
                extri_opencv = anno['extrinsics']
            else:
                # 使用单位外参（相机在原点，面向+Z）
                extri_opencv = np.eye(4, dtype=np.float32)

            # 处理图像和相机参数
            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=filepath,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        # 构建batch
        batch = {
            "seq_name": f"eth3d_{seq_name}",
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }

        return batch
