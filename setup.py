#!/usr/bin/env python3
"""
VGGT 包安装脚本
"""

from setuptools import setup, find_packages

setup(
    name="vggt",
    version="1.0.0",
    description="VGGT: Video Geometry and Gaussians Transformer",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "torch>=2.0.0",
        "opencv-python>=4.5.0",
        "pillow>=9.0.0",
        "scipy>=1.7.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.10.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "fvcore>=0.1.5",
        "iopath>=0.1.9",
        "einops>=0.6.0",
        "timm>=0.9.0",
        "wcmatch>=8.4.0",
        "pyyaml>=6.0",
    ],
)
