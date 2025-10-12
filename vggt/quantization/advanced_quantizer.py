#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
高级量化工具 - 支持多种量化方案对比

支持的量化方案：
1. INT8 对称量化 (Symmetric)
2. INT8 非对称量化 (Asymmetric)
3. INT4 分组量化 (Group-wise)
4. 混合精度量化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import copy

logger = logging.getLogger(__name__)


@dataclass
class AdvancedQuantConfig:
    """高级量化配置"""
    quant_type: str = "int8_symmetric"  # int8_symmetric, int8_asymmetric, int4_group
    group_size: int = 128  # INT4 分组量化的组大小
    bits: int = 8  # 量化位数
    symmetric: bool = True  # 对称/非对称
    per_channel: bool = True  # 逐通道量化
    device: str = "cuda"


class SymmetricQuantizer:
    """INT8 对称量化器"""

    def __init__(self, bits=8):
        self.bits = bits
        self.qmin = -2 ** (bits - 1)
        self.qmax = 2 ** (bits - 1) - 1

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对称量化：Q = round(x / scale)
        scale = max(|x|) / (2^(bits-1) - 1)
        """
        # 计算 scale
        max_val = torch.max(torch.abs(tensor))
        scale = max_val / self.qmax

        # 量化
        q_tensor = torch.clamp(torch.round(tensor / scale), self.qmin, self.qmax)

        return q_tensor, scale

    def dequantize_tensor(self, q_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """反量化：x = Q * scale"""
        return q_tensor * scale


class AsymmetricQuantizer:
    """INT8 非对称量化器"""

    def __init__(self, bits=8):
        self.bits = bits
        self.qmin = 0
        self.qmax = 2 ** bits - 1

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        非对称量化：Q = round((x - zero_point) / scale)
        scale = (max - min) / (2^bits - 1)
        zero_point = round(-min / scale)
        """
        # 计算 scale 和 zero_point
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)

        scale = (max_val - min_val) / self.qmax
        zero_point = torch.round(-min_val / scale)

        # 量化
        q_tensor = torch.clamp(
            torch.round(tensor / scale + zero_point),
            self.qmin,
            self.qmax
        )

        return q_tensor, scale, zero_point

    def dequantize_tensor(
        self,
        q_tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """反量化：x = (Q - zero_point) * scale"""
        return (q_tensor - zero_point) * scale


class GroupWiseQuantizer:
    """INT4 分组量化器"""

    def __init__(self, bits=4, group_size=128):
        self.bits = bits
        self.group_size = group_size
        self.qmin = -2 ** (bits - 1)
        self.qmax = 2 ** (bits - 1) - 1

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分组量化：将张量分成多个组，每组独立量化
        """
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()

        # 填充到 group_size 的倍数
        pad_len = (self.group_size - len(tensor_flat) % self.group_size) % self.group_size
        if pad_len > 0:
            tensor_flat = torch.cat([tensor_flat, torch.zeros(pad_len, device=tensor.device)])

        # 重塑为组
        tensor_groups = tensor_flat.reshape(-1, self.group_size)
        num_groups = tensor_groups.shape[0]

        # 为每组计算 scale
        scales = torch.zeros(num_groups, device=tensor.device)
        q_groups = torch.zeros_like(tensor_groups)

        for i in range(num_groups):
            group = tensor_groups[i]
            max_val = torch.max(torch.abs(group))
            scale = max_val / self.qmax
            scales[i] = scale

            # 量化该组
            q_groups[i] = torch.clamp(torch.round(group / scale), self.qmin, self.qmax)

        # 恢复原始形状
        q_tensor = q_groups.flatten()[:original_shape.numel()].reshape(original_shape)

        return q_tensor, scales

    def dequantize_tensor(
        self,
        q_tensor: torch.Tensor,
        scales: torch.Tensor
    ) -> torch.Tensor:
        """反量化：每组使用对应的 scale"""
        original_shape = q_tensor.shape
        q_flat = q_tensor.flatten()

        # 填充
        pad_len = (self.group_size - len(q_flat) % self.group_size) % self.group_size
        if pad_len > 0:
            q_flat = torch.cat([q_flat, torch.zeros(pad_len, device=q_tensor.device)])

        # 重塑为组
        q_groups = q_flat.reshape(-1, self.group_size)
        num_groups = q_groups.shape[0]

        # 反量化每组
        deq_groups = torch.zeros_like(q_groups)
        for i in range(num_groups):
            if i < len(scales):
                deq_groups[i] = q_groups[i] * scales[i]
            else:
                deq_groups[i] = q_groups[i] * scales[-1]

        # 恢复原始形状
        deq_tensor = deq_groups.flatten()[:original_shape.numel()].reshape(original_shape)

        return deq_tensor


class QuantizedLinear(nn.Module):
    """量化的线性层"""

    def __init__(self, original_layer: nn.Linear, config: AdvancedQuantConfig):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.config = config

        # 选择量化器
        if config.quant_type == "int8_symmetric":
            self.quantizer = SymmetricQuantizer(bits=config.bits)
        elif config.quant_type == "int8_asymmetric":
            self.quantizer = AsymmetricQuantizer(bits=config.bits)
        elif config.quant_type == "int4_group":
            self.quantizer = GroupWiseQuantizer(bits=config.bits, group_size=config.group_size)
        else:
            raise ValueError(f"Unknown quant_type: {config.quant_type}")

        # 量化权重
        self.quantize_weights(original_layer.weight.data)

        # 处理偏置
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None

    def quantize_weights(self, weight: torch.Tensor):
        """量化权重"""
        if self.config.quant_type == "int8_symmetric":
            q_weight, scale = self.quantizer.quantize_tensor(weight)
            self.register_buffer('q_weight', q_weight.to(torch.int8))
            self.register_buffer('weight_scale', scale)

        elif self.config.quant_type == "int8_asymmetric":
            q_weight, scale, zero_point = self.quantizer.quantize_tensor(weight)
            self.register_buffer('q_weight', q_weight.to(torch.uint8))
            self.register_buffer('weight_scale', scale)
            self.register_buffer('weight_zero_point', zero_point)

        elif self.config.quant_type == "int4_group":
            q_weight, scales = self.quantizer.quantize_tensor(weight)
            self.register_buffer('q_weight', q_weight.to(torch.int8))  # INT4 存储为 INT8
            self.register_buffer('weight_scales', scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 反量化权重
        if self.config.quant_type == "int8_symmetric":
            weight = self.quantizer.dequantize_tensor(
                self.q_weight.float(),
                self.weight_scale
            )
        elif self.config.quant_type == "int8_asymmetric":
            weight = self.quantizer.dequantize_tensor(
                self.q_weight.float(),
                self.weight_scale,
                self.weight_zero_point
            )
        elif self.config.quant_type == "int4_group":
            weight = self.quantizer.dequantize_tensor(
                self.q_weight.float(),
                self.weight_scales
            )

        # 线性变换
        return nn.functional.linear(x, weight, self.bias)


def quantize_model_advanced(
    model: nn.Module,
    config: AdvancedQuantConfig
) -> nn.Module:
    """
    高级量化：替换模型中的线性层

    Args:
        model: 原始模型
        config: 量化配置

    Returns:
        量化后的模型
    """
    logger.info(f"Starting advanced quantization: {config.quant_type}")

    model_quantized = copy.deepcopy(model)
    model_quantized.eval()

    # 统计和替换线性层
    total_layers = 0
    quantized_layers = 0

    def replace_linear(module):
        nonlocal total_layers, quantized_layers

        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                total_layers += 1
                # 替换为量化层
                quantized_layer = QuantizedLinear(child, config)
                setattr(module, name, quantized_layer)
                quantized_layers += 1
            else:
                replace_linear(child)

    replace_linear(model_quantized)

    logger.info(f"Quantized {quantized_layers}/{total_layers} linear layers")
    logger.info(f"Quantization type: {config.quant_type}")
    logger.info(f"Bits: {config.bits}")

    return model_quantized


def compare_quantization_methods(
    original_model: nn.Module,
    test_input: torch.Tensor,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    对比不同量化方法的效果

    Returns:
        包含各种指标的字典
    """
    results = {}

    # 原始模型推理
    logger.info("Running original model...")
    original_model = original_model.to(device)
    original_model.eval()

    with torch.no_grad():
        original_output = original_model(test_input.to(device))

    # 测试不同量化方法
    quant_configs = [
        ("INT8 Symmetric", AdvancedQuantConfig(quant_type="int8_symmetric", bits=8, device=device)),
        ("INT8 Asymmetric", AdvancedQuantConfig(quant_type="int8_asymmetric", bits=8, device=device)),
        ("INT4 Group-128", AdvancedQuantConfig(quant_type="int4_group", bits=4, group_size=128, device=device)),
        ("INT4 Group-64", AdvancedQuantConfig(quant_type="int4_group", bits=4, group_size=64, device=device)),
    ]

    for name, config in quant_configs:
        logger.info(f"\nTesting {name}...")

        try:
            # 量化模型
            quantized_model = quantize_model_advanced(original_model, config)
            quantized_model = quantized_model.to(device)
            quantized_model.eval()

            # 推理
            import time
            start = time.time()
            with torch.no_grad():
                quantized_output = quantized_model(test_input.to(device))
            inference_time = time.time() - start

            # 计算精度指标
            if isinstance(original_output, dict) and isinstance(quantized_output, dict):
                metrics = {}
                for key in original_output.keys():
                    if key == "images":
                        continue

                    orig = original_output[key].cpu().float()
                    quant = quantized_output[key].cpu().float()

                    mse = torch.mean((orig - quant) ** 2).item()
                    mae = torch.mean(torch.abs(orig - quant)).item()
                    max_diff = torch.max(torch.abs(orig - quant)).item()

                    # 计算 PSNR
                    if mse > 0:
                        psnr = 10 * np.log10(1.0 / mse)
                    else:
                        psnr = float('inf')

                    metrics[key] = {
                        "mse": mse,
                        "mae": mae,
                        "max_diff": max_diff,
                        "psnr": psnr
                    }
            else:
                orig = original_output.cpu().float()
                quant = quantized_output.cpu().float()

                mse = torch.mean((orig - quant) ** 2).item()
                mae = torch.mean(torch.abs(orig - quant)).item()
                max_diff = torch.max(torch.abs(orig - quant)).item()

                if mse > 0:
                    psnr = 10 * np.log10(1.0 / mse)
                else:
                    psnr = float('inf')

                metrics = {
                    "mse": mse,
                    "mae": mae,
                    "max_diff": max_diff,
                    "psnr": psnr
                }

            # 计算模型大小
            model_size = sum(
                p.numel() * p.element_size()
                for p in quantized_model.parameters()
            ) / 1024 / 1024

            results[name] = {
                "metrics": metrics,
                "model_size_mb": model_size,
                "inference_time": inference_time,
                "config": config
            }

            logger.info(f"  Model size: {model_size:.2f} MB")
            logger.info(f"  Inference time: {inference_time:.4f}s")

        except Exception as e:
            logger.error(f"Error with {name}: {e}")
            results[name] = {"error": str(e)}

    return results
