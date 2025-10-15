#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
完整的量化实验框架 - 支持从FP32到INT8/INT4的多精度量化

这个模块提供了一个全面的量化实验框架，包括：
1. Baseline: FP32原始模型
2. 多种量化方案: Per-Tensor, Per-Channel, Group-wise
3. 多种评估指标: MAE, MSE, RMSE, PSNR, CE, Cosine Similarity
4. 详细的实验参数配置

支持的量化方案:
- INT8 Per-Tensor Symmetric/Asymmetric
- INT8 Per-Channel Symmetric/Asymmetric
- INT4 Group-wise (多种组大小)
- Mixed Precision (混合精度)

作者: Quantization Research Team
日期: 2025-10-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import logging
import copy
import time

logger = logging.getLogger(__name__)


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class QuantizationConfig:
    """
    完整的量化配置

    Args:
        name: 量化方案名称
        quant_type: 量化类型 (int8_per_tensor_sym, int8_per_channel_sym,
                            int8_per_tensor_asym, int8_per_channel_asym,
                            int4_group, mixed_precision)
        bits: 量化位数 (4, 8)
        symmetric: 是否对称量化
        per_channel: 是否逐通道量化
        group_size: 分组量化的组大小 (仅用于int4_group)
        calibration_samples: 校准样本数量
        quantize_activations: 是否量化激活
        skip_first_last: 是否跳过第一层和最后一层（保持FP32以提高精度）
        device: 设备 (cuda/cpu)
    """
    name: str = "INT8_Per_Tensor_Symmetric"
    quant_type: str = "int8_per_tensor_sym"
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    group_size: int = 128
    calibration_samples: int = 100
    quantize_activations: bool = False
    skip_first_last: bool = True
    device: str = "cuda"

    # 混合精度特定配置
    sensitive_layers: List[str] = field(default_factory=list)
    sensitive_bits: int = 8
    normal_bits: int = 4


# ============================================================================
# 量化器实现
# ============================================================================

class PerTensorSymmetricQuantizer:
    """Per-Tensor对称量化器

    量化公式: Q = clamp(round(x / scale), qmin, qmax)
    scale = max(|x|) / (2^(bits-1) - 1)

    优点: 最简单，计算开销最小
    缺点: 对不同通道的动态范围敏感
    """

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin = -2 ** (bits - 1)
        self.qmax = 2 ** (bits - 1) - 1

    def calibrate(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """校准：计算量化参数"""
        max_val = torch.max(torch.abs(tensor))
        scale = max_val / self.qmax
        return {"scale": scale}

    def quantize(self, tensor: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """量化"""
        scale = params["scale"]
        q_tensor = torch.clamp(torch.round(tensor / scale), self.qmin, self.qmax)
        return q_tensor

    def dequantize(self, q_tensor: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """反量化"""
        scale = params["scale"]
        return q_tensor * scale


class PerTensorAsymmetricQuantizer:
    """Per-Tensor非对称量化器

    量化公式: Q = clamp(round(x / scale + zero_point), qmin, qmax)
    scale = (max - min) / (2^bits - 1)
    zero_point = round(-min / scale)

    优点: 更适合非对称分布的数据
    缺点: 需要额外存储zero_point
    """

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin = 0
        self.qmax = 2 ** bits - 1

    def calibrate(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """校准：计算量化参数"""
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        scale = (max_val - min_val) / self.qmax
        zero_point = torch.round(-min_val / scale)
        return {"scale": scale, "zero_point": zero_point}

    def quantize(self, tensor: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """量化"""
        scale = params["scale"]
        zero_point = params["zero_point"]
        q_tensor = torch.clamp(
            torch.round(tensor / scale + zero_point),
            self.qmin, self.qmax
        )
        return q_tensor

    def dequantize(self, q_tensor: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """反量化"""
        scale = params["scale"]
        zero_point = params["zero_point"]
        return (q_tensor - zero_point) * scale


class PerChannelSymmetricQuantizer:
    """Per-Channel对称量化器

    特点: 为每个输出通道独立计算scale

    优点: 对不同通道的动态范围适应性更好，精度更高
    缺点: 需要存储多个scale值，计算稍复杂
    """

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin = -2 ** (bits - 1)
        self.qmax = 2 ** (bits - 1) - 1

    def calibrate(self, tensor: torch.Tensor, channel_dim: int = 0) -> Dict[str, torch.Tensor]:
        """校准：为每个通道计算scale"""
        # 获取通道数
        num_channels = tensor.shape[channel_dim]

        # 计算每个通道的最大绝对值
        # 需要重塑tensor以便计算每个通道的统计量
        if channel_dim == 0:
            # 对于权重 [out_features, in_features]
            max_vals = torch.max(torch.abs(tensor.reshape(num_channels, -1)), dim=1)[0]
        else:
            # 其他情况
            max_vals = torch.max(torch.abs(tensor), dim=channel_dim, keepdim=True)[0]

        scales = max_vals / self.qmax
        # 避免除以零
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))

        return {"scales": scales, "channel_dim": torch.tensor(channel_dim)}

    def quantize(self, tensor: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """量化"""
        scales = params["scales"]
        channel_dim = params["channel_dim"].item()

        # 扩展scales以匹配tensor形状
        if channel_dim == 0 and len(tensor.shape) == 2:
            scales = scales.view(-1, 1)

        q_tensor = torch.clamp(torch.round(tensor / scales), self.qmin, self.qmax)
        return q_tensor

    def dequantize(self, q_tensor: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """反量化"""
        scales = params["scales"]
        channel_dim = params["channel_dim"].item()

        # 扩展scales以匹配tensor形状
        if channel_dim == 0 and len(q_tensor.shape) == 2:
            scales = scales.view(-1, 1)

        return q_tensor * scales


class PerChannelAsymmetricQuantizer:
    """Per-Channel非对称量化器

    特点: 为每个输出通道独立计算scale和zero_point

    优点: 最高精度，最适应数据分布
    缺点: 存储开销最大
    """

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin = 0
        self.qmax = 2 ** bits - 1

    def calibrate(self, tensor: torch.Tensor, channel_dim: int = 0) -> Dict[str, torch.Tensor]:
        """校准：为每个通道计算scale和zero_point"""
        num_channels = tensor.shape[channel_dim]

        if channel_dim == 0:
            # 对于权重 [out_features, in_features]
            reshaped = tensor.reshape(num_channels, -1)
            min_vals = torch.min(reshaped, dim=1)[0]
            max_vals = torch.max(reshaped, dim=1)[0]
        else:
            min_vals = torch.min(tensor, dim=channel_dim, keepdim=True)[0]
            max_vals = torch.max(tensor, dim=channel_dim, keepdim=True)[0]

        scales = (max_vals - min_vals) / self.qmax
        # 避免除以零
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))

        zero_points = torch.round(-min_vals / scales)

        return {
            "scales": scales,
            "zero_points": zero_points,
            "channel_dim": torch.tensor(channel_dim)
        }

    def quantize(self, tensor: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """量化"""
        scales = params["scales"]
        zero_points = params["zero_points"]
        channel_dim = params["channel_dim"].item()

        # 扩展scales和zero_points以匹配tensor形状
        if channel_dim == 0 and len(tensor.shape) == 2:
            scales = scales.view(-1, 1)
            zero_points = zero_points.view(-1, 1)

        q_tensor = torch.clamp(
            torch.round(tensor / scales + zero_points),
            self.qmin, self.qmax
        )
        return q_tensor

    def dequantize(self, q_tensor: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """反量化"""
        scales = params["scales"]
        zero_points = params["zero_points"]
        channel_dim = params["channel_dim"].item()

        # 扩展scales和zero_points以匹配tensor形状
        if channel_dim == 0 and len(q_tensor.shape) == 2:
            scales = scales.view(-1, 1)
            zero_points = zero_points.view(-1, 1)

        return (q_tensor - zero_points) * scales


class GroupWiseQuantizer:
    """分组量化器（主要用于INT4）

    特点: 将tensor分成多个组，每组独立量化

    优点: 比per-tensor精度高，比per-channel存储开销小
    缺点: 需要额外的分组逻辑

    推荐组大小:
    - 128: 标准配置，平衡精度和存储
    - 64: 精度优先
    - 32: 最高精度，存储开销较大
    """

    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self.qmin = -2 ** (bits - 1)
        self.qmax = 2 ** (bits - 1) - 1

    def calibrate(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """校准：为每个组计算scale"""
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()

        # 填充到group_size的倍数
        pad_len = (self.group_size - len(tensor_flat) % self.group_size) % self.group_size
        if pad_len > 0:
            tensor_flat = torch.cat([tensor_flat, torch.zeros(pad_len, device=tensor.device)])

        # 重塑为组
        tensor_groups = tensor_flat.reshape(-1, self.group_size)
        num_groups = tensor_groups.shape[0]

        # 计算每组的scale
        max_vals = torch.max(torch.abs(tensor_groups), dim=1)[0]
        scales = max_vals / self.qmax
        # 避免除以零
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))

        return {
            "scales": scales,
            "original_shape": torch.tensor(original_shape),
            "pad_len": torch.tensor(pad_len)
        }

    def quantize(self, tensor: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """量化"""
        scales = params["scales"]
        pad_len = params["pad_len"].item()

        tensor_flat = tensor.flatten()

        # 填充
        if pad_len > 0:
            tensor_flat = torch.cat([tensor_flat, torch.zeros(pad_len, device=tensor.device)])

        # 重塑为组
        tensor_groups = tensor_flat.reshape(-1, self.group_size)

        # 量化每组
        q_groups = torch.clamp(
            torch.round(tensor_groups / scales.view(-1, 1)),
            self.qmin, self.qmax
        )

        # 恢复原始形状
        q_tensor = q_groups.flatten()[:tensor.numel()].reshape(tensor.shape)

        return q_tensor

    def dequantize(self, q_tensor: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """反量化"""
        scales = params["scales"]
        pad_len = params["pad_len"].item()

        q_flat = q_tensor.flatten()

        # 填充
        if pad_len > 0:
            q_flat = torch.cat([q_flat, torch.zeros(pad_len, device=q_tensor.device)])

        # 重塑为组
        q_groups = q_flat.reshape(-1, self.group_size)

        # 反量化每组
        deq_groups = q_groups * scales.view(-1, 1)

        # 恢复原始形状
        deq_tensor = deq_groups.flatten()[:q_tensor.numel()].reshape(q_tensor.shape)

        return deq_tensor


# ============================================================================
# 量化层实现
# ============================================================================

class QuantizedLinear(nn.Module):
    """量化的线性层

    支持多种量化方案，在前向传播时自动进行量化和反量化
    """

    def __init__(self, original_layer: nn.Linear, config: QuantizationConfig):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.config = config

        # 选择量化器
        self.quantizer = self._create_quantizer(config)

        # 量化权重
        self._quantize_weights(original_layer.weight.data)

        # 处理偏置（通常保持FP32以提高精度）
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None

    def _create_quantizer(self, config: QuantizationConfig):
        """创建量化器"""
        if config.quant_type == "int8_per_tensor_sym":
            return PerTensorSymmetricQuantizer(bits=config.bits)
        elif config.quant_type == "int8_per_tensor_asym":
            return PerTensorAsymmetricQuantizer(bits=config.bits)
        elif config.quant_type == "int8_per_channel_sym":
            return PerChannelSymmetricQuantizer(bits=config.bits)
        elif config.quant_type == "int8_per_channel_asym":
            return PerChannelAsymmetricQuantizer(bits=config.bits)
        elif config.quant_type == "int4_group":
            return GroupWiseQuantizer(bits=config.bits, group_size=config.group_size)
        else:
            raise ValueError(f"Unknown quantization type: {config.quant_type}")

    def _quantize_weights(self, weight: torch.Tensor):
        """量化权重并存储"""
        # 校准
        if hasattr(self.quantizer, 'calibrate'):
            if "per_channel" in self.config.quant_type:
                params = self.quantizer.calibrate(weight, channel_dim=0)
            else:
                params = self.quantizer.calibrate(weight)
        else:
            params = {}

        # 量化
        q_weight = self.quantizer.quantize(weight, params)

        # 存储量化后的权重和参数
        if self.config.bits == 8:
            self.register_buffer('q_weight', q_weight.to(torch.int8))
        else:
            self.register_buffer('q_weight', q_weight.to(torch.int8))  # INT4存储为INT8

        # 存储量化参数
        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                self.register_buffer(f'weight_{key}', value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 收集量化参数
        params = {}
        for name, buffer in self.named_buffers():
            if name.startswith('weight_'):
                param_name = name[len('weight_'):]
                params[param_name] = buffer

        # 反量化权重
        weight = self.quantizer.dequantize(self.q_weight.float(), params)

        # 线性变换
        return F.linear(x, weight, self.bias)


# ============================================================================
# 模型量化函数
# ============================================================================

def quantize_model_comprehensive(
    model: nn.Module,
    config: QuantizationConfig
) -> nn.Module:
    """
    全面的模型量化函数

    Args:
        model: 原始FP32模型
        config: 量化配置

    Returns:
        量化后的模型
    """
    logger.info("=" * 80)
    logger.info(f"开始量化: {config.name}")
    logger.info(f"量化类型: {config.quant_type}")
    logger.info(f"量化位数: {config.bits} bits")
    logger.info(f"对称量化: {config.symmetric}")
    logger.info(f"逐通道量化: {config.per_channel}")
    if config.quant_type == "int4_group":
        logger.info(f"分组大小: {config.group_size}")
    logger.info("=" * 80)

    # 深拷贝模型
    model_quantized = copy.deepcopy(model)
    model_quantized.eval()

    # 统计
    total_layers = 0
    quantized_layers = 0
    skipped_layers = 0

    # 递归替换线性层
    def replace_linear(module, prefix=""):
        nonlocal total_layers, quantized_layers, skipped_layers

        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                total_layers += 1

                # 检查是否跳过第一层和最后一层
                if config.skip_first_last and (total_layers == 1 or "head" in name.lower()):
                    skipped_layers += 1
                    logger.debug(f"跳过层: {full_name}")
                    continue

                # 替换为量化层
                try:
                    quantized_layer = QuantizedLinear(child, config)
                    setattr(module, name, quantized_layer)
                    quantized_layers += 1
                    logger.debug(f"量化层: {full_name}")
                except Exception as e:
                    logger.warning(f"无法量化层 {full_name}: {e}")
                    skipped_layers += 1
            else:
                # 递归处理子模块
                replace_linear(child, full_name)

    replace_linear(model_quantized)

    logger.info(f"量化完成:")
    logger.info(f"  总线性层数: {total_layers}")
    logger.info(f"  已量化: {quantized_layers}")
    logger.info(f"  已跳过: {skipped_layers}")
    logger.info("=" * 80)

    return model_quantized


def get_all_quantization_configs(device: str = "cuda") -> List[QuantizationConfig]:
    """
    获取所有量化配置

    Returns:
        量化配置列表，包括baseline和所有量化方案
    """
    configs = [
        # Baseline - 不进行量化，用于对比
        # 注意：这个配置不会真正量化模型，只是用于标记baseline

        # INT8 Per-Tensor
        QuantizationConfig(
            name="INT8_Per_Tensor_Symmetric",
            quant_type="int8_per_tensor_sym",
            bits=8,
            symmetric=True,
            per_channel=False,
            device=device
        ),
        QuantizationConfig(
            name="INT8_Per_Tensor_Asymmetric",
            quant_type="int8_per_tensor_asym",
            bits=8,
            symmetric=False,
            per_channel=False,
            device=device
        ),

        # INT8 Per-Channel (更高精度)
        QuantizationConfig(
            name="INT8_Per_Channel_Symmetric",
            quant_type="int8_per_channel_sym",
            bits=8,
            symmetric=True,
            per_channel=True,
            device=device
        ),
        QuantizationConfig(
            name="INT8_Per_Channel_Asymmetric",
            quant_type="int8_per_channel_asym",
            bits=8,
            symmetric=False,
            per_channel=True,
            device=device
        ),

        # INT4 Group-wise (不同组大小)
        QuantizationConfig(
            name="INT4_Group_128",
            quant_type="int4_group",
            bits=4,
            group_size=128,
            device=device
        ),
        QuantizationConfig(
            name="INT4_Group_64",
            quant_type="int4_group",
            bits=4,
            group_size=64,
            device=device
        ),
        QuantizationConfig(
            name="INT4_Group_32",
            quant_type="int4_group",
            bits=4,
            group_size=32,
            device=device
        ),
    ]

    return configs


# ============================================================================
# 工具函数
# ============================================================================

def estimate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    估算模型大小

    Returns:
        包含总大小、参数大小、缓冲区大小的字典（单位：MB）
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        "total_mb": total_size / 1024 / 1024,
        "params_mb": param_size / 1024 / 1024,
        "buffers_mb": buffer_size / 1024 / 1024,
    }


def measure_inference_time(
    model: nn.Module,
    input_data: torch.Tensor,
    warmup: int = 5,
    iterations: int = 20,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    测量推理时间

    Args:
        model: 模型
        input_data: 输入数据
        warmup: 预热次数
        iterations: 测试次数
        device: 设备

    Returns:
        包含平均时间、标准差等统计信息的字典
    """
    model.eval()
    model = model.to(device)
    input_data = input_data.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)

    # 同步CUDA
    if device == "cuda":
        torch.cuda.synchronize()

    # 测量
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(input_data)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }
