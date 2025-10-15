#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
量化感知训练 (Quantization-Aware Training, QAT)

这个模块实现了从FP32到INT8/INT4的量化感知训练框架，适合论文研究使用。

主要特点：
1. 支持多种量化方案（Per-Tensor, Per-Channel, Group-wise）
2. 渐进式量化：FP32 -> INT8 -> INT4
3. 混合精度训练：敏感层保持高精度
4. 详细的性能评估和对比

参考文献：
[1] Jacob et al., "Quantization and Training of Neural Networks for Efficient
    Integer-Arithmetic-Only Inference", CVPR 2018
[2] Shen et al., "Q-BERT: Hessian Based Ultra Low Precision Quantization",
    AAAI 2020
[3] Liu et al., "Post-training Quantization for Vision Transformer", NeurIPS 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import copy

from .comprehensive_quantizer import (
    QuantizationConfig,
    PerTensorSymmetricQuantizer,
    PerChannelSymmetricQuantizer,
    GroupWiseQuantizer,
    PerTensorAsymmetricQuantizer,
    PerChannelAsymmetricQuantizer,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 量化感知训练配置
# ============================================================================

@dataclass
class QATConfig:
    """
    量化感知训练配置

    Args:
        quant_config: 基础量化配置
        enable_qat: 是否启用QAT
        freeze_bn: 是否冻结BatchNorm层
        qat_start_epoch: 开始QAT的epoch
        use_ema: 是否使用指数移动平均
        ema_decay: EMA衰减率
        straight_through_estimator: 是否使用直通估计器（STE）
    """
    quant_config: QuantizationConfig
    enable_qat: bool = True
    freeze_bn: bool = True
    qat_start_epoch: int = 0
    use_ema: bool = False
    ema_decay: float = 0.999
    straight_through_estimator: bool = True


# ============================================================================
# 量化感知层实现
# ============================================================================

class FakeQuantize(nn.Module):
    """
    伪量化模块 - 用于QAT

    在训练时模拟量化效果，但保持浮点运算以支持反向传播
    """

    def __init__(
        self,
        quantizer,
        per_channel: bool = False,
        channel_dim: int = 0
    ):
        super().__init__()
        self.quantizer = quantizer
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.enabled = True

        # 量化参数（会在训练中学习）
        self.scale = None
        self.zero_point = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.training:
            return x

        # 计算量化参数
        if self.scale is None:
            if self.per_channel:
                params = self.quantizer.calibrate(x, channel_dim=self.channel_dim)
            else:
                params = self.quantizer.calibrate(x)

            # 注册为buffer（不参与梯度更新）
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    self.register_buffer(key, value)

        # 收集量化参数
        params = {}
        for name, buffer in self.named_buffers():
            params[name] = buffer

        # 伪量化：量化后立即反量化
        # 使用Straight-Through Estimator处理梯度
        x_quant = self.quantizer.quantize(x, params)
        x_dequant = self.quantizer.dequantize(x_quant, params)

        # STE: 前向传播使用量化值，反向传播使用原始梯度
        return x + (x_dequant - x).detach()


class QATLinear(nn.Module):
    """
    量化感知训练的线性层

    在训练时使用伪量化，推理时可转换为真实量化层
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qat_config: QATConfig = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qat_config = qat_config

        # 原始线性层参数
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # 权重伪量化
        if qat_config and qat_config.enable_qat:
            quantizer = self._create_quantizer(qat_config.quant_config)
            self.weight_fake_quant = FakeQuantize(
                quantizer,
                per_channel=qat_config.quant_config.per_channel,
                channel_dim=0
            )
        else:
            self.weight_fake_quant = None

        # 激活伪量化（如果启用）
        if qat_config and qat_config.enable_qat and qat_config.quant_config.quantize_activations:
            act_quantizer = PerTensorSymmetricQuantizer(bits=8)
            self.act_fake_quant = FakeQuantize(act_quantizer)
        else:
            self.act_fake_quant = None

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 激活量化
        if self.act_fake_quant is not None:
            x = self.act_fake_quant(x)

        # 权重量化
        weight = self.weight
        if self.weight_fake_quant is not None:
            weight = self.weight_fake_quant(weight)

        return F.linear(x, weight, self.bias)

    @classmethod
    def from_float(cls, float_linear: nn.Linear, qat_config: QATConfig):
        """从浮点线性层创建QAT层"""
        qat_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            float_linear.bias is not None,
            qat_config
        )
        qat_linear.weight.data = float_linear.weight.data.clone()
        if float_linear.bias is not None:
            qat_linear.bias.data = float_linear.bias.data.clone()
        return qat_linear


# ============================================================================
# QAT训练器
# ============================================================================

class QATTrainer:
    """
    量化感知训练器

    支持渐进式量化训练：
    1. FP32 baseline训练
    2. INT8 QAT fine-tuning
    3. INT4 QAT fine-tuning
    """

    def __init__(
        self,
        model: nn.Module,
        qat_configs: List[QATConfig],
        device: str = "cuda"
    ):
        """
        Args:
            model: 要训练的模型
            qat_configs: QAT配置列表（支持渐进式量化）
            device: 训练设备
        """
        self.original_model = model
        self.qat_configs = qat_configs
        self.device = device
        self.current_qat_model = None
        self.current_config_idx = 0

    def prepare_qat_model(self, config_idx: int = 0) -> nn.Module:
        """
        准备QAT模型

        Args:
            config_idx: 使用的配置索引

        Returns:
            准备好的QAT模型
        """
        if config_idx >= len(self.qat_configs):
            raise ValueError(f"Invalid config index: {config_idx}")

        self.current_config_idx = config_idx
        qat_config = self.qat_configs[config_idx]

        logger.info("=" * 80)
        logger.info(f"Preparing QAT model with config: {qat_config.quant_config.name}")
        logger.info(f"  Quantization type: {qat_config.quant_config.quant_type}")
        logger.info(f"  Bits: {qat_config.quant_config.bits}")
        logger.info("=" * 80)

        # 深拷贝模型
        qat_model = copy.deepcopy(self.original_model)

        # 转换线性层为QAT层
        qat_model = self._convert_to_qat(qat_model, qat_config)

        # 冻结BatchNorm
        if qat_config.freeze_bn:
            self._freeze_bn(qat_model)

        self.current_qat_model = qat_model
        return qat_model

    def _convert_to_qat(self, module: nn.Module, qat_config: QATConfig, prefix: str = "") -> nn.Module:
        """递归转换模块为QAT模块"""
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                # 检查是否跳过
                if qat_config.quant_config.skip_first_last and ("head" in name.lower() or "embed" in name.lower()):
                    logger.debug(f"Skipping QAT conversion for: {full_name}")
                    continue

                # 转换为QAT层
                qat_layer = QATLinear.from_float(child, qat_config)
                setattr(module, name, qat_layer)
                logger.debug(f"Converted to QAT layer: {full_name}")
            else:
                # 递归处理
                self._convert_to_qat(child, qat_config, full_name)

        return module

    def _freeze_bn(self, model: nn.Module):
        """冻结所有BatchNorm层"""
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        logger.info("BatchNorm layers frozen")

    def enable_quantization(self, enabled: bool = True):
        """启用或禁用量化"""
        if self.current_qat_model is None:
            return

        for module in self.current_qat_model.modules():
            if isinstance(module, FakeQuantize):
                module.enabled = enabled

    def export_quantized_model(self) -> nn.Module:
        """
        导出量化后的模型（用于推理）

        Returns:
            量化后的模型
        """
        if self.current_qat_model is None:
            raise ValueError("No QAT model prepared")

        # TODO: 实现真实量化层的转换
        # 这里返回当前的QAT模型
        logger.warning("Export to real quantized model not fully implemented yet")
        return self.current_qat_model


# ============================================================================
# 渐进式量化训练策略
# ============================================================================

def create_progressive_qat_configs(device: str = "cuda") -> List[QATConfig]:
    """
    创建渐进式QAT配置

    训练策略：
    1. Stage 1: FP32 baseline (无量化)
    2. Stage 2: INT8 Per-Channel QAT
    3. Stage 3: INT4 Group-wise QAT

    Returns:
        QAT配置列表
    """
    configs = []

    # Stage 1: FP32 Baseline (不启用QAT)
    fp32_config = QuantizationConfig(
        name="FP32_Baseline",
        quant_type="int8_per_tensor_sym",  # 占位符
        bits=32,
        device=device
    )
    configs.append(QATConfig(
        quant_config=fp32_config,
        enable_qat=False,  # 禁用QAT
        qat_start_epoch=0
    ))

    # Stage 2: INT8 Per-Channel Symmetric QAT
    int8_config = QuantizationConfig(
        name="INT8_Per_Channel_Symmetric_QAT",
        quant_type="int8_per_channel_sym",
        bits=8,
        symmetric=True,
        per_channel=True,
        skip_first_last=True,
        device=device
    )
    configs.append(QATConfig(
        quant_config=int8_config,
        enable_qat=True,
        freeze_bn=True,
        qat_start_epoch=0
    ))

    # Stage 3: INT4 Group-wise QAT
    int4_config = QuantizationConfig(
        name="INT4_Group_128_QAT",
        quant_type="int4_group",
        bits=4,
        group_size=128,
        skip_first_last=True,
        device=device
    )
    configs.append(QATConfig(
        quant_config=int4_config,
        enable_qat=True,
        freeze_bn=True,
        qat_start_epoch=0
    ))

    return configs


# ============================================================================
# 辅助函数
# ============================================================================

def count_qat_layers(model: nn.Module) -> Dict[str, int]:
    """
    统计QAT层数量

    Returns:
        包含各类层数量的字典
    """
    stats = {
        'qat_linear': 0,
        'fake_quant': 0,
        'total_params': 0,
        'trainable_params': 0,
    }

    for module in model.modules():
        if isinstance(module, QATLinear):
            stats['qat_linear'] += 1
        elif isinstance(module, FakeQuantize):
            stats['fake_quant'] += 1

    stats['total_params'] = sum(p.numel() for p in model.parameters())
    stats['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return stats
