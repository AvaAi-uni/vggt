# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
INT8 Quantization utilities for VGGT model.

This module provides tools to convert VGGT from FP32/FP16/BF16 to INT8,
significantly reducing memory usage while maintaining accuracy.

Key features:
- Post-Training Quantization (PTQ) with calibration
- Dynamic quantization for linear layers
- Static quantization with observer-based calibration
- Memory-efficient quantization schemes
"""

import torch
import torch.nn as nn
from torch.quantization import (
    quantize_dynamic,
    prepare_qat,
    convert,
    QConfig,
    default_qconfig,
    default_dynamic_qconfig,
    get_default_qconfig,
)
from torch.ao.quantization import (
    get_default_qat_qconfig,
    QConfigMapping,
    prepare,
    MinMaxObserver,
    HistogramObserver,
    PerChannelMinMaxObserver,
)
import copy
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.

    Args:
        quantization_type: Type of quantization ('dynamic', 'static', or 'qat')
        calibration_samples: Number of samples for calibration (static quantization)
        observer_type: Type of observer for calibration ('minmax', 'histogram', 'per_channel')
        quantize_embeddings: Whether to quantize embedding layers
        quantize_attention: Whether to quantize attention layers
        quantize_heads: Whether to quantize prediction heads
        symmetric: Use symmetric quantization (default: True)
        reduce_range: Reduce quantization range for better compatibility (default: False)
    """
    quantization_type: str = "dynamic"  # 'dynamic', 'static', or 'qat'
    calibration_samples: int = 100
    observer_type: str = "minmax"  # 'minmax', 'histogram', 'per_channel'
    quantize_embeddings: bool = False
    quantize_attention: bool = True
    quantize_heads: bool = True
    symmetric: bool = True
    reduce_range: bool = False
    dtype: torch.dtype = torch.qint8


def get_qconfig_mapping(config: QuantizationConfig) -> QConfigMapping:
    """
    Create QConfig mapping based on quantization configuration.

    Args:
        config: QuantizationConfig instance

    Returns:
        QConfigMapping for the specified configuration
    """
    qconfig_mapping = QConfigMapping()

    # Select observer based on config
    if config.observer_type == "minmax":
        activation_observer = MinMaxObserver
        weight_observer = MinMaxObserver
    elif config.observer_type == "histogram":
        activation_observer = HistogramObserver
        weight_observer = MinMaxObserver
    elif config.observer_type == "per_channel":
        activation_observer = MinMaxObserver
        weight_observer = PerChannelMinMaxObserver
    else:
        raise ValueError(f"Unknown observer type: {config.observer_type}")

    # Get default qconfig
    if config.quantization_type == "qat":
        qconfig = get_default_qat_qconfig("x86")
    else:
        qconfig = get_default_qconfig("x86")

    # Set global qconfig
    qconfig_mapping.set_global(qconfig)

    return qconfig_mapping


def prepare_model_for_quantization(
    model: nn.Module,
    config: QuantizationConfig,
    example_inputs: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Prepare model for quantization by fusing operations and inserting observers.

    Args:
        model: The model to prepare
        config: Quantization configuration
        example_inputs: Example inputs for tracing (required for static quantization)

    Returns:
        Prepared model with observers inserted
    """
    logger.info(f"Preparing model for {config.quantization_type} quantization...")

    model_copy = copy.deepcopy(model)
    model_copy.eval()

    if config.quantization_type == "dynamic":
        # Dynamic quantization doesn't need preparation
        logger.info("Dynamic quantization selected - no preparation needed")
        return model_copy

    elif config.quantization_type in ["static", "qat"]:
        # Fuse operations for better performance
        # Note: VGGT uses LayerNorm and GELU, which can be fused
        logger.info("Fusing operations...")
        # model_copy = torch.quantization.fuse_modules(model_copy, [['conv', 'bn', 'relu']])

        # Get qconfig mapping
        qconfig_mapping = get_qconfig_mapping(config)

        # Prepare model
        if config.quantization_type == "qat":
            prepared_model = prepare_qat(model_copy, qconfig_mapping)
        else:
            prepared_model = prepare(model_copy, qconfig_mapping)

        logger.info("Model prepared for quantization")
        return prepared_model

    else:
        raise ValueError(f"Unknown quantization type: {config.quantization_type}")


def calibrate_model(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    num_samples: int = 100,
    device: str = "cuda",
) -> nn.Module:
    """
    Calibrate the model using representative data samples.

    Args:
        model: Prepared model with observers
        calibration_loader: DataLoader with calibration data
        num_samples: Number of samples to use for calibration
        device: Device to run calibration on

    Returns:
        Calibrated model ready for conversion
    """
    logger.info(f"Calibrating model with {num_samples} samples...")

    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= num_samples:
                break

            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            elif isinstance(batch, dict):
                images = batch["images"]
            else:
                images = batch

            images = images.to(device)

            # Forward pass to collect statistics
            try:
                _ = model(images)
            except Exception as e:
                logger.warning(f"Error during calibration batch {i}: {e}")
                continue

            if (i + 1) % 10 == 0:
                logger.info(f"Calibrated {i + 1}/{num_samples} samples")

    logger.info("Calibration completed")
    return model


def convert_to_quantized(model: nn.Module) -> nn.Module:
    """
    Convert prepared and calibrated model to quantized version.

    Args:
        model: Prepared and calibrated model

    Returns:
        Quantized model
    """
    logger.info("Converting model to quantized version...")
    quantized_model = convert(model)
    logger.info("Conversion completed")
    return quantized_model


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
) -> nn.Module:
    """
    Main function to quantize VGGT model from FP32 to INT8.

    Args:
        model: Original FP32/FP16/BF16 model
        config: Quantization configuration
        calibration_loader: DataLoader for calibration (required for static quantization)

    Returns:
        Quantized INT8 model

    Example:
        >>> from vggt.models.vggt import VGGT
        >>> from vggt.quantization import quantize_model, QuantizationConfig
        >>>
        >>> # Load original model
        >>> model = VGGT.from_pretrained("facebook/VGGT-1B")
        >>>
        >>> # Configure quantization
        >>> config = QuantizationConfig(
        ...     quantization_type="dynamic",
        ...     quantize_attention=True,
        ...     quantize_heads=True,
        ... )
        >>>
        >>> # Quantize model
        >>> quantized_model = quantize_model(model, config)
        >>>
        >>> # Use quantized model
        >>> with torch.no_grad():
        ...     predictions = quantized_model(images)
    """
    logger.info("Starting model quantization...")
    logger.info(f"Configuration: {config}")

    if config.quantization_type == "dynamic":
        # Dynamic quantization - simplest approach
        logger.info("Applying dynamic quantization...")

        # Specify layers to quantize
        layers_to_quantize = {nn.Linear}
        if config.quantize_attention:
            layers_to_quantize.add(nn.MultiheadAttention)

        quantized_model = quantize_dynamic(
            model,
            qconfig_spec=layers_to_quantize,
            dtype=config.dtype,
        )

        logger.info("Dynamic quantization completed")
        return quantized_model

    elif config.quantization_type == "static":
        # Static quantization - requires calibration
        if calibration_loader is None:
            raise ValueError("calibration_loader is required for static quantization")

        # Prepare model
        prepared_model = prepare_model_for_quantization(model, config)

        # Calibrate model
        calibrated_model = calibrate_model(
            prepared_model,
            calibration_loader,
            num_samples=config.calibration_samples,
        )

        # Convert to quantized
        quantized_model = convert_to_quantized(calibrated_model)

        logger.info("Static quantization completed")
        return quantized_model

    elif config.quantization_type == "qat":
        # Quantization-aware training
        logger.warning("QAT selected - this requires training. Returning prepared model.")
        prepared_model = prepare_model_for_quantization(model, config)
        logger.info("Model prepared for QAT - please train before conversion")
        return prepared_model

    else:
        raise ValueError(f"Unknown quantization type: {config.quantization_type}")


def estimate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Estimate model size in different formats.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with size estimates in MB
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


def compare_model_outputs(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_input: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compare outputs between original and quantized models.

    Args:
        original_model: Original FP32 model
        quantized_model: Quantized INT8 model
        test_input: Test input tensor
        device: Device to run comparison on

    Returns:
        Dictionary with comparison metrics
    """
    original_model.eval()
    quantized_model.eval()

    # Get original output (keep on CPU if quantized model is on CPU)
    with torch.no_grad():
        if device == "cuda" and torch.cuda.is_available():
            original_model = original_model.to(device)
            test_input = test_input.to(device)

        original_output = original_model(test_input)

        # Quantized model typically runs on CPU
        quantized_model = quantized_model.to("cpu")
        test_input_cpu = test_input.to("cpu")
        quantized_output = quantized_model(test_input_cpu)

    # Compare outputs
    if isinstance(original_output, dict):
        # Compare each output
        metrics = {}
        for key in original_output.keys():
            if key == "images":
                continue  # Skip images comparison

            orig = original_output[key].cpu().float()
            quant = quantized_output[key].cpu().float()

            # Calculate metrics
            mse = torch.mean((orig - quant) ** 2).item()
            mae = torch.mean(torch.abs(orig - quant)).item()
            max_diff = torch.max(torch.abs(orig - quant)).item()

            metrics[f"{key}_mse"] = mse
            metrics[f"{key}_mae"] = mae
            metrics[f"{key}_max_diff"] = max_diff

        return metrics
    else:
        orig = original_output.cpu().float()
        quant = quantized_output.cpu().float()

        return {
            "mse": torch.mean((orig - quant) ** 2).item(),
            "mae": torch.mean(torch.abs(orig - quant)).item(),
            "max_diff": torch.max(torch.abs(orig - quant)).item(),
        }
