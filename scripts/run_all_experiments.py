#!/usr/bin/env python3
"""
一键运行所有量化实验脚本

本脚本将依次运行所有量化配置，用于论文实验对比：
1. FP32 Baseline
2. INT8 Per-Tensor Symmetric
3. INT8 Per-Channel Symmetric
4. INT4 Group-128
5. INT4 Group-64
6. INT4 Group-32

使用方法：
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --configs fp32 int8_per_tensor  # 只运行指定配置
    python scripts/run_all_experiments.py --skip_trained  # 跳过已训练的配置
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 所有实验配置
EXPERIMENTS = {
    'fp32': {
        'config': 'eth3d_fp32_baseline',
        'description': 'FP32 Baseline - 完整精度基准',
        'expected_time': '6-8 hours'
    },
    'int8_per_tensor': {
        'config': 'eth3d_int8_per_tensor',
        'description': 'INT8 Per-Tensor - 简单量化',
        'expected_time': '6-8 hours'
    },
    'int8_per_channel': {
        'config': 'eth3d_int8_per_channel',
        'description': 'INT8 Per-Channel - 精细量化',
        'expected_time': '6-8 hours'
    },
    'int4_group128': {
        'config': 'eth3d_int4_group128',
        'description': 'INT4 Group-128 - 极致压缩',
        'expected_time': '8-10 hours'
    },
    'int4_group64': {
        'config': 'eth3d_int4_group64',
        'description': 'INT4 Group-64 - 平衡方案',
        'expected_time': '8-10 hours'
    },
    'int4_group32': {
        'config': 'eth3d_int4_group32',
        'description': 'INT4 Group-32 - 高精度方案',
        'expected_time': '8-10 hours'
    },
}


def check_experiment_completed(exp_name: str, log_dir: str = "logs") -> bool:
    """检查实验是否已完成"""
    checkpoint_dir = Path(log_dir) / exp_name / "checkpoints"
    if not checkpoint_dir.exists():
        return False

    # 检查是否有checkpoint文件
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth"))
    return len(checkpoints) > 0


def run_experiment(config_name: str, exp_name: str):
    """运行单个实验"""
    logger.info("=" * 80)
    logger.info(f"开始实验: {exp_name}")
    logger.info(f"配置文件: {config_name}")
    logger.info("=" * 80)

    # 构建训练命令
    cmd = [
        sys.executable,
        "training/launch.py",
        "--config", config_name
    ]

    try:
        # 运行训练
        start_time = datetime.now()
        result = subprocess.run(cmd, check=True)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds() / 3600  # 转换为小时
        logger.info(f"\n✓ 实验完成: {exp_name}")
        logger.info(f"训练时间: {duration:.2f} 小时")

        return True, duration

    except subprocess.CalledProcessError as e:
        logger.error(f"\n✗ 实验失败: {exp_name}")
        logger.error(f"错误: {e}")
        return False, 0
    except KeyboardInterrupt:
        logger.warning("\n用户中断训练")
        return False, 0


def main():
    parser = argparse.ArgumentParser(description="运行所有量化实验")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(EXPERIMENTS.keys()) + ['all'],
        default=['all'],
        help="要运行的实验配置"
    )
    parser.add_argument(
        "--skip_trained",
        action="store_true",
        help="跳过已训练的实验"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅列出将要运行的实验，不实际运行"
    )

    args = parser.parse_args()

    # 确定要运行的实验
    if 'all' in args.configs:
        configs_to_run = list(EXPERIMENTS.keys())
    else:
        configs_to_run = args.configs

    # 打印实验计划
    logger.info("\n" + "=" * 80)
    logger.info("实验计划")
    logger.info("=" * 80)

    total_experiments = 0
    skipped_experiments = 0

    for exp_key in configs_to_run:
        exp = EXPERIMENTS[exp_key]
        exp_name = exp['config']

        # 检查是否已完成
        is_completed = check_experiment_completed(exp_name)
        skip = is_completed and args.skip_trained

        status = "✓ 已完成" if is_completed else "待运行"
        if skip:
            status = "⊘ 跳过"
            skipped_experiments += 1

        logger.info(f"\n{total_experiments + 1}. [{status}] {exp['description']}")
        logger.info(f"   配置: {exp_name}")
        logger.info(f"   预计时间: {exp['expected_time']}")

        if not skip:
            total_experiments += 1

    if args.dry_run:
        logger.info(f"\n总计: {total_experiments} 个实验待运行, {skipped_experiments} 个已跳过")
        logger.info("这是 dry-run 模式，不会实际运行实验")
        return

    if total_experiments == 0:
        logger.info("\n所有实验已完成！")
        return

    # 确认运行
    logger.info("\n" + "=" * 80)
    logger.info(f"将运行 {total_experiments} 个实验")
    logger.info("=" * 80)

    response = input("\n是否继续? (y/n): ")
    if response.lower() != 'y':
        logger.info("取消运行")
        return

    # 运行实验
    results = {}
    successful = 0
    failed = 0

    for exp_key in configs_to_run:
        exp = EXPERIMENTS[exp_key]
        exp_name = exp['config']

        # 跳过已完成的实验
        if args.skip_trained and check_experiment_completed(exp_name):
            logger.info(f"\n跳过已完成的实验: {exp_name}")
            continue

        # 运行实验
        success, duration = run_experiment(exp['config'], exp_name)

        results[exp_key] = {
            'success': success,
            'duration': duration,
            'description': exp['description']
        }

        if success:
            successful += 1
        else:
            failed += 1

            # 询问是否继续
            response = input("\n实验失败，是否继续下一个实验? (y/n): ")
            if response.lower() != 'y':
                break

    # 总结结果
    logger.info("\n" + "=" * 80)
    logger.info("实验总结")
    logger.info("=" * 80)

    for exp_key, result in results.items():
        status = "✓ 成功" if result['success'] else "✗ 失败"
        duration = f"{result['duration']:.2f}h" if result['success'] else "N/A"
        logger.info(f"{status} - {result['description']} (耗时: {duration})")

    logger.info(f"\n总计: {successful} 成功, {failed} 失败")

    # 保存结果
    results_file = Path("experiment_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n实验结果已保存至: {results_file}")

    logger.info("\n下一步:")
    logger.info("  1. 查看训练日志: logs/eth3d_*/tensorboard")
    logger.info("  2. 评估模型性能: python scripts/evaluate_all_models.py")
    logger.info("  3. 生成对比报告: python scripts/generate_comparison_report.py")


if __name__ == "__main__":
    main()
