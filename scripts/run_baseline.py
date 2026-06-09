"""
运行单个 baseline 模型

用法:
    python scripts/run_baseline.py --model BPR --dataset ml-100k
    python scripts/run_baseline.py --model LightGCN --dataset ml-1m --config configs/models/recall/lightgcn.yaml
"""

import argparse
import os
import sys
from datetime import datetime

from recbole.quick_start import run_recbole
from recbole.utils import init_seed


def parse_args():
    parser = argparse.ArgumentParser(description="运行 RecBole baseline 模型")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称，如 BPR, LightGCN, SASRec, DeepFM 等",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-100k",
        help="数据集名称，如 ml-100k, ml-1m, amazon-books 等",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="自定义配置文件路径（可选）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="随机种子",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="是否保存实验结果到 experiments/results/",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    init_seed(args.seed, reproducibility=True)

    # 构建配置
    config_dict = {
        "seed": args.seed,
        "reproducibility": True,
        "data_path": "data/",
        "checkpoint_dir": "experiments/checkpoints/",
        "show_progress": True,
    }

    # 加载自定义配置文件
    config_file_list = []
    if args.config and os.path.exists(args.config):
        config_file_list.append(args.config)

    print(f"\n{'='*60}")
    print(f"模型: {args.model}")
    print(f"数据集: {args.dataset}")
    print(f"随机种子: {args.seed}")
    if args.config:
        print(f"配置文件: {args.config}")
    print(f"{'='*60}\n")

    # 运行模型
    result = run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list if config_file_list else None,
        config_dict=config_dict,
    )

    # 保存结果
    if args.save_result:
        save_dir = "experiments/results/"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            save_dir, f"{args.model}_{args.dataset}_{timestamp}.txt"
        )

        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"\nResults:\n")
            for key, value in result.items():
                f.write(f"  {key}: {value}\n")

        print(f"\n结果已保存到: {result_file}")

    return result


if __name__ == "__main__":
    main()
