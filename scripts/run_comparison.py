"""
批量对比多个模型

用法:
    python scripts/run_comparison.py --models BPR,LightGCN,NeuMF --dataset ml-100k
    python scripts/run_comparison.py --models SASRec,GRU4Rec,BERT4Rec --dataset ml-1m --task sequential
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd
from recbole.quick_start import run_recbole
from recbole.utils import init_seed


# 预定义的模型组
MODEL_GROUPS = {
    "recall_cf": ["BPR", "NeuMF", "LightGCN", "NGCF", "ItemKNN"],
    "recall_graph": ["LightGCN", "NGCF", "SGL", "SimGCL"],
    "sequential": ["GRU4Rec", "SASRec", "BERT4Rec", "Caser", "SRGNN"],
    "ctr": ["FM", "DeepFM", "xDeepFM", "DCN", "AutoInt"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="批量对比多个推荐模型")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="模型列表，用逗号分隔，或使用预定义组名: recall_cf, recall_graph, sequential, ctr",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-100k",
        help="数据集名称",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="随机种子",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出文件名（不含扩展名）",
    )
    return parser.parse_args()


def get_model_list(models_str: str) -> list:
    """解析模型列表"""
    if models_str in MODEL_GROUPS:
        return MODEL_GROUPS[models_str]
    return [m.strip() for m in models_str.split(",")]


def main():
    args = parse_args()

    models = get_model_list(args.models)
    results = []

    print(f"\n{'='*60}")
    print(f"批量对比实验")
    print(f"模型: {models}")
    print(f"数据集: {args.dataset}")
    print(f"{'='*60}\n")

    # 配置
    config_dict = {
        "seed": args.seed,
        "reproducibility": True,
        "data_path": "data/",
        "checkpoint_dir": "experiments/checkpoints/",
        "show_progress": True,
    }

    # 逐个运行模型
    for model in models:
        print(f"\n>>> 正在运行: {model}")
        print("-" * 40)

        try:
            init_seed(args.seed, reproducibility=True)
            result = run_recbole(
                model=model,
                dataset=args.dataset,
                config_dict=config_dict,
            )

            result_dict = {"model": model, **result}
            results.append(result_dict)

            print(f"<<< {model} 完成")

        except Exception as e:
            print(f"<<< {model} 失败: {e}")
            results.append({"model": model, "error": str(e)})

    # 汇总结果
    df = pd.DataFrame(results)

    # 保存结果
    save_dir = "experiments/results/"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output or f"comparison_{args.dataset}_{timestamp}"

    # 保存 CSV
    csv_path = os.path.join(save_dir, f"{output_name}.csv")
    df.to_csv(csv_path, index=False)

    # 保存 JSON
    json_path = os.path.join(save_dir, f"{output_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("实验结果汇总:")
    print(df.to_string(index=False))
    print(f"\n结果已保存到:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
