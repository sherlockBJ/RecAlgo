"""
下载常用推荐系统数据集

用法:
    python scripts/download_data.py --dataset ml-100k
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --list
"""

import argparse
import os

from recbole.utils import get_dataset


# 常用数据集列表
DATASETS = {
    # MovieLens 系列
    "ml-100k": "MovieLens 100K - 10万条评分",
    "ml-1m": "MovieLens 1M - 100万条评分",
    "ml-10m": "MovieLens 10M - 1000万条评分",
    # Amazon 系列
    "amazon-books": "Amazon Books - 书籍推荐",
    "amazon-electronics": "Amazon Electronics - 电子产品",
    "amazon-movies": "Amazon Movies - 电影",
    # 其他
    "yelp": "Yelp - 商户评价",
    "gowalla": "Gowalla - 地理位置签到",
    "lastfm": "Last.fm - 音乐",
    "steam": "Steam - 游戏",
}


def parse_args():
    parser = argparse.ArgumentParser(description="下载推荐系统数据集")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称，或 'all' 下载所有常用数据集",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用数据集",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/",
        help="数据保存目录",
    )
    return parser.parse_args()


def list_datasets():
    """列出所有可用数据集"""
    print("\n可用数据集:")
    print("-" * 50)
    for name, desc in DATASETS.items():
        print(f"  {name:20} - {desc}")
    print("-" * 50)
    print("\nRecBole 还支持更多数据集，详见:")
    print("https://recbole.io/dataset_list.html")


def download_dataset(dataset: str, data_path: str):
    """下载单个数据集"""
    print(f"\n正在下载: {dataset}")

    os.makedirs(data_path, exist_ok=True)

    config_dict = {
        "data_path": data_path,
    }

    try:
        # RecBole 会自动下载数据集
        dataset_obj = get_dataset(dataset, config_dict)
        print(f"✓ {dataset} 下载完成")
        print(f"  - 用户数: {dataset_obj.user_num}")
        print(f"  - 物品数: {dataset_obj.item_num}")
        print(f"  - 交互数: {dataset_obj.inter_num}")
        return True
    except Exception as e:
        print(f"✗ {dataset} 下载失败: {e}")
        return False


def main():
    args = parse_args()

    if args.list:
        list_datasets()
        return

    if args.dataset is None:
        print("请指定数据集名称，或使用 --list 查看可用数据集")
        return

    if args.dataset == "all":
        print("下载所有常用数据集...")
        for dataset in DATASETS.keys():
            download_dataset(dataset, args.data_path)
    else:
        download_dataset(args.dataset, args.data_path)


if __name__ == "__main__":
    main()
