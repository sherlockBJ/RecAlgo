"""
Prepare the Gowalla dataset using the LightGCN authors' *fixed* train/test
split, converted into RecBole atomic files.

Why the fixed split matters
---------------------------
LightGCN's Table 3 numbers (Gowalla Recall@20=0.1830, NDCG@20=0.1554) are
computed on the exact train/test partition released with NGCF and reused by
LightGCN. A random 80/10/10 split would train and test on different
interactions and therefore *cannot* reproduce the paper's numbers. To align
with the paper we must consume the same `train.txt` / `test.txt`.

Source files (LightGCN official repo, identical to NGCF's):
    https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla/train.txt
    https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla/test.txt
Each line: `user_id item_id item_id item_id ...` (space-separated, one user/line).

Output (RecBole "benchmark" pre-split atomic files):
    data/gowalla/gowalla.train.inter
    data/gowalla/gowalla.test.inter
    data/gowalla/gowalla.valid.inter   (small slice carved from train for early stopping)

Each .inter file:
    user_id:token<TAB>item_id:token
    <u>\t<i>
    ...

Run on the machine that has network access (GPU node or local), then point the
config at it. This script only writes files; it does not train.
"""

import argparse
import os
import random
import urllib.request

BASE = "https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla"


def download(url, dest):
    if os.path.exists(dest):
        print(f"[skip] {dest} already exists")
        return
    print(f"[get ] {url}")
    urllib.request.urlretrieve(url, dest)


def parse_adj_txt(path):
    """Yield (user, item) pairs from a `user item item ...` adjacency file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = parts[0]
            for item in parts[1:]:
                yield user, item


def write_inter(pairs, dest):
    with open(dest, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\n")
        for user, item in pairs:
            f.write(f"{user}\t{item}\n")
    print(f"[write] {dest} ({len(pairs)} interactions)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/gowalla")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="fraction of TRAIN carved out for early-stopping validation")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    raw_train = os.path.join(args.out, "_train.txt")
    raw_test = os.path.join(args.out, "_test.txt")
    download(f"{BASE}/train.txt", raw_train)
    download(f"{BASE}/test.txt", raw_test)

    train_pairs = list(parse_adj_txt(raw_train))
    test_pairs = list(parse_adj_txt(raw_test))

    # Carve a small validation slice out of TRAIN only. The TEST set stays
    # exactly the paper's, so the reported test metric is comparable to Table 3.
    rng = random.Random(args.seed)
    rng.shuffle(train_pairs)
    n_valid = int(len(train_pairs) * args.valid_ratio)
    valid_pairs = train_pairs[:n_valid]
    fit_pairs = train_pairs[n_valid:]

    write_inter(fit_pairs, os.path.join(args.out, "gowalla.train.inter"))
    write_inter(valid_pairs, os.path.join(args.out, "gowalla.valid.inter"))
    write_inter(test_pairs, os.path.join(args.out, "gowalla.test.inter"))

    print("\nDone. Train with:")
    print("  PYTHONPATH=. python scripts/run_custom.py --model LightGCN "
          "--dataset gowalla --config configs/models/recall/lightgcn_gowalla.yaml --save_result")


if __name__ == "__main__":
    main()
