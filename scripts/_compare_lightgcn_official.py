"""
Run RecBole's *built-in* LightGCN with a scipy-compatibility shim, so it can be
compared against our from-scratch implementation on equal footing.

Why this exists
---------------
RecBole 1.2.0's LightGCN builds its normalized adjacency with
`scipy.sparse.dok_matrix._update(...)`. `_update` was an internal method back
when dok_matrix subclassed `dict`; modern scipy dropped it, so the built-in
model crashes with `AttributeError: 'dok_matrix' object has no attribute
'_update'`. Our own LightGCN sidesteps this by building the graph in COO format.

To get an apples-to-apples reference number we monkeypatch the missing method
back onto dok_matrix (delegating to the backing dict) before importing recbole.
This changes nothing about the math — it only restores the old in-place update.

Usage:
    python scripts/_compare_lightgcn_official.py --dataset ml-100k \
        --config configs/models/recall/lightgcn.yaml
"""

import argparse
import os

import scipy.sparse as sp


# --- scipy shim: restore dok_matrix._update for RecBole 1.2.0 -----------------
if not hasattr(sp.dok_matrix, "_update"):
    def _dok_update(self, data):
        # Newer scipy keeps entries in self._dict; older exposed dict.update.
        backing = getattr(self, "_dict", self)
        dict.update(backing, data) if isinstance(backing, dict) else backing.update(data)
    sp.dok_matrix._update = _dok_update
# -----------------------------------------------------------------------------

import torch

_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from recbole.quick_start import run_recbole
from recbole.utils import init_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-100k")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    init_seed(args.seed, reproducibility=True)
    config_file_list = [args.config] if args.config and os.path.exists(args.config) else None
    result = run_recbole(
        model="LightGCN",
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict={
            "seed": args.seed,
            "reproducibility": True,
            "data_path": "data/",
            "checkpoint_dir": "experiments/checkpoints/",
            "show_progress": True,
        },
    )
    print("OFFICIAL_LIGHTGCN_TEST_RESULT:", result["test_result"])


if __name__ == "__main__":
    main()
