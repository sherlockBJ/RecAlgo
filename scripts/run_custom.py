"""
Run a from-scratch model implementation from custom_models/.

RecBole's run_recbole() resolves the model by *name* against its built-in
registry, so it cannot drive a hand-written class directly. Instead we assemble
the pipeline from RecBole's lower-level building blocks and instantiate our own
model class explicitly:

    config -> dataset -> data_preparation -> OUR model class -> trainer -> eval

This is the same pipeline run_recbole() uses internally; we only swap in our
class at the model-construction step, so training/evaluation stay identical to
the built-in baselines.

Usage:
    python scripts/run_custom.py --model BPR --config configs/models/recall/bpr.yaml
    python scripts/run_custom.py --model BPR --dataset ml-100k --save_result
"""

import argparse
import os
from datetime import datetime
from logging import getLogger

import torch

# PyTorch 2.6 flipped torch.load's default to weights_only=True, which breaks
# loading RecBole 1.2.0 checkpoints (they pickle the full Config object). The
# checkpoints are produced by this same script, so loading with weights_only
# disabled is safe. Pin the default back so trainer.evaluate(load_best_model)
# works.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color

# Registry of hand-written models: name -> class.
from custom_models.recall.bpr import BPR
from custom_models.recall.lightgcn import LightGCN

CUSTOM_MODELS = {
    "BPR": BPR,
    "LightGCN": LightGCN,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a custom-implemented model")
    parser.add_argument("--model", type=str, required=True,
                        help=f"Custom model name, one of: {list(CUSTOM_MODELS)}")
    parser.add_argument("--dataset", type=str, default="ml-100k")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--save_result", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model not in CUSTOM_MODELS:
        raise ValueError(
            f"Unknown custom model '{args.model}'. Available: {list(CUSTOM_MODELS)}"
        )
    model_class = CUSTOM_MODELS[args.model]

    config_dict = {
        "seed": args.seed,
        "reproducibility": True,
        "data_path": "data/",
        "checkpoint_dir": "experiments/checkpoints/",
        "show_progress": True,
    }
    config_file_list = [args.config] if args.config and os.path.exists(args.config) else None

    # Build config around OUR model class so MODEL_TYPE / input_type are taken
    # from the class itself rather than a name lookup.
    config = Config(
        model=model_class,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    # Data pipeline (identical to run_recbole).
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Instantiate the hand-written model explicitly.
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model)

    # Train + evaluate through RecBole's standard trainer.
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    if args.save_result:
        save_dir = "experiments/results/"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(
            save_dir, f"custom_{args.model}_{args.dataset}_{timestamp}.txt"
        )
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Model: {args.model} (custom implementation)\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Config: {args.config}\n\n")
            f.write(f"Best valid: {best_valid_result}\n\nTest results:\n")
            for key, value in test_result.items():
                f.write(f"  {key}: {value}\n")
        print(f"\nResult saved to: {result_file}")

    return {"best_valid_result": best_valid_result, "test_result": test_result}


if __name__ == "__main__":
    main()
