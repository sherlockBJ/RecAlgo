# BPR Reproduction Report

**Paper:** BPR: Bayesian Personalized Ranking from Implicit Feedback (Rendle et al., UAI 2009)
**Implementation:** from-scratch, `custom_models/recall/bpr.py`
**Dataset:** MovieLens 100K (ml-100k), rating >= 3 as implicit positive
**Config:** `configs/models/recall/bpr.yaml` (embedding_size=64, reg_weight=1e-4, 80/10/10 RS split, full-sort eval)
**Seed:** 2024 (reproducibility=True)

## Result: hand-written vs RecBole official BPR

Both runs use the *same* config and seed. Metrics match to every decimal place,
confirming the from-scratch implementation is mathematically equivalent to the
reference.

| Metric        | Custom (ours) | Official RecBole | Match |
|---------------|---------------|------------------|-------|
| Recall@10     | 0.2534        | 0.2534           | yes   |
| Recall@20     | 0.3755        | 0.3755           | yes   |
| MRR@10        | 0.4197        | 0.4197           | yes   |
| MRR@20        | 0.4273        | 0.4273           | yes   |
| NDCG@10       | 0.2605        | 0.2605           | yes   |
| NDCG@20       | 0.2859        | 0.2859           | yes   |
| Hit@10        | 0.7285        | 0.7285           | yes   |
| Hit@20        | 0.8356        | 0.8356           | yes   |
| Precision@10  | 0.1667        | 0.1667           | yes   |
| Precision@20  | 0.1261        | 0.1261           | yes   |

Best validation epoch: 53 (identical for both).

## How to reproduce

```bash
# Hand-written implementation
PYTHONPATH=. python scripts/run_custom.py --model BPR --dataset ml-100k \
    --config configs/models/recall/bpr.yaml --save_result

# Official RecBole BPR (same config) for comparison
PYTHONPATH=. python scripts/run_baseline.py --model BPR --dataset ml-100k \
    --config configs/models/recall/bpr.yaml --save_result
```

## Environment notes

- RecBole 1.2.0, PyTorch 2.6.0 (CPU), ray 2.55, kmeans_pytorch.
- PyTorch 2.6 changed `torch.load` default to `weights_only=True`, which breaks
  loading RecBole 1.2.0 checkpoints. `scripts/run_custom.py` patches the default
  back to `False` (checkpoints are self-produced and trusted).

## What was implemented from scratch

- MF scorer `x_ui = <p_u, q_i>` (user/item embeddings, dot product).
- BPR pairwise objective `-ln sigma(x_ui - x_uj)` with one uniform negative
  per positive, plus L2 regularization on the embeddings in each batch.
- `forward` / `calculate_loss` / `predict` / `full_sort_predict` against
  RecBole's `GeneralRecommender` interface.
