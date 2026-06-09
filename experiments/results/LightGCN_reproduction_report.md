# LightGCN Reproduction Report

**Paper:** LightGCN: Simplifying and Powering Graph Convolution Network for
Recommendation (He et al., SIGIR 2020)
**Implementation:** from-scratch, `custom_models/recall/lightgcn.py`
**Dataset (this report):** MovieLens 100K (ml-100k) — *local correctness check*
**Config:** `configs/models/recall/lightgcn.yaml`
(embedding_size=64, n_layers=3, reg_weight=1e-4, 80/10/10 RS split, full-sort eval)
**Seed:** 2024 (reproducibility=True)

## Part 1 — Correctness: hand-written vs RecBole official LightGCN

Both runs use the *same* config and seed. After aligning two
implementation details (see below), **every metric matches to the last
decimal place**, confirming the from-scratch implementation is mathematically
and numerically equivalent to the reference.

| Metric        | Custom (ours) | Official RecBole | Match |
|---------------|---------------|------------------|-------|
| Recall@10     | 0.2439        | 0.2439           | yes   |
| Recall@20     | 0.3666        | 0.3666           | yes   |
| MRR@10        | 0.4034        | 0.4034           | yes   |
| MRR@20        | 0.4119        | 0.4119           | yes   |
| NDCG@10       | 0.2518        | 0.2518           | yes   |
| NDCG@20       | 0.2773        | 0.2773           | yes   |
| Hit@10        | 0.7084        | 0.7084           | yes   |
| Hit@20        | 0.8293        | 0.8293           | yes   |
| Precision@10  | 0.1624        | 0.1624           | yes   |
| Precision@20  | 0.1243        | 0.1243           | yes   |

Best validation epoch: 154 (identical for both).

### Two alignment details that mattered

The first hand-written run already matched the official model on the
*validation* set bit-for-bit, but the *test* numbers were ~1% off. Tracking
that gap down surfaced two non-obvious details in RecBole's reference:

1. **Degree epsilon.** The official model normalizes with
   `degree = (A>0).sum(axis=1) + 1e-7` before `power(-0.5)`. We initially
   skipped zero-degree nodes via a mask; switching to the same `+1e-7` keeps
   the normalized adjacency numerically identical. (No effect on ml-100k after
   the `[5,inf)` k-core filter, but it matters on sparser graphs.)
2. **Propagation cache in the checkpoint.** The official model declares
   `other_parameter_name = ["restore_user_e", "restore_item_e"]`, so the
   layer-combined embeddings from the best epoch are *saved into the
   checkpoint* and restored by `load_best_model`. Without this, test-time
   evaluation re-propagates from the loaded weights, which (due to RecBole's
   load/restore ordering) produced a slightly different embedding. Declaring
   the same attribute closed the test-set gap exactly.

## Part 2 — Paper alignment (Table 3): status

The paper reports on **Gowalla / Yelp2018 / Amazon-Book** with the LightGCN
authors' *fixed* train/test split (inherited from NGCF). Those datasets are
**public and obtainable**, so unlike BPR the paper's exact numbers *can* be
targeted — but only with the same fixed split (a random RS split would not
reproduce Table 3). Target numbers (3 layers):

| Dataset      | Recall@20 (paper) | NDCG@20 (paper) |
|--------------|-------------------|-----------------|
| Gowalla      | 0.1830            | 0.1554          |
| Yelp2018     | 0.0649            | 0.0530          |
| Amazon-Book  | 0.0411            | 0.0315          |

This is the GPU-node phase (Gowalla is ~1M interactions, 40k users / 40k
items). See `configs/models/recall/lightgcn_gowalla.yaml` (to be added) and
the run plan.

## How to reproduce Part 1

```bash
# Hand-written implementation
PYTHONPATH=. python scripts/run_custom.py --model LightGCN --dataset ml-100k \
    --config configs/models/recall/lightgcn.yaml --save_result

# Official RecBole LightGCN (needs the scipy shim — see script header)
PYTHONPATH=. python scripts/_compare_lightgcn_official.py --dataset ml-100k \
    --config configs/models/recall/lightgcn.yaml
```

## Environment notes

- RecBole 1.2.0, PyTorch 2.6.0 (CPU local), ray 2.55, kmeans_pytorch.
- **RecBole 1.2.0 official LightGCN is broken on modern scipy**: it calls
  `scipy.sparse.dok_matrix._update`, removed in current scipy, raising
  `AttributeError`. Our implementation builds the graph in COO and avoids this
  entirely. To run the official model for comparison,
  `scripts/_compare_lightgcn_official.py` monkeypatches `_update` back on
  before importing recbole (math unchanged).
- PyTorch 2.6's `weights_only=True` default is patched back to `False` in the
  runner scripts (self-produced, trusted checkpoints).

## What was implemented from scratch

- Symmetric normalized adjacency `Ã = D^{-1/2} A D^{-1/2}` of the bipartite
  user-item graph, built once as a sparse COO tensor.
- K-layer linear light graph convolution `E^{(k+1)} = Ã E^{(k)}` (no feature
  transform, no nonlinearity).
- Layer combination by mean (`α_k = 1/(K+1)`).
- BPR pairwise loss + L2 reg on the 0-th layer embeddings.
- `forward` / `calculate_loss` / `predict` / `full_sort_predict` against
  RecBole's `GeneralRecommender` interface.
