"""
LightGCN: Simplifying and Powering Graph Convolution Network for
Recommendation (He et al., SIGIR 2020).

A from-scratch re-implementation for the RecAlgo project.

Core idea
---------
LightGCN strips the standard GCN down to its essential component for
collaborative filtering: neighborhood aggregation. It removes feature
transformation and nonlinear activation, keeping only linear propagation over
the user-item interaction graph.

Propagation (light graph convolution), with symmetric normalization:

    e_u^(k+1) = sum_{i in N_u} 1 / sqrt(|N_u| |N_i|) * e_i^(k)
    e_i^(k+1) = sum_{u in N_i} 1 / sqrt(|N_i| |N_u|) * e_u^(k)

In matrix form, stacking user+item embeddings into E^(k):

    E^(k+1) = (D^{-1/2} A D^{-1/2}) E^(k)

where A is the (|U|+|I|) x (|U|+|I|) adjacency matrix of the bipartite graph
and D its degree matrix.

Layer combination — the final embedding is the average over all layers
(alpha_k = 1/(K+1), the paper's default), which also implicitly handles the
over-smoothing problem:

    e_u = sum_{k=0}^{K} alpha_k * e_u^(k)

Only the 0-th layer embeddings are trainable; deeper layers are derived purely
by propagation. The model is trained with the BPR pairwise loss, exactly as in
the paper.

Implementation notes
--------------------
- The normalized adjacency is built once as a sparse tensor and reused every
  forward pass (it does not depend on the embeddings).
- input_type = PAIRWISE so RecBole supplies (user, pos, neg) triples for BPR.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class LightGCN(GeneralRecommender):
    """LightGCN: linear graph propagation + layer combination, BPR-trained."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        self.latent_dim = config["embedding_size"]   # dimension of e^(0)
        self.n_layers = config["n_layers"]           # K: number of propagation layers
        self.reg_weight = config["reg_weight"]       # L2 regularization on e^(0)

        # Only the 0-th layer embeddings are learnable parameters.
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)

        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Precompute the symmetrically-normalized adjacency as a sparse tensor.
        self.norm_adj = self._build_norm_adj(dataset).to(self.device)

        self.apply(xavier_uniform_initialization)

        # Cache for full-sort evaluation; invalidated each training step.
        # Listing these in other_parameter_name makes RecBole save/restore them
        # with the checkpoint, so load_best_model uses the exact propagated
        # embeddings from the best epoch (matches the official model bit-for-bit).
        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def _build_norm_adj(self, dataset):
        """Build D^{-1/2} A D^{-1/2} for the bipartite user-item graph.

        A is (n_users + n_items) square. Items are offset by n_users so users
        occupy [0, n_users) and items occupy [n_users, n_users + n_items).
        """
        inter = dataset.inter_matrix(form="coo").astype(np.float32)  # users x items
        n_nodes = self.n_users + self.n_items

        # Build the symmetric adjacency: upper-right = R, lower-left = R^T.
        rows = np.concatenate([inter.row, inter.col + self.n_users])
        cols = np.concatenate([inter.col + self.n_users, inter.row])
        data = np.ones(len(rows), dtype=np.float32)
        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

        # Symmetric normalization D^{-1/2} A D^{-1/2}. The +1e-7 on the degree
        # mirrors RecBole's reference implementation (avoids divide-by-zero and
        # keeps our numbers bit-comparable to the official model).
        degree = np.array((adj > 0).sum(axis=1)).flatten() + 1e-7
        d_inv_sqrt = np.power(degree, -0.5)
        d_mat = sp.diags(d_inv_sqrt)
        norm_adj = (d_mat @ adj @ d_mat).tocoo()

        # Convert to a torch sparse tensor.
        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size([n_nodes, n_nodes])).coalesce()

    def forward(self):
        """Propagate K layers and return layer-averaged user/item embeddings."""
        e0 = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings = [e0]

        e = e0
        for _ in range(self.n_layers):
            e = torch.sparse.mm(self.norm_adj, e)  # E^(k+1) = A_norm E^(k)
            embeddings.append(e)

        # Layer combination with alpha_k = 1 / (K + 1) (mean over layers).
        final = torch.stack(embeddings, dim=1).mean(dim=1)
        user_all, item_all = torch.split(final, [self.n_users, self.n_items])
        return user_all, item_all

    def calculate_loss(self, interaction):
        """BPR loss on propagated embeddings + L2 reg on the 0-th layer."""
        # Any parameter update invalidates the eval cache.
        self.restore_user_e = None
        self.restore_item_e = None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all, item_all = self.forward()
        u_e = user_all[user]
        pos_e = item_all[pos_item]
        neg_e = item_all[neg_item]

        pos_score = torch.mul(u_e, pos_e).sum(dim=1)
        neg_score = torch.mul(u_e, neg_e).sum(dim=1)
        loss = self.bpr_loss(pos_score, neg_score)

        # Regularize the *initial* (0-th layer) embeddings, as in the paper.
        u_e0 = self.user_embedding(user)
        pos_e0 = self.item_embedding(pos_item)
        neg_e0 = self.item_embedding(neg_item)
        reg = self.reg_loss(u_e0, pos_e0, neg_e0)

        return loss + self.reg_weight * reg

    def predict(self, interaction):
        """Score given (user, item) pairs."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all, item_all = self.forward()
        return torch.mul(user_all[user], item_all[item]).sum(dim=1)

    def full_sort_predict(self, interaction):
        """Score every item for each user (top-k evaluation)."""
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_e = self.restore_user_e[user]
        return torch.matmul(u_e, self.restore_item_e.transpose(0, 1))
