"""
BPR: Bayesian Personalized Ranking from Implicit Feedback (UAI 2009)
Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme.

A from-scratch re-implementation for the RecAlgo project.

Core idea
---------
BPR optimizes a pairwise ranking objective on implicit feedback. For each
observed (user u, positive item i), an unobserved item j is sampled. The model
is trained so that the predicted score of the positive pair ranks above the
negative pair:

    x_uij = x_ui - x_uj
    loss  = - sum_{(u,i,j)} ln sigma(x_uij) + lambda * ||Theta||^2

where x_ui is the matrix-factorization score (dot product of user and item
latent vectors) and the last term is L2 regularization on the embeddings.

Implementation notes
--------------------
- input_type = PAIRWISE tells RecBole's sampler to draw one negative item per
  positive interaction and expose it via NEG_ITEM_ID, giving us the (u, i, j)
  triples the BPR objective needs.
- The scorer is a plain dot product (matrix factorization), exactly as in the
  original paper. The only learnable parameters are the user/item embeddings.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class BPR(GeneralRecommender):
    """Bayesian Personalized Ranking with a matrix-factorization scorer."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)

        # Latent dimension of user / item embeddings.
        self.embedding_size = config["embedding_size"]
        # Weight of the L2 regularization term (lambda in the paper).
        self.reg_weight = config["reg_weight"]

        # The only parameters of BPR: one latent vector per user and per item.
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # -ln(sigma(x_ui - x_uj)) ; EmbLoss is the L2 regularizer ||Theta||^2.
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        """Return MF scores x_ui = <p_u, q_i> for aligned user/item id tensors."""
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=-1)

    def calculate_loss(self, interaction):
        """BPR pairwise loss + L2 regularization on the involved embeddings."""
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_score = self.forward(user, pos_item)  # x_ui
        neg_score = self.forward(user, neg_item)  # x_uj
        loss = self.bpr_loss(pos_score, neg_score)

        # Regularize only the embeddings touched in this batch.
        reg = self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )
        return loss + self.reg_weight * reg

    def predict(self, interaction):
        """Score given (user, item) pairs — used by pointwise evaluation."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def full_sort_predict(self, interaction):
        """Score every item for each user — used by top-k ranking metrics."""
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)            # (batch, dim)
        all_item_e = self.item_embedding.weight        # (n_items, dim)
        return torch.matmul(user_e, all_item_e.transpose(0, 1))  # (batch, n_items)
