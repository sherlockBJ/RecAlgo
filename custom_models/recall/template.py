"""
自定义召回模型模板

继承 RecBole 的 GeneralRecommender 基类来实现自定义召回模型。

用法:
    1. 复制此模板
    2. 修改类名和实现
    3. 在 __init__.py 中注册模型
"""

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class CustomRecallModel(GeneralRecommender):
    """
    自定义召回模型模板

    继承 GeneralRecommender 用于一般推荐场景（user-item 交互）
    如果是序列推荐，请继承 SequentialRecommender
    """

    # 指定输入类型
    input_type = InputType.POINTWISE  # 或 PAIRWISE, LISTWISE

    def __init__(self, config, dataset):
        super(CustomRecallModel, self).__init__(config, dataset)

        # 从 config 读取超参数
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config.get("hidden_size", 128)

        # 定义模型组件
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # 可选：添加更多层
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        # 损失函数
        self.loss = nn.BCEWithLogitsLoss()

        # 初始化参数
        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        """
        前向传播

        Args:
            user: 用户 ID tensor
            item: 物品 ID tensor

        Returns:
            预测分数
        """
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)

        # 简单拼接 + MLP（可替换为其他交互方式）
        x = torch.cat([user_e, item_e], dim=-1)
        score = self.mlp(x).squeeze(-1)

        return score

    def calculate_loss(self, interaction):
        """
        计算损失

        Args:
            interaction: RecBole 的 Interaction 对象，包含 batch 数据

        Returns:
            loss tensor
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        score = self.forward(user, item)
        loss = self.loss(score, label.float())

        return loss

    def predict(self, interaction):
        """
        预测分数（用于评估）

        Args:
            interaction: 包含 user 和 item 的 Interaction

        Returns:
            预测分数
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        score = self.forward(user, item)
        return score

    def full_sort_predict(self, interaction):
        """
        全量排序预测（用于 top-k 评估）

        Args:
            interaction: 包含 user 的 Interaction

        Returns:
            所有物品的分数 (batch_size, n_items)
        """
        user = interaction[self.USER_ID]

        user_e = self.user_embedding(user)  # (batch_size, embedding_size)
        all_item_e = self.item_embedding.weight  # (n_items, embedding_size)

        # 计算所有 user-item 对的分数
        # 这里使用简单的内积，如果用 MLP 需要改写
        score = torch.matmul(user_e, all_item_e.T)  # (batch_size, n_items)

        return score
