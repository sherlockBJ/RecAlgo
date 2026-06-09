"""
自定义排序/CTR 模型模板

继承 RecBole 的 ContextRecommender 基类来实现自定义 CTR 预估模型。

用法:
    1. 复制此模板
    2. 修改类名和实现
    3. 在 __init__.py 中注册模型
"""

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class CustomRankModel(ContextRecommender):
    """
    自定义排序/CTR 模型模板

    继承 ContextRecommender 用于包含上下文特征的推荐场景
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(CustomRankModel, self).__init__(config, dataset)

        # 从 config 读取超参数
        self.embedding_size = config["embedding_size"]
        self.hidden_sizes = config.get("mlp_hidden_size", [256, 128, 64])
        self.dropout = config.get("dropout_prob", 0.1)

        # 特征 embedding 大小（由父类自动计算）
        # self.embedding_output_dim 包含所有特征 embedding 拼接后的维度

        # DNN 部分
        dnn_layers = []
        input_dim = self.embedding_output_dim
        for hidden_size in self.hidden_sizes:
            dnn_layers.append(nn.Linear(input_dim, hidden_size))
            dnn_layers.append(nn.BatchNorm1d(hidden_size))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_size

        self.dnn = nn.Sequential(*dnn_layers)
        self.output_layer = nn.Linear(input_dim, 1)

        # 损失函数
        self.loss = nn.BCEWithLogitsLoss()

        # 初始化
        self.apply(xavier_normal_initialization)

    def forward(self, interaction):
        """
        前向传播

        Args:
            interaction: RecBole Interaction 对象

        Returns:
            预测分数
        """
        # 获取所有特征的 embedding（父类方法）
        embed_output = self.concat_embed_input_fields(interaction)
        # embed_output shape: (batch_size, embedding_output_dim)

        # DNN
        dnn_output = self.dnn(embed_output)
        score = self.output_layer(dnn_output).squeeze(-1)

        return score

    def calculate_loss(self, interaction):
        """
        计算损失
        """
        label = interaction[self.LABEL]
        score = self.forward(interaction)
        loss = self.loss(score, label.float())
        return loss

    def predict(self, interaction):
        """
        预测（返回概率）
        """
        score = self.forward(interaction)
        return torch.sigmoid(score)
