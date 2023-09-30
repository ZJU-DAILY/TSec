import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    def __init__(self, feature_num: int, to_num: int = 2):
        """
        :param feature_num: 外层特征数量
        :param to_num: 最终得到特征数量
        """
        super(Time2Vec, self).__init__()
        self.wb_sin = nn.Linear(feature_num, feature_num - 1)
        self.wb0_sin = nn.Linear(feature_num, 1)
        self.wb_cos = nn.Linear(feature_num, feature_num - 1)
        self.wb0_cos = nn.Linear(feature_num, 1)
        self.decode = nn.Linear(feature_num, to_num)

    def forward(self, x):
        x_sin_encoder = torch.cat([torch.sin(self.wb_sin(x)), self.wb0_sin(x)], -1)
        x_cos_encoder = torch.cat([torch.cos(self.wb_cos(x)), self.wb0_cos(x)], -1)
        x = x_sin_encoder + x_cos_encoder
        out = self.decode(x)
        return out
