# wide model for categorical features and cross features
# author: WenYi
# time:  2019-09-24
import torch.nn as nn
from utils import linear


class WideModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        """
        wide model using LR
        :param input_dim: int the dimension of wide model input
        :param output_dim: int the dimension of wide model output
        """
        super(WideModel, self).__init__()
        self.linear = linear(input_dim, output_dim, dropout)

    def forward(self, x):
        out = self.linear(x)
        return out