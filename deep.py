# deep model for continuous features and some sparse categorical features
# author: WenYi
# time: 2019-09-24
import torch
import torch.nn as nn
from utils import linear


class DeepModel(nn.Module):
    def __init__(self, deep_columns_idx, embedding_columns_dict, hidden_layers, dropouts, output_dim):
        """
        init parameters
        :param deep_columns_idx: dict include column name and it's index
            e.g. {'age': 0, 'career': 1,...}
        :param embedding_columns_dict: dict include categories columns name and number of unique val and embedding dimension
            e.g. {'age':(10, 32),...}
        :param hidden_layers: number of hidden layers
        :param deep_columns_idx: dict of columns name and columns index
        :param dropouts: list of float each hidden layers dropout len(dropouts) == hidden_layers - 1
        """
        super(DeepModel, self).__init__()
        self.embedding_columns_dict = embedding_columns_dict
        self.deep_columns_idx = deep_columns_idx
        for key, val in embedding_columns_dict.items():
            setattr(self, 'dense_col_'+key, nn.Embedding(val[0], val[1]))
        embedding_layer = 0
        for col in self.deep_columns_idx.keys():
            if col in embedding_columns_dict:
                embedding_layer += embedding_columns_dict[col][1]
            else:
                embedding_layer += 1
        self.layers = nn.Sequential()
        hidden_layers = [embedding_layer] + hidden_layers
        dropouts = [0.0] + dropouts
        for i in range(1, len(hidden_layers)):
            self.layers.add_module(
                'hidden_layer_{}'.format(i-1),
                linear(hidden_layers[i-1], hidden_layers[i], dropouts[i-1])
            )
        self.layers.add_module('last_linear', nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x):
        emb = []
        continuous_cols = [col for col in self.deep_columns_idx.keys() if col not in self.embedding_columns_dict]
        for col, _ in self.embedding_columns_dict.items():
            if col not in self.deep_columns_idx:
                raise ValueError("ERROR column name may be your deep_columns_idx dict is not math the"
                                 "embedding_columns_dict")
            else:
                idx = self.deep_columns_idx[col]
                emb.append(getattr(self, 'dense_col_'+col)(x[:, idx].long()))

        for col in continuous_cols:
            idx = self.deep_columns_idx[col]
            emb.append(x[:, idx].view(-1, 1).float())
        embedding_layers = torch.cat(emb, dim=1)
        out = self.layers(embedding_layers)
        return out