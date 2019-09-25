import torch.nn as nn
import torch.nn.functional as F
from wide import WideModel
from deep import DeepModel


class WideDeep(nn.Module):
    def __init__(self, wide_model_params, deep_model_params, activation):
        """
        init parameters of wide deep model
        :param wide_model_params: dict parameters for set wide model
        :param deep_model_params: dict parameters for set deep model
        :param activation: activation function for model
        """
        super(WideDeep, self).__init__()
        self.activation = activation

        # wide model parameters
        wide_input_dim = wide_model_params['wide_input_dim']
        wide_output_dim = wide_model_params['wide_output_dim']
        self.wide = WideModel(wide_input_dim, wide_output_dim)

        # deep model parameters
        deep_columns_idx = deep_model_params['deep_columns_idx']
        embedding_columns_dict = deep_model_params['embedding_columns_dict']
        hidden_layers = deep_model_params['hidden_layers']
        dropouts = deep_model_params['dropouts']
        deep_output_dim = deep_model_params['deep_output_dim']
        self.deep = DeepModel(deep_columns_idx=deep_columns_idx,
                              embedding_columns_dict=embedding_columns_dict,
                              hidden_layers=hidden_layers,
                              dropouts=dropouts,
                              output_dim=deep_output_dim)

    def forward(self, x):
        """
        input and forward
        :param x: tuple(wide_model_data, deep_model_data, target)
        :return:
        """
        # wide model
        wide_data = x[0]
        wide_out = self.wide(wide_data.float())

        # deep model
        deep_data = x[1]
        deep_out = self.deep(deep_data)

        assert wide_out.size() == deep_out.size()
        wide_deep = wide_out.add(deep_out)
        if not self.activation:
            return wide_deep
        elif self.activation == F.softmax:
            out = self.activation(wide_deep, dim=1)
        else:
            out = self.activation(wide_deep)
        return out

