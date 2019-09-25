# some tools for wide deep model
# author: WenYi
# create: 2019-09-23
import torch.nn as nn
import torch
import torch.nn.functional as F


def linear(inp, out, dropout):
    """
    linear model module by nn.sequential
    :param inp: int, linear model input dimensio
    :param out: int, linear model output dimension
    :param dropout: float dropout probability for linear layer
    :return: tensor
    """
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LeakyReLU(),
        nn.Dropout(dropout)
    )


def set_method(method):
    if method == 'regression':
        return None, F.mse_loss
    if method == 'binary':
        return torch.sigmoid, F.binary_cross_entropy
    if method == 'multiclass':
        return F.softmax, F.cross_entropy


# save model的问题，保存整个模型及参数，如果模型很大加载的时候会很慢，不适用与在线预测，在线预测一般是只将模型参数保存及model.state_dict
# 如果实现的是模型的继续训练，则需要同时保存优化器和当前训练的epoch，不保存epoch的话每次训练依然是从第epoch=0开始训练
# 在线预测的时候需要现定义好网络结构，然后load网络参数进行预测即可
def save_model(model, path):
    torch.save(model.state_dict(), path)


# 参数model是事先定义好的网络模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))


def to_device():
    """
    user gpu or cpu
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")