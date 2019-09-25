# predict for new input data,input data x(wide_data, deep_data)
# notice: the dimension of input data must equal to the train data
from utils import load_model
from WideDeep import WideDeep
from prepare_data import read_data, feature_engine
import torch
from utils import set_method
import argparse

parse = argparse.ArgumentParser(description="wide deep model include arguments")
parse.add_argument("--hidden_layers", nargs='+', type=int, default=[64, 32, 16])
parse.add_argument("--dropouts", nargs='+', type=int, default=[0.5, 0.5])
parse.add_argument("--deep_out_dim", default=1, type=int)
parse.add_argument("--wide_out_dim", default=1, type=int)
parse.add_argument("--batch_size", default=32, type=int)
parse.add_argument("--lr", default=0.01, type=float)
parse.add_argument("--print_step", default=200, type=int)
parse.add_argument("--epochs", default=10, type=int)
parse.add_argument("--validation", default=True, type=bool)
parse.add_argument("--method", choices=['multiclass', 'binary', 'regression'], default='binary',type=str)
args = parse.parse_args()
data = read_data()
train_data, test_data, deep_columns_idx, embedding_columns_dict = feature_engine(data)
data_wide = train_data[0]

# 预测数据的输入格式，这里预测一条数据
t = (torch.from_numpy(train_data[0].values[0].reshape(-1, train_data[0].values.shape[1])),
     torch.from_numpy(train_data[1].values[0].reshape(-1, train_data[1].values.shape[1])))
print(t)

# parameters setting
deep_model_params = {
    'deep_columns_idx': deep_columns_idx,
    'embedding_columns_dict': embedding_columns_dict,
    'hidden_layers': args.hidden_layers,
    'dropouts': args.dropouts,
    'deep_output_dim': args.deep_out_dim}
wide_model_params = {
    'wide_input_dim': data_wide.shape[1],
    'wide_output_dim': args.wide_out_dim
}
activation, criterion = set_method(args.method)
widedeep = WideDeep(wide_model_params, deep_model_params, activation)
# path 为存储模型参数的位置
path = 'wide_deep_model_1569328938.2377222.pkl'
load_model(widedeep, path)
print(widedeep(t))