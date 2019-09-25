# train wide deep model
# author: WenYi
# time: 2019-09-24
import torch
from prepare_data import read_data, feature_engine
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from WideDeep import WideDeep
import numpy as np
from utils import set_method, to_device, save_model
import argparse
import time
import warnings
warnings.filterwarnings("ignore")


class trainset(Dataset):
    def __init__(self, data):
        self.wide_data = data[0]
        self.deep_data = data[1]
        self.target = data[2]

    def __getitem__(self, index):
        wide_data = self.wide_data[index]
        deep_data = self.deep_data[index]
        target = self.target[index]
        data = (wide_data, deep_data, target)
        return data

    def __len__(self):
        return len(self.target)


def valid_epoch(model, valid_loader, criterion, device):
    model.eval()
    losses = []
    targets = []
    outs = []
    for idx, (data_wide, data_deep, target) in enumerate(valid_loader):
        data_wide, data_deep, target = data_wide.to(device), data_deep.to(device), target.to(device)
        x = (data_wide, data_deep)
        out = model(x)
        loss = criterion(out, target.float())
        losses.append(loss.item())
        targets += list(target.numpy())
        out = out.view(-1).detach().numpy()
        outs += list(np.int64(out > 0.5))
    met = accuracy_score(targets, outs)
    return met, sum(losses) / len(losses)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, print_step):
    model.train()
    for idx, (data_wide, data_deep, target) in enumerate(train_loader):
        data_wide, data_deep, target = data_wide.to(device), data_deep.to(device), target.to(device)
        x = (data_wide, data_deep)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target.float())
        loss.backward()
        optimizer.step()

        if (idx + 1) % print_step == 0:
            print("Epoch %d iteration %d loss is %.4f" % (epoch+1, idx+1, loss.item()))
        if idx == len(train_loader):
            break


def train(model, train_loader, test_loader, optimizers, criterion, device, epochs,  print_step, validation=True):
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizers, criterion, device, epoch, print_step)

        if validation:
            met, loss = valid_epoch(model, test_loader, criterion, device)
            print("Epoch %d validation loss is %.4f and validation metrics is %.4f" % (epoch + 1, loss, met))


def main(args):
    data = read_data()
    train_data, test_data, deep_columns_idx, embedding_columns_dict = feature_engine(data)
    data_wide = train_data[0]
    train_data = (torch.from_numpy(train_data[0].values), torch.from_numpy(train_data[1].values),
                  torch.from_numpy(train_data[2].values))
    train_data = trainset(train_data)
    test_data = (torch.from_numpy(test_data[0].values), torch.from_numpy(test_data[1].values),
                 torch.from_numpy(test_data[2].values))
    test_data = trainset(test_data)
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    device = to_device()
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
    widedeep = widedeep.to(device)
    optimizer = torch.optim.Adam(widedeep.parameters(), lr=args.lr)
    train(widedeep, trainloader, testloader, optimizer, criterion, device, epochs=args.epochs,
          print_step=args.print_step, validation=args.validation)
    save_model(widedeep, "wide_deep_model_{}.pkl".format(time.time()))


if __name__ == "__main__":
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
    main(args)
