import os
import numpy as np
import pandas as pd
import rasterio
import tqdm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.metrics import roc_auc_score
import os
import cv2
import json
import glob
import math
from random import shuffle  # 打乱数据
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from d2l import torch as d2l

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'    # 只显示 Error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
lr, num_epochs = 0.0001, 200

def export_pred_img(vl_path, rows_cols, val_pred, out):
    """
    :param vl_path:
    :param rows_cols:
    :param val_pred:
    :param out: pred val_img
    """
    label_val_ = rasterio.open(vl_path).read(1)
    result = np.zeros_like(label_val_) + 255

    for a in range(len(rows_cols)):
        rc = rows_cols[a]
        result[rc[0], rc[1]] = val_pred[a]

    print(np.array_equal(result, label_val_))

    x_size, y_size = np.shape(result)
    with rasterio.open(out, "w",
                       driver="GTiff",
                       width=y_size,
                       height=x_size,
                       count=1,
                       crs=rasterio.open(vl_path).crs,
                       transform=rasterio.open(vl_path).transform,
                       dtype=rasterio.float32,
                       nodata=255) as dt:
        dt.write(result.astype(rasterio.float32), 1)
    dt.close()


def load_data(path):
    df = pd.read_csv(path)
    data_sets = pd.DataFrame(df, dtype=np.float32)
    feature_data = data_sets.drop(['FSC'], axis=1)
    # feature_data.drop(columns=['NDVI', 'NDSI', 'NDFSI', 'SC'], inplace=True)
    label_data = data_sets['FSC']

    return feature_data, label_data

def load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=False, drop_last=True)

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Sequential(nn.Linear(self.num_hiddens, 128),
                                        nn.Linear(128, self.vocab_size))
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = torch.permute(inputs, (1, 0, 2))
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))

num_features, num_hiddens, num_outputs = 21, 256, 1
gru_layer = nn.GRU(num_features, num_hiddens, num_layers=2)
net = RNNModel(gru_layer, num_outputs)
net = net.to(device)

def train_epoch(net, train_iter, loss_function, optimizer, device, use_random_iter):
    state = None
    train_loss = []
    train_XT, train_YT, i = [], [], 0
    for X, Y in tqdm(train_iter):

        if (i + 1) % 2 != 0:
            train_XT.append(X)
            train_YT.append(Y)
            i += 1
            continue
        X = torch.stack(train_XT, dim=1)
        y = torch.stack(train_YT, dim=1).reshape(-1, 1)
        # print(X.shape, y.shape)  # torch.Size([256, T, 21]) torch.Size([256 * T, 1])

        train_XT, train_YT, i = [], [], 0
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        # X, y = X.unsqueeze(1), Y.T.reshape(-1, 1)
        X, y = X.to(device), y.to(device)
        loss_function.to(device)
        y_hat, state = net(X, state)
        loss = loss_function(y_hat, y).mean()
        optimizer.zero_grad()
        loss.backward()
        d2l.grad_clipping(net, 1)
        optimizer.step()
        train_loss.append(loss)
    return sum(train_loss) / len(train_loss)

def train(net, train_iter, lr, num_epochs, device, use_random_iter=False):
    loss_function = nn.MSELoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr)  # lr=0.1, layers=1, loss=0.016711
    optimizer = torch.optim.Adam(net.parameters(), lr)  # lr=0.01, layers=1, loss=0.015748
    # lr=0.005, layers=2, loss=0.014813
    # lr=0.005, layers=2, linears=2. loss=0.014548

    # writer = SummaryWriter('./logs/GRU_train_log')

    for epoch in range(num_epochs):
        ppl = train_epoch(net, train_iter, loss_function, optimizer, device, use_random_iter)
        print(f'Perplexity: {ppl:.6f}')
        # if (epoch + 1) % 10 == 0:
        #     print(f'Perplexity: {ppl:.6f}')
        #     writer.add_scalar('train_loss', ppl, epoch + 1)

    # writer.close()

# def train(net, train_iter, test_iter, num_epochs, lr, device):
#     def init_weights(m):
#         if type(m) == nn.Linear:
#             nn.init.xavier_uniform_(m.weight)
#     net.apply(init_weights)
#
#     print('training on', device)
#     net.to(device)
#     loss_function = nn.MSELoss(reduction='mean')
#     loss_function.to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
#
#     writer = SummaryWriter('./logs/train_all_log')
#
#     best_loss = 1e8
#     for epoch in range(num_epochs):
#         # writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch + 1)
#
#         net.train()
#         train_loss = []
#         for feature, label in tqdm(train_iter):
#             feature, label = feature.to(device), label.to(device)
#             label_hat = net(feature)
#
#             loss = loss_function(label_hat, label)
#             # RMSELoss = torch.sqrt(loss_function(label_hat, label))
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             # scheduler.step()
#
#             train_loss.append(loss.item())
#
#         train_loss = sum(train_loss) / len(train_loss)
#         print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}")
#
#         net.eval()
#         valid_loss = []
#         with torch.no_grad():
#             for feature, label in tqdm(test_iter):
#                 feature, label = feature.to(device), label.to(device)
#                 label_hat = net(feature)
#                 loss = loss_function(label_hat, label)
#                 # RMSELoss = torch.sqrt(loss_function(label_hat, label))
#                 valid_loss.append(loss.item())
#
#         valid_loss = sum(valid_loss) / len(valid_loss)
#         print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}")
#
#         writer.add_scalars('loss', {'train': train_loss,
#                                     'valid': valid_loss}, epoch + 1)
#
#         if valid_loss < best_loss:
#             best_loss = valid_loss
#             torch.save(net.state_dict(), './save/train_all.params')
#             print('saving model with loss {:.3f}'.format(best_loss))
#
#     writer.close()

if __name__ == "__main__":

    val_dp = './IMGValidation'
    out_p = './IMGPred'

    train_feature, train_label = load_data('./Data/train_data.csv')
    test_feature, test_label = load_data('./Data/valid_data.csv')
    # print(train_feature, '\n', train_label, '\n', np.shape(train_feature), '\n', np.shape(train_label))
    # print(test_feature, '\n', test_label, '\n', np.shape(test_feature), '\n', np.shape(test_label))
    print('Data reading completed !!!')
    train_feature = torch.tensor(np.array(train_feature), dtype=torch.float32)
    train_label = torch.tensor(np.array(train_label), dtype=torch.float32).reshape(-1, 1)
    test_feature = torch.tensor(np.array(test_feature), dtype=torch.float32)
    test_label = torch.tensor(np.array(test_label), dtype=torch.float32).reshape(-1, 1)
    print(train_feature.shape, train_label.shape)
    print(test_feature.shape, test_label.shape)

    train_loader = load_array((train_feature, train_label), batch_size)
    test_loader = load_array((test_feature, test_label), batch_size)
    feature, label = next(iter(train_loader))
    print(feature.shape)  # torch.Size([256, 21])
    print(label.shape)  # torch.Size([256])
    feature = feature.unsqueeze(1)
    print(feature.shape)

    train(net, train_loader, lr, num_epochs, device)

    # df = pd.DataFrame({
    #     'id': list(range(1, len(os.listdir(self.root_dir)) + 1)),
    #     'label': np.zeros(len(os.listdir(self.root_dir))),
    #     'encoded': np.zeros(len(os.listdir(self.root_dir)))
    # })
