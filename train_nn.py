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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'    # 只显示 Error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
# lr, num_epochs = 0.000001, 200
# lr, num_epochs = 0.0005, 200
lr, num_epochs = 0.005, 100

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
    # feature_data.drop(columns=['SensorZenith',
    #                            'SolarZenith', 'Slope', 'Aspect', 'A2T'], inplace=True)
    label_data = data_sets['FSC']

    return feature_data, label_data

def load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(21, 32), nn.BatchNorm1d(32), nn.LeakyReLU(),
                                 # nn.Linear(32, 32), nn.Dropout(0.1), nn.LeakyReLU(),
                                 nn.Linear(32, 64), nn.BatchNorm1d(64), nn.LeakyReLU(),
                                 nn.Linear(64, 128), nn.BatchNorm1d(128), nn.LeakyReLU(),
                                 # nn.Linear(64, 64), nn.Dropout(0.2), nn.LeakyReLU(),
                                 nn.Linear(128, 32), nn.Dropout(0.2), nn.LeakyReLU(),
                                 nn.Linear(32, 16), nn.BatchNorm1d(16), nn.LeakyReLU(),
                                 nn.Linear(16, 1))

    def forward(self, X):
        return self.net(X)

net = MLP()

def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    print('training on', device)
    net.to(device)
    loss_function = nn.MSELoss(reduction='mean')
    loss_function.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)

    writer = SummaryWriter('./logs/DL_no_feature_select_train_log')

    best_loss = 1e8
    for epoch in range(num_epochs):
        # writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch + 1)

        net.train()
        train_loss = []
        for feature, label in tqdm(train_iter):
            feature, label = feature.to(device), label.to(device)
            label_hat = net(feature)

            loss = loss_function(label_hat, label)
            # RMSELoss = torch.sqrt(loss_function(label_hat, label))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)
        print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}")

        net.eval()
        valid_loss = []
        with torch.no_grad():
            for feature, label in tqdm(test_iter):
                feature, label = feature.to(device), label.to(device)
                label_hat = net(feature)
                loss = loss_function(label_hat, label)
                # RMSELoss = torch.sqrt(loss_function(label_hat, label))
                valid_loss.append(loss.item())

        valid_loss = sum(valid_loss) / len(valid_loss)
        print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}")

        writer.add_scalars('loss', {'train': train_loss,
                                    'valid': valid_loss}, epoch + 1)

        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     torch.save(net.state_dict(), './save/DL_no_feature_select.params')
        #     print('saving model with loss {:.3f}'.format(best_loss))

    writer.close()

if __name__ == "__main__":

    val_dp = './IMGValidation'
    out_p = './IMGPred'

    train_feature, train_label = load_data('./Data/train_data.csv')
    test_feature, test_label = load_data('./Data/test_data.csv')
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

    train(net, train_loader, test_loader, num_epochs, lr, device)

    # df = pd.DataFrame({
    #     'id': list(range(1, len(os.listdir(self.root_dir)) + 1)),
    #     'label': np.zeros(len(os.listdir(self.root_dir))),
    #     'encoded': np.zeros(len(os.listdir(self.root_dir)))
    # })

    exit(0)
