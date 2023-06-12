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
from typing import Dict, Optional

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

def create_src_lengths_mask(batch_size: int, src_lengths: torch.Tensor, max_src_len: Optional[int]=None):
    """
    Generate boolean mask to prevent attention beyond the end of source
    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths
      max_src_len: Optionally override max_src_len for the mask
    Outputs:
      [batch_size, max_src_len]
    """
    if max_src_len is None:
        max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()

def masked_softmax(scores, src_lengths, src_length_masking=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
    if src_length_masking:
        bsz, max_src_len = scores.size()
        # compute masks
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        # Fill pad positions with -inf
        scores = scores.masked_fill(src_mask == 0, -np.inf)

    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)

# s(x, q) = v.T * tanh (W * x + b)
class MLPAttentionNetwork(nn.Module):
    def __init__(self, hidden_dim, attention_dim, src_length_masking=True):
        super(MLPAttentionNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.src_length_masking = src_length_masking

        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, x, x_lengths):
        """
        :param x: seq_len * batch_size * hidden_dim
        :param x_lengths: batch_size
        :return: batch_size * seq_len, batch_size * hidden_dim
        """
        seq_len, batch_size, _ = x.size()
        # (seq_len * batch_size, hidden_dim)
        flat_inputs = x.view(-1, self.hidden_dim)
        # (seq_len * batch_size, attention_dim)
        mlp_x = self.proj_w(flat_inputs)
        # (batch_size, seq_len)
        att_scores = self.proj_v(mlp_x).view(seq_len, batch_size).t()
        # (seq_len, batch_size)
        normalized_masked_att_scores = masked_softmax(att_scores, x_lengths, self.src_length_masking).t()
        # (batch_size, hidden_dim)
        attn_x = (x * normalized_masked_att_scores.unsqueeze(2)).sum(0).mean(1)

        return normalized_masked_att_scores.t(), attn_x

net = MLPAttentionNetwork(21, 32)
# x = torch.rand((21, 3, 21))
# x_lengths = torch.LongTensor([21])
# att_scores, attn_x = net(x, x_lengths)
# print(att_scores)
# print(attn_x)
# exit(0)

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

    # writer = SummaryWriter('./logs/DL_no_feature_select_train_log')

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

        # writer.add_scalars('loss', {'train': train_loss,
        #                             'valid': valid_loss}, epoch + 1)

        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     torch.save(net.state_dict(), './save/DL_no_feature_select.params')
        #     print('saving model with loss {:.3f}'.format(best_loss))

    # writer.close()

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
