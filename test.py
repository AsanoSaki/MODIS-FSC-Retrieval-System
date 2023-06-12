import os
import numpy as np
import pandas as pd
import rasterio
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
import tqdm
from train_nn import MLP

def read_val_image(img_path):
    img_data = rasterio.open(img_path).read()
    band_num, height_num, width_num = np.shape(img_data)

    img_data_list, row_col = [], []
    for i in tqdm.trange(height_num):
        for j in range(width_num):
            temp = img_data[::, i, j]
            if np.array(np.isnan(temp), dtype=np.int8).sum() > 0:
                continue
            else:
                img_data_list.append(temp.tolist())
                row_col.append([i, j])

    img_arr = np.array(img_data_list)
    labels = img_arr[:, 0]
    feature_data = img_arr[:, 1:]
    feature_name = ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'SR6', 'SR7', 'NDVI', 'NDSI', 'NDFSI',
                    'SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth',
                    'Dem', 'Slope', 'Aspect', 'LST', 'A2T', 'SC', 'LCT']
    feature_data = pd.DataFrame(feature_data, columns=feature_name)
    feature_data.drop(columns=['NDVI', 'NDSI', 'NDFSI', 'SC'], inplace=True)
    rows_cols = np.array(row_col)
    print(os.path.basename(img_path), 'Val读取成功!')

    return feature_data, labels, rows_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
lr, num_epochs = 0.01, 200

net = MLP()
net.to(device)
net.load_state_dict(torch.load('./save/train_all.params'))

loss_function = nn.MSELoss(reduction='mean')
loss_function.to(device)

net.eval()

val_path = './IMGValidation'
out_path = './IMGPred'
val_path_list = os.listdir(val_path)  # [131035_20151220.tif, ...]

predictions = []
for i in range(0, len(val_path_list)):
    img_path = os.path.join(val_path, val_path_list[i])
    out_name = val_path_list[i].split('.')[0] + "_pred.tif"
    path = os.path.join(out_path, out_name)
    # print(out_name)

    val_data, val_label, rcs = read_val_image(img_path)
    val_data_tensor = torch.tensor(np.array(val_data), dtype=torch.float32).to(device)
    val_label_tensor = torch.tensor(np.array(val_label), dtype=torch.float32).reshape(-1, 1).to(device)
    print(val_data_tensor.shape, val_label_tensor.shape)

    with torch.no_grad():
        label_hat = net(val_data_tensor)
    loss = loss_function(label_hat, val_label_tensor)
    print(f'[ Predict | {i + 1:03d}/{len(val_path_list):03d} ] loss = {loss:.5f}')

    # export_pred_img(vl_path=vdp, rows_cols=rcs, val_pred=vpd, out=out_name)

    print(val_label.sum(), label_hat.sum())
    ss = [val_label.sum(), label_hat.sum().cpu().numpy(), loss.item()]
    print(ss)
    predictions.append(ss)

    df = pd.DataFrame(np.array(predictions))
    df.to_excel('./result/MLP.xlsx', float_format='%.6f', index=0)
