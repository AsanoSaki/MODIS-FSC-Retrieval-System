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
import tqdm
import rioxarray

# --------------- Hyper Parameter ---------------
batch_size = 4
lr = 0.0001
num_epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------- Dataset ---------------
class ModisDataset(Dataset):
    def __init__(self, root_dir, csv_file=None, transform=None):
        # images directory
        self.root_dir = root_dir
        # transformations if any
        self.transform = transform

        if csv_file:
            # read csv file
            self.df = pd.read_csv(csv_file)

            # {class: num}
            self.class_to_num = {}
            for i, label in enumerate(self.df['label'].unique()):
                self.class_to_num[label] = i

            # {num: class}
            self.num_to_class = {v : k for k, v in self.class_to_num.items()}

            # add a column with decoded labels
            self.df['encoded'] = self.df['label'].map(self.class_to_num)

        else:
            # if no csv file provided -> create a df, fill labels with 0's
            self.df = pd.DataFrame({
                'dir': os.listdir(self.root_dir),
                'label': np.zeros(len(os.listdir(self.root_dir))),
            })

    def __len__(self):
        # num of records in the df
        return len(self.df)

    def __getitem__(self, index):
        # path to the image
        img_path = os.path.join(self.root_dir, str(self.df.iloc[index, 0]))
        # read image
        image = rioxarray.open_rasterio(img_path)
        image = torch.tensor(np.array(image.values), dtype=torch.float32)
        image = torch.where(torch.isnan(image), torch.full_like(image, 0), image)
        # get label
        y_label = image.sum([1, 2])[0]

        # apply transformations
        if self.transform:
            image = self.transform(image)

        return (image, y_label)

trans = transforms.Compose([transforms.Resize((448, 448))])

dataset = ModisDataset(root_dir='./IMGTrain', transform=trans)
# print(len(dataset))  # 53

# --------------- Split Dataset ---------------
indexes = list(range(len(dataset)))

train_indexes, val_indexes = train_test_split(indexes, test_size=0.3, random_state=1)

train_dataset = Subset(dataset, train_indexes)
val_dataset = Subset(dataset, val_indexes)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# data_iter = iter(train_loader)
# images, labels = next(data_iter)
#
# print(images.shape, labels.shape)  # torch.Size([37, 22, 448, 448]) torch.Size([37])

# --------------- ResNet34 ---------------
# load a pretrained model
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(22, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

# --------------- Train ---------------
def train(net, train_iter, test_iter, num_epochs, lr, device):
    print('training on', device)
    net.to(device)
    loss_function = nn.MSELoss()
    loss_function.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # writer = SummaryWriter('./logs/ResNet_train_log')

    for epoch in range(num_epochs):
        net.train()
        train_loss = []
        for img, label in tqdm.tqdm(train_iter):
            img, label = img.to(device), label.to(device).reshape(-1, 1)
            label_hat = net(img)

            loss = loss_function(label_hat, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)
        print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}")

        net.eval()
        valid_loss = []
        with torch.no_grad():
            for img, label in tqdm.tqdm(test_iter):
                img, label = img.to(device), label.to(device).reshape(-1, 1)
                label_hat = net(img)
                loss = loss_function(label_hat, label)
                valid_loss.append(loss.item())

        valid_loss = sum(valid_loss) / len(valid_loss)
        print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}")

        # writer.add_scalars('loss', {'train': train_loss,
        #                             'valid': valid_loss}, epoch + 1)

    # writer.close()

train(net, train_loader, val_loader, num_epochs, lr, device)

