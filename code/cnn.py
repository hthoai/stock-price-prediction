import warnings
warnings.filterwarnings('ignore')

import os
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

root_dir = '/root/hoai_workspace/stock-price-prediction/'

# STOCK CHART DATASET
class StockChartDataset(object):
    def __init__(self, dir_path, transforms):
        self.dir_path = dir_path
        self.transforms = transforms
        df = pd.read_csv(dir_path + 'target.csv')
        self.imgs = df.filename.tolist()
        self.log_target = df.target.tolist()

    def __getitem__(self, idx):
        # Load images
        img_path = os.path.join(self.dir_path, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        
        target = torch.tensor([self.log_target[idx]])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


# STOCK CHART DATA MODULE
class StockChartDataModule(pl.LightningDataModule):
    def setup(self, stage):
        # transforms for images
        transform=transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train = StockChartDataset(root_dir + 'data/charts/train/',
                                       transforms=transform)
        self.val = StockChartDataset(root_dir + 'data/charts/val/',
                                     transforms=transform)
        self.test = StockChartDataset(root_dir + 'data/charts/test/',
                                      transforms=transform)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64)


# RESIDUAL BLOCK
class ResidualBlock(nn.Module):
    def __init__(self, num_channels, output_channels, stride1, stride2, stride3, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.cond = any([stride1 != 1, stride2 != 1, stride3 != 1])
        self.conv1 = nn.Conv2d(num_channels, num_channels, padding=1, 
                            kernel_size=3, stride=stride1)
        self.batch_norm = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, padding=1, 
                            kernel_size=3, stride=stride2)
        if self.cond:
            self.conv = nn.Conv2d(num_channels, num_channels, padding=0,
                                kernel_size=1, stride=max(stride1, stride2, stride3))
        # Last convolutional layer to reduce output block shape.
        self.conv3 = nn.Conv2d(num_channels, output_channels, padding=0, 
                            kernel_size=1, stride=stride3)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, X):
        if self.cond:
            Y = self.conv(X)
        else:
            Y = X
        X = self.conv1(X)
        X = self.batch_norm(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.batch_norm(X)
        X = self.relu(X+Y)
        X = self.conv3(X)
        return X


# STOCK CHART CNN MODEL
class StockChartCNN(pl.LightningModule):
    def __init__(self, output_shape=1):
        super(StockChartCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.batch_norm = nn.BatchNorm2d(32)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.res_conv1 = ResidualBlock(
            num_channels=32, output_channels=128,
            stride1=1, stride2=1, stride3=1)
        self.res_conv2 = ResidualBlock(
            num_channels=128, output_channels=256,
            stride1=2, stride2=1, stride3=1)
        self.res_conv3 = ResidualBlock(
            num_channels=256, output_channels=512,
            stride1=2, stride2=1, stride3=1)
        self.average_pool = nn.AvgPool2d(kernel_size=7, padding=0)
        # self.layer_norm = nn.LayerNorm([512, 1, 1])
        self.fc1 = nn.Linear(in_features=512, out_features=500)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=500, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=25)
        self.out = nn.Linear(in_features=25, out_features=output_shape)
        
    def forward(self, X):
        X = self.conv(X)
        X = self.batch_norm(X)
        X = self.relu(X)
        X = self.max_pool(X)
        X = self.res_conv1(X)
        X = self.res_conv2(X)
        X = self.res_conv3(X)
        X = self.average_pool(X)
        # X = self.layer_norm(X)
        X = X.view(X.size(0), -1)
        X = self.fc1(X)
        X = self.dropout(X)
        X = self.fc2(X)
        X = self.dropout(X)
        X = self.fc3(X)
        X = self.dropout(X)
        X = self.out(X)
        return X
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        logits = self.forward(x)
        loss = F.mse_loss(logits, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.mse_loss(logits, y)
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, patience=2, verbose=True, min_lr=1e-5
        # )
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


# TRAINER
# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath=root_dir + 'model/cnn_1227_0',
    filename='dji-non-norm-{epoch:02d}-{val_loss:.9f}',
    save_top_k=5,
    mode='min',
)
data_module = StockChartDataModule()
model = StockChartCNN()
trainer = pl.Trainer(gpus=1, max_epochs=100,
                     callbacks=[checkpoint_callback],
                     default_root_dir=root_dir,
                     progress_bar_refresh_rate=2)
trainer.fit(model, data_module)
