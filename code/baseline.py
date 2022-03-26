import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

from torchmetrics import Accuracy, F1Score

import pytorch_lightning as pl
from pytorch_lightning import Trainer

class TimeSeriesDataset(Dataset):
    def __init__(self, test=False):
        train_file = "data/mitbih_train.csv"
        test_file = "data/mitbih_test.csv"
        if test:
            self.df = pd.read_csv(test_file, header=None)
        else:
            self.df = pd.read_csv(train_file, header=None)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X = self.df.loc[idx, list(range(187))].astype(np.float32)
        y = self.df.loc[idx, 187].astype(np.int8)
        return torch.tensor(X), torch.tensor(y, dtype=torch.long)
        

class MITDataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 64, train_split=0.8, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.train_split = train_split
        self.num_workers = num_workers
        
    
    def setup(self, stage=None):
        timeseries_full = TimeSeriesDataset()
        train_length = int(self.train_split*len(timeseries_full))
        val_length = len(timeseries_full) - train_length
        self.train_set, self.val_set = random_split(timeseries_full, [train_length, val_length])
        self.test_set = TimeSeriesDataset(test=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)  

class RNNModel(pl.LightningModule):
    def __init__(self, 
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout = 0,
                 num_classes = 5,
                 
    ):
        super(RNNModel, self).__init__()
        
        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        self.train_f1 = F1Score()
        self.val_f1 = F1Score()


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state

        x = x.unsqueeze(2)
        out, (hn, cn) = self.lstm(x, (h_0, c_0))
        x = self.linear(out[:, -1, :])
        x = F.softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.train_f1(y_hat, y)
        self.log("train_loss", loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.valid_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.log("val_loss", val_loss)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)

if __name__ == '__main__':
    pl.seed_everything(1234)

    model = RNNModel(1, 64, 1)
    trainer = Trainer()

    mit = MITDataModule()
    trainer.fit(model, datamodule=mit)   