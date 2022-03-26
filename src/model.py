import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import Accuracy, F1Score


class RNNModel(pl.LightningModule):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout=0,
                 num_classes=5,

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
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                       self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0),
                       self.hidden_size))  # internal state

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


class CNNModel(pl.LightningModule):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.c1 = nn.Conv1d(1, 8, 5)
        self.c2 = nn.Conv1d(8, 16, 5)
        self.c3 = nn.Conv1d(16, 32, 5)

        self.fc1 = nn.Linear(608, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.maxP = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.flatten0 = nn.Flatten(0)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.maxP(F.relu(self.c1(x)))
        x = self.maxP(F.relu(self.c2(x)))
        x = self.maxP(F.relu(self.c3(x)))

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.flatten0(x)
        x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.binary_cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.binary_cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)
