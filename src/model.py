from re import X
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import Accuracy, F1Score, AUROC, AUC, PrecisionRecallCurve

class CNNBaseline(pl.LightningModule):
    def __init__(self, num_classes, lr):
        super(CNNBaseline, self).__init__()

        self.lr = lr

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(32, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        out = F.softmax(y_hat, dim=1)
        acc = self.train_acc(out, y)
        self.log("train_loss", loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        out = F.softmax(y_hat, dim=1)
        acc = self.val_acc(out, y)
        self.log("val_loss", val_loss)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        out = F.softmax(y_hat, dim=1)
        acc = self.test_acc(out, y)
        self.log("test_loss", test_loss)
        self.log("test_acc", acc)


class RNNModel(pl.LightningModule):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout=0,
                 num_classes=5,
                 lr=1e-3
                 ):
        super(RNNModel, self).__init__()

        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr

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
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        out = F.softmax(y_hat, dim=1)
        acc = self.train_acc(out, y)
        self.log("train_loss", loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        out = F.softmax(y_hat, dim=1)
        acc = self.valid_acc(out, y)
        self.log("val_loss", val_loss)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        out = F.softmax(y_hat, dim=1)
        acc = self.test_acc(out, y)
        self.log("test_loss", test_loss)
        self.log("test_acc", acc)


class CNNModel(pl.LightningModule):
    def __init__(self, nr_classes = 1, lr = 1e-3, dropout=0.1):
        super(CNNModel, self).__init__()

        self.lr = lr
        self.nr_classes = nr_classes
        self.dropout = dropout

        self.c1 = nn.Conv1d(1, 8, 5)
        self.c2 = nn.Conv1d(8, 16, 5)
        self.c3 = nn.Conv1d(16, 32, 5)

        self.fc1 = nn.Linear(608, 187)
        self.fc2 = nn.Linear(187, 64)
        self.fc3 = nn.Linear(64, self.nr_classes)

        self.maxP = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(self.dropout)
        self.flatten = nn.Flatten()
        self.flatten0 = nn.Flatten(0)


        self.test_acc  = Accuracy()
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        self.test_f1 = F1Score()
        self.train_f1 = F1Score()
        self.valid_f1 = F1Score()

        self.test_ROC = AUROC(pos_label=1)
        self.test_PRC = PrecisionRecallCurve(pos_label=1)
        self.test_AUC = AUC()
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.maxP(F.relu(self.c1(self.dropout(x))))
        x = self.maxP(F.relu(self.c2(self.dropout(x))))
        x = self.maxP(F.relu(self.c3(self.dropout(x))))

        x = self.flatten(x)

        x = self.fc1(F.relu(self.dropout(x)))
        x = self.fc2(F.relu(self.dropout(x)))
        x = self.fc3(F.relu(self.dropout(x)))

        if self.nr_classes == 1:   
            x = self.flatten0(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.nr_classes==1:
            y_hat = self(x.to(torch.float32))
            y_hat = torch.sigmoid(y_hat)
            loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        else:
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            y_hat = F.softmax(y_hat, dim=1)

        self.train_acc(y_hat, y)
        self.train_f1(y_hat, y)
        self.log("train_loss", loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.nr_classes==1:
            y_hat = self(x.to(torch.float32))
            y_hat = torch.sigmoid(y_hat)
            loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        else:
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            y_hat = F.softmax(y_hat, dim=1)

        self.valid_acc(y_hat, y)
        self.log("valid_loss", loss)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)


    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.nr_classes==1:
            y_hat = self(x.to(torch.float32))
            y_hat = torch.sigmoid(y_hat)
            loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
            self.test_ROC.update(y_hat, y)
            self.test_PRC.update(y_hat, y)
        else:
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            y_hat = F.softmax(y_hat, dim=1)

        acc = self.test_acc(y_hat, y.to(torch.int8))
        f1 = self.test_f1(y_hat, y.to(torch.int8))
        self.log("test_loss", loss)
        self.log("test_f1", f1)
        self.log("test_acc", acc)

    def test_step_end(self, test_step_outputs):
        if(self.nr_classes == 1):
            test_ROC = self.test_ROC.compute()
            self.log("test roc", test_ROC)

            precision, recall, threshold = self.test_PRC.compute()
            test_PRC = self.test_AUC(recall, precision)
            self.log("test prc", test_PRC)

class CNNResidual(pl.LightningModule):
    def __init__(self, nr_classes = 1, lr = 1e-3, dropout= 0.1):
        super(CNNResidual, self).__init__()

        self.lr = lr
        self.nr_classes = nr_classes
        self.dropout = dropout

        self.c1 = nn.Conv1d(1, 8, 5)
        self.c2 = nn.Conv1d(8, 15, 5)
        self.c3 = nn.Conv1d(16, 32, 5)

        self.fc1 = nn.Linear(608, 187)
        self.fc2 = nn.Linear(187, 32)
        self.fc3 = nn.Linear(32, self.nr_classes)

        self.fc4 = nn.Linear(187, 43)

        self.maxP = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(self.dropout)
        self.flatten = nn.Flatten()
        self.flatten0 = nn.Flatten(0)

        self.test_acc  = Accuracy()
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        self.test_f1 = F1Score()
        self.train_f1 = F1Score()
        self.valid_f1 = F1Score()

        self.test_ROC = AUROC(pos_label=1)
        self.test_PRC = PrecisionRecallCurve(pos_label=1)
        self.test_AUC = AUC()

    def forward(self, x):
        r1 = self.fc4(x)
        r1 = r1.unsqueeze(1)

        r2 = torch.div(x, self.nr_classes + 1)
        
        x = x.unsqueeze(1)

        x = self.maxP(F.relu(self.c1(self.dropout(x))))
        x = self.maxP(F.relu(self.c2(self.dropout(x))))
        x = torch.cat((x, r1), dim=1)
        x = self.maxP(F.relu(self.c3(self.dropout(x))))

        x = self.flatten(x)

        x = self.fc1(F.relu(self.dropout(x)))
        x = self.fc2(F.relu(self.dropout(x)) + r2)
        x = self.fc3(F.relu(self.dropout(x)))

        if self.nr_classes == 1:   
            x = self.flatten0(x)

        x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.nr_classes==1:
            y_hat = self(x.to(torch.float32))
            y_hat = torch.sigmoid(y_hat)
            loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        else:
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            y_hat = F.softmax(y_hat, dim=1)

        self.train_acc(y_hat, y)
        self.train_f1(y_hat, y)
        self.log("train_loss", loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.nr_classes==1:
            y_hat = self(x.to(torch.float32))
            y_hat = torch.sigmoid(y_hat)
            loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        else:
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            y_hat = F.softmax(y_hat, dim=1)

        self.valid_acc(y_hat, y)
        self.log("valid_loss", loss)
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)


    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.nr_classes==1:
            y_hat = self(x.to(torch.float32))
            y_hat = torch.sigmoid(y_hat)
            loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
            self.test_ROC.update(y_hat, y)
            self.test_PRC.update(y_hat, y)
        else:
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            y_hat = F.softmax(y_hat, dim=1)

        acc = self.test_acc(y_hat, y.to(torch.int8))
        f1 = self.test_f1(y_hat, y.to(torch.int8))
        self.log("test_loss", loss)
        self.log("test_f1", f1)
        self.log("test_acc", acc)

    def test_step_end(self, test_step_outputs):
        if(self.nr_classes == 1):
            test_ROC = self.test_ROC.compute()
            self.log("test roc", test_ROC)

            precision, recall, threshold = self.test_PRC.compute()
            test_PRC = self.test_AUC(recall, precision)
            self.log("test prc", test_PRC)