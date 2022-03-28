from re import X
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import Accuracy, AUROC, AUC, PrecisionRecallCurve

OPTIMIZER = {
    "adam": Adam
}

CRITERION = {
    "cross_entropy": nn.CrossEntropyLoss,
    "binary_cross_entropy": nn.BCEWithLogitsLoss
}

class CNNBaseline(pl.LightningModule):
    def __init__(self, num_classes, lr, optimizer, criterion):
        super(CNNBaseline, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.num_classes = num_classes
        self.optimizer = OPTIMIZER[optimizer]
        self.criterion = CRITERION[criterion]()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.test_AUROC = AUROC(pos_label=1)
        self.test_PRC = PrecisionRecallCurve(pos_label=1)
        self.test_AUC = AUC()

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
        logits = self.model(x)
        return logits

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
            loss = self.criterion(y_hat, y.float())
            y_pred = torch.sigmoid(y_hat)
        else:
            loss = self.criterion(y_hat, y)
            y_pred = torch.softmax(y_hat, dim=1)
        acc = self.train_acc(y_pred, y)
        self.log("train_loss", loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
            val_loss = self.criterion(y_hat, y.float())
            y_pred = torch.sigmoid(y_hat)
        else:
            val_loss = self.criterion(y_hat, y)
            y_pred = torch.softmax(y_hat, dim=1)
        acc = self.val_acc(y_pred, y)
        self.log("val_loss", val_loss)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
            test_loss = self.criterion(y_hat, y.float())
            y_pred = torch.sigmoid(y_hat)
            self.test_AUROC(y_pred, y)
            self.test_PRC(y_pred, y)
        else:
            test_loss = self.criterion(y_hat, y)
            y_pred = torch.softmax(y_hat, dim=1)
        acc = self.test_acc(y_pred, y)
        self.log("test_loss", test_loss)
        self.log("test_acc", acc)

    def test_step_end(self, test_step_outputs):
        if(self.num_classes == 1):
            test_ROC = self.test_AUROC.compute()
            self.log("test roc", test_ROC)

            precision, recall, threshold = self.test_PRC.compute()
            self.log("test_precision", precision)
            self.log("test_recall", recall)
            self.log("test_threshold", threshold)
            test_PRC = self.test_AUC(recall, precision)
            self.log("test prc", test_PRC)


class RNNModel(pl.LightningModule):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 optimizer,
                 criterion,
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
        self.optimizer = OPTIMIZER[optimizer]
        self.loss = CRITERION[criterion]()

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
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        out = F.softmax(y_hat, dim=1)
        acc = self.train_acc(out, y)
        self.log("train_loss", loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        out = F.softmax(y_hat, dim=1)
        acc = self.valid_acc(out, y)
        self.log("val_loss", val_loss)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss(y_hat, y)
        out = F.softmax(y_hat, dim=1)
        acc = self.test_acc(out, y)
        self.log("test_loss", test_loss)
        self.log("test_acc", acc)


class CNNModel(pl.LightningModule):
    def __init__(self, criterion, optimizer, num_classes, lr = 1e-3, dropout=0.1):
        super(CNNModel, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.num_classes = num_classes
        self.dropout = dropout        
        self.optimizer = OPTIMIZER[optimizer]
        self.criterion = CRITERION[criterion]()

        self.c1 = nn.Conv1d(1, 8, 5)
        self.c2 = nn.Conv1d(8, 16, 5)
        self.c3 = nn.Conv1d(16, 32, 5)

        self.fc1 = nn.Linear(608, 187)
        self.fc2 = nn.Linear(187, 64)
        self.fc3 = nn.Linear(64, self.num_classes)

        self.maxP = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(self.dropout)
        self.flatten = nn.Flatten()
        self.flatten0 = nn.Flatten(0)


        self.test_acc  = Accuracy()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        self.test_AUROC = AUROC(pos_label=1)
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
        logits = self.fc3(F.relu(self.dropout(x)))

        return logits

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
            loss = self.criterion(y_hat, y.float())
            y_pred = torch.sigmoid(y_hat)
        else:
            loss = self.criterion(y_hat, y)
            y_pred = torch.softmax(y_hat, dim=1)

        acc = self.train_acc(y_pred, y)
        self.log("train_loss", loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
            loss = self.criterion(y_hat, y.float())
            y_pred = torch.sigmoid(y_hat)
        else:
            loss = self.criterion(y_hat, y)
            y_pred = torch.softmax(y_hat, dim=1)

        acc = self.val_acc(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log('val_acc', acc)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
            loss = self.criterion(y_hat, y.float())
            y_pred = torch.sigmoid(y_hat)
            self.test_AUROC(y_pred, y)
            self.test_PRC(y_pred, y)

        else:
            loss = self.criterion(y_hat, y)
            y_pred = torch.softmax(y_hat, dim=1)

        acc = self.test_acc(y_pred, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def test_step_end(self, test_step_outputs):
        if(self.num_classes == 1):
            test_ROC = self.test_AUROC.compute()
            self.log("test roc", test_ROC)

            precision, recall, threshold = self.test_PRC.compute()
            self.log("test_precision", precision)
            self.log("test_recall", recall)
            self.log("test_threshold", threshold)
            test_PRC = self.test_AUC(recall, precision)
            self.log("test prc", test_PRC)

class CNNResidual(pl.LightningModule):
    def __init__(self, criterion, optimizer, num_classes = 1, lr = 1e-3, dropout= 0.1):
        super(CNNResidual, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.num_classes = num_classes
        self.dropout = dropout
        self.optimizer = OPTIMIZER[optimizer]
        self.criterion = CRITERION[criterion]()

        self.c1 = nn.Conv1d(1, 8, 5)
        self.c2 = nn.Conv1d(8, 15, 5)
        self.c3 = nn.Conv1d(16, 32, 5)

        self.fc1 = nn.Linear(608, 187)
        self.fc2 = nn.Linear(187, 32)
        self.fc3 = nn.Linear(32, self.num_classes)

        self.fc4 = nn.Linear(187, 43)

        self.maxP = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(self.dropout)
        self.flatten = nn.Flatten()
        self.flatten0 = nn.Flatten(0)

        self.test_acc  = Accuracy()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        self.test_AUROC = AUROC(pos_label=1)
        self.test_PRC = PrecisionRecallCurve(pos_label=1)
        self.test_AUC = AUC()

    def forward(self, x):
        r1 = self.fc4(x)
        r1 = r1.unsqueeze(1)

        r2 = torch.div(x, self.num_classes + 1)
        
        x = x.unsqueeze(1)

        x = self.maxP(F.relu(self.c1(self.dropout(x))))
        x = self.maxP(F.relu(self.c2(self.dropout(x))))
        x = torch.cat((x, r1), dim=1)
        x = self.maxP(F.relu(self.c3(self.dropout(x))))

        x = self.flatten(x)

        x = self.fc1(F.relu(self.dropout(x)))
        x = self.fc2(F.relu(self.dropout(x)) + r2)
        logits = self.fc3(F.relu(self.dropout(x)))

        return logits

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
            loss = self.criterion(y_hat, y.float())
            y_pred = torch.sigmoid(y_hat)
        else:
            loss = self.criterion(y_hat, y)
            y_pred = torch.softmax(y_hat, dim=1)

        acc = self.train_acc(y_pred, y)
        self.log("train_loss", loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
            loss = self.criterion(y_hat, y.float())
            y_pred = torch.sigmoid(y_hat)
        else:
            loss = self.criterion(y_hat, y)
            y_pred = torch.softmax(y_hat, dim=1)

        acc = self.val_acc(y_pred, y)
        self.log("val_loss", loss)
        self.log('val_acc', acc, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.num_classes == 1:
            y_hat = y_hat.view(-1)
            loss = self.criterion(y_hat, y.float())
            y_pred = torch.sigmoid(y_hat)
            self.test_AUROC(y_pred, y)
            self.test_PRC(y_pred, y)
        else:
            loss = self.criterion(y_hat, y)
            y_pred = torch.softmax(y_hat, dim=1)

        acc = self.test_acc(y_pred, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def test_step_end(self, test_step_outputs):
        if(self.num_classes == 1):
            test_ROC = self.test_AUROC.compute()
            self.log("test roc", test_ROC)

            precision, recall, threshold = self.test_PRC.compute()
            self.log("test_precision", precision)
            self.log("test_recall", recall)
            self.log("test_threshold", threshold)
            test_PRC = self.test_AUC(recall, precision)
            self.log("test prc", test_PRC)