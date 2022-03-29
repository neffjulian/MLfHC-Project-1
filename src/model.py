from re import X
from turtle import forward
from unicodedata import bidirectional
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import Accuracy, AUROC, AUC, PrecisionRecallCurve
import torchvision.models as models

OPTIMIZER = {
    "adam": Adam,
    "sgd": SGD
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
                 hidden_size,
                 num_layers,
                 optimizer,
                 criterion,
                 input_size=1,
                 dropout=0,
                 num_classes=5,
                 lr=1e-3
                 ):
        super(RNNModel, self).__init__()

        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.test_AUROC = AUROC(pos_label=1)
        self.test_PRC = PrecisionRecallCurve(pos_label=1)
        self.test_AUC = AUC()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.num_classes = num_classes
        self.optimizer = OPTIMIZER[optimizer]
        self.criterion = CRITERION[criterion]()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # Initial hidden state
        c0 = Variable(torch.zeros(self.num_layers, x.size(0),self.hidden_size))  # Initial cell state
        x = x.unsqueeze(2)
        x, _ = self.lstm(x, (h0, c0))
        x = F.dropout(x[:, -1, :], 0.1)
        x = F.relu(x)
        logits = self.linear(x)
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


class BidirectionalLSTM(pl.LightningModule):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 optimizer,
                 criterion,
                 input_size=1,
                 dropout=0,
                 num_classes=5,
                 lr=1e-3
                 ):
        super(BidirectionalLSTM, self).__init__()

        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.test_AUROC = AUROC(pos_label=1)
        self.test_PRC = PrecisionRecallCurve(pos_label=1)
        self.test_AUC = AUC()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.num_classes = num_classes
        self.optimizer = OPTIMIZER[optimizer]
        self.criterion = CRITERION[criterion]()

        self.cn1 = nn.Conv1d(1, input_size, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool1d(2)

        self.cn2 = nn.Conv1d(input_size, input_size, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=True,
                            batch_first=True)

        self.linear = nn.Linear(2*hidden_size, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # Initial hidden state
        c0 = Variable(torch.zeros(self.num_layers, x.size(0),self.hidden_size))  # Initial cell state
        x = x.unsqueeze(1)

        x = self.cn1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.cn2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = x.transpose(1, 2)

        x, _ = self.lstm(x, (h0, c0))
        x = F.dropout(x[:, -1, :], 0.1)
        x = F.relu(x)
        logits = self.linear(x)
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


class CNNModel(pl.LightningModule):
    def __init__(self, criterion, optimizer, num_classes, lr=1e-3, dropout=0.1):
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

        self.test_acc = Accuracy()
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
    def __init__(self, criterion, optimizer, num_classes=1, lr=1e-3, dropout=0.1):
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

        self.test_acc = Accuracy()
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


class TLModel(pl.LightningModule):
    def __init__(self, criterion, optimizer, num_classes=1, lr=1e-3, dropout=0.1):
        super(TLModel, self).__init__()

        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.test_AUROC = AUROC(pos_label=1)
        self.test_PRC = PrecisionRecallCurve(pos_label=1)
        self.test_AUC = AUC()

        self.lr = lr
        self.num_classes = num_classes
        self.dropout = dropout
        self.optimizer = OPTIMIZER[optimizer]
        self.criterion = CRITERION[criterion]()

        self.pretrained = models.resnet18(pretrained=(True))
        self.pretrained.eval()

        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(1000, self.num_classes)
        self.fc1 = nn.Linear(187, 53)

        self.conv = nn.Conv1d(1, 3, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(dim=0), 3, 7, 7)
        x = self.pretrained(x)
        x = self.classifier(x)
        return x

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
