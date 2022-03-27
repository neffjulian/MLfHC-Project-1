import torch
import numpy as np
import pandas as pd

from pytorch_lightning import LightningDataModule

from torch.utils.data import Dataset, DataLoader, random_split


class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.X = df.loc[:, list(range(187))].astype(np.float32).values
        self.y = df.loc[:, 187].astype(np.int8).values

    def __len__(self):
        return self.X.__len__()

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class MITDataModule(LightningDataModule):
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, batch_size=32, train_split=0.8, num_workers=4):
        super().__init__()
        self.df_train = df_train
        self.df_test = df_test
        self.batch_size = batch_size
        self.train_split = train_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        timeseries_full = TimeSeriesDataset(self.df_train)
        train_length = int(self.train_split*len(timeseries_full))
        val_length = len(timeseries_full) - train_length
        self.train_set, self.val_set = random_split(
            timeseries_full, [train_length, val_length])
        self.test_set = TimeSeriesDataset(self.df_test)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
