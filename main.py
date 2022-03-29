import argparse
import os

import pandas as pd
from sklearn.utils import resample
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from src.dataset import MITDataModule
from src.model import CNNBaseline, CNNModel, RNNModel, CNNResidual, BidirectionalLSTM, TLModel


MODEL_DICT = {
    "baseline_cnn": CNNBaseline,
    "vanilla_rnn": RNNModel,
    "vanilla_cnn": CNNModel,
    "cnn_residual": CNNResidual,
    "bidirectional_lstm": BidirectionalLSTM,
    "transfer_learning": TLModel
}


def get_balanced_dataset(df):
    df0 = df[df[187] == 0]
    df1 = df[df[187] == 1]
    df2 = df[df[187] == 2]
    df3 = df[df[187] == 3]
    df4 = df[df[187] == 4]

    df0_resampled = df0.sample(n=20000, random_state=42)
    df2_resampled = resample(df2, n_samples=20000,
                             random_state=42, replace=True)
    df3_resampled = resample(df3, n_samples=20000,
                             random_state=42, replace=True)
    df4_resampled = resample(df4, n_samples=20000,
                             random_state=42, replace=True)
    df1_resampled = resample(df1, n_samples=20000,
                             random_state=42, replace=True)

    return pd.concat([df0_resampled, df1_resampled, df2_resampled, df3_resampled, df4_resampled]).reset_index(drop=True)


def get_datamodule(name, do_resample=False, **kwargs):
    if name == "mitbih":
        df_train = pd.read_csv("data/mitbih_train.csv", header=None)
        df_test = pd.read_csv("data/mitbih_test.csv", header=None)
        if do_resample:
            df_train = get_balanced_dataset(df_train)
    elif name == "ptbdb":
        df_1 = pd.read_csv("data/ptbdb_normal.csv", header=None)
        df_2 = pd.read_csv("data/ptbdb_abnormal.csv", header=None)
        if do_resample:
            df_1 = resample(df_1, n_samples=7500, random_state=42, replace=True)
            df_2 = resample(df_2, n_samples=7500, random_state=42, replace=True)
        df = pd.concat([df_1, df_2])
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=1337, stratify=df[187])
    else:
        raise AttributeError("Dataset not available")

    return MITDataModule(df_train, df_test, **kwargs)


def run_experiment(cfg):
    seed_everything(1234)

    datamodule = get_datamodule(
        cfg["dataset"],
        cfg.get("resample", False),
        batch_size=cfg["batch_size"],
        train_split=cfg["train_val_split"]
    )

    model = MODEL_DICT[cfg["model"]](**cfg["model_args"])

    callbacks = []
    if cfg["early_stopping"]:
        callbacks.append(EarlyStopping(
            monitor="val_loss", patience=cfg["patience"]))

    logger = TensorBoardLogger(save_dir="logs", name=cfg["experiment_name"])

    trainer = Trainer(max_epochs=cfg["n_epochs"], callbacks=callbacks,
                      accelerator=cfg["device"], default_root_dir="logs", logger=logger)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs='+', help="path to one or more config file",
                        default="configs/mitbih_baseline_cnn.yaml")

    args = parser.parse_args()

    for filename in args.config:
        with open(filename) as f:
            config = yaml.load(f, yaml.FullLoader)
        run_experiment(config)
