import argparse
import os
import sys

import pandas as pd
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split

from src.dataset import MITDataModule
from src.model import CNNBaseline, CNNModelNonBinary, RNNModel



MODEL_DICT = {
    "baseline_cnn": CNNBaseline,
    "vanilla_rnn": RNNModel,
    "vanilla_cnn": CNNModelNonBinary
}


def get_datamodule(name, **kwargs):
    if name == "mitbih":
        df_train = pd.read_csv("data/mitbih_train.csv", header=None)
        df_test = pd.read_csv("data/mitbih_test.csv", header=None)
    elif name == "ptbdb":
        df_1 = pd.read_csv("data/ptbdb_normal.csv", header=None)
        df_2 = pd.read_csv("data/ptbdb_abnormal.csv", header=None)
        df = pd.concat([df_1, df_2])
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=1337, stratify=df[187])
    else:
        raise AttributeError("Dataset not available")

    return MITDataModule(df_train, df_test, **kwargs)


def main(cfg):
    seed_everything(1234)

    datamodule = get_datamodule(
        cfg["dataset"],
        batch_size=cfg["batch_size"],
        train_split=cfg["train_val_split"]
    )

    model = MODEL_DICT[cfg["model"]](**cfg["model_args"])

    callbacks = []
    if cfg["early_stopping"]:
        callbacks.append(EarlyStopping(monitor="val_loss"))

    trainer = Trainer(max_epochs=cfg["n_epochs"], callbacks=callbacks,
                      accelerator=cfg["device"], default_root_dir="logs")
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file",
                        default="configs/default_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    main(config)
