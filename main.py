import argparse
import os

import pandas as pd
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split

from src.dataset import MITDataModule
from src.model import CNNBaseline, CNNModel, RNNModel, CNNResidual


MODEL_DICT = {
    "baseline_cnn": CNNBaseline,
    "vanilla_rnn": RNNModel,
    "vanilla_cnn": CNNModel,
    "cnn_residual": CNNResidual
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


def run_experiment(cfg):
    seed_everything(1234)

    datamodule = get_datamodule(
        cfg["dataset"],
        batch_size=cfg["batch_size"],
        train_split=cfg["train_val_split"]
    )

    model = MODEL_DICT[cfg["model"]](**cfg["model_args"])

    callbacks = []
    if cfg["early_stopping"]:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=cfg["patience"]))

    logger = TensorBoardLogger(save_dir="logs", name=cfg["experiment_name"])

    trainer = Trainer(max_epochs=cfg["n_epochs"], callbacks=callbacks,
                      accelerator=cfg["device"], default_root_dir="logs", logger=logger)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file",
                        default="configs/mitbih_baseline_cnn.yaml")
    parser.add_argument("--all", help="run all experiments", action="store_true")
    args = parser.parse_args()

    if args.all:
        config_dir = "configs"
        for filename in os.listdir(config_dir):
            file = os.path.join(config_dir, filename)
            with open(file) as f:
                config = yaml.load(f, yaml.FullLoader)
            run_experiment(config)
    else:
        with open(args.config) as f:
            config = yaml.load(f, yaml.FullLoader)
        run_experiment(config)
