import os
import yaml
import torch
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pytorch_lightning.loggers.neptune import NeptuneLogger
import torch.nn.functional as fctnl
from torch.utils.data import DataLoader
import random
import numpy as np
from data.sim_binary_s2 import generate_datasets
from data.load_real import load_oregon
from data.load_real_staff import load_staff_data


def get_device():
    if torch.cuda.is_available():
        gpu = 1
    else:
        gpu = 0
    return gpu


def get_device_string():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_project_path():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(path.absolute())


def load_yaml(path_relative):
    return yaml.safe_load(open(get_project_path() + path_relative + ".yaml", 'r'))

def save_yaml(path_relative, file):
    with open(get_project_path() + path_relative + ".yaml", 'w') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)



def load_data(config_data, standardize=True):
    if config_data["dataset"] == "sim":
        return generate_datasets(config_data)
    if config_data["dataset"] == "real":
        return load_oregon(config_data, standardize=standardize)
    if config_data["dataset"] == "real_staff":
        return load_staff_data(config_data)


def get_logger(neptune=True):
    if neptune:
        logger = NeptuneLogger(project='dennisfrauen/fair-policy')
    else:
        logger = True
    return logger


def get_config_names(model_configs):
    config_names = []
    for model_config in model_configs:
        config_names.append(model_config["name"])
    return config_names

def get_config_af(model_configs):
    config_af = []
    for model_config in model_configs:
        if "action_fair" in model_config.keys():
            config_af.append(model_config["action_fair"])
    return config_af


# Plot TSNE of representation (to check for treatment predictiveness)
def plot_TSNE_repr(psi, phi):
    n = psi.shape[0]
    # Repr (n,p), treat (n)
    tsne = TSNE(n_components=1, n_iter=300)
    embedd1 = tsne.fit_transform(psi)
    embedd2 = tsne.fit_transform(phi)
    df_plot = pd.DataFrame(columns=['x', 'y'], index=list(range(n)))
    df_plot.iloc[:n, 0:1] = embedd1
    df_plot.iloc[:n, 1:2] = embedd2
    plt.plot(df_plot.x, df_plot.y, marker='o', linestyle='', markersize=5, alpha=0.5)
    # plt.legend()
    plt.xlabel("Fair")
    plt.ylabel("Sensitive")
    plt.xlim((-10, 10))
    plt.title("TSNE of representations")
    plt.show()


def plot_TSNE_repr_label(repr, label, binary=True, title="TSNE of representations"):
    if binary:
        label = label[:, 0]
    n = repr.shape[0]
    # Repr (n,p), treat (n)
    tsne = TSNE(n_components=2, n_iter=300)
    embedd1 = tsne.fit_transform(repr)
    df_plot = pd.DataFrame(columns=['x', 'y'], index=list(range(n)))
    df_plot.iloc[:, 0:2] = embedd1
    plt.scatter(df_plot.x, df_plot.y, marker='o', c=label)
    # plt.legend()
    # plt.xlabel("Fair")
    # plt.ylabel("Sensitive")
    # plt.xlim((-10, 10))
    plt.title(title)
    plt.show()


def mse_bce(y, y_hat, y_type="continuous"):
    if y_type == "continuous":
        return torch.mean((y - y_hat) ** 2)
    if y_type == "binary":
        return fctnl.binary_cross_entropy(y_hat, y, reduction='mean')


def train_model(model, datasets, config):
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    validation = config["experiment"]["validation"]
    logger = get_logger(config["experiment"]["neptune"])

    trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False,
                         gpus=get_device(), logger=logger, enable_checkpointing=False)

    train_loader = DataLoader(dataset=datasets["d_train"], batch_size=batch_size, shuffle=True)
    if validation:
        val_loader = DataLoader(dataset=datasets["d_val"], batch_size=batch_size, shuffle=False)
        trainer.fit(model, train_loader, val_loader)
        val_results = trainer.validate(model=model, dataloaders=val_loader, verbose=False)
    else:
        trainer.fit(model, train_loader)
        val_results = None
    return {"trained_model": model, "val_results": val_results[0], "logger": logger}
