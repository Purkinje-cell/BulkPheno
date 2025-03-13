import numpy as np
import scvi
import torch.nn
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.dataloaders import AnnDataLoader
import torch
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import squidpy as sq
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import os
import sys
# sys.path.append("~/BulkPheno/src")  # 将 src 目录添加到 Python 路径
from dataset import ContrastiveDataModule
from model import ContrastiveAE
plt.style.use('default')
# sys.path.append("../src")  # 将 src 目录添加到 Python 路径
import re
bulk_adata = sc.read_h5ad('./data/Simulation/sim1_bulk.h5ad')

import pytorch_lightning as pl
data_module = ContrastiveDataModule(bulk_adata, label_key='Pheno', normalize=True, log1p=True)
model = ContrastiveAE(input_dim=500)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, data_module)