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
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import pytorch_lightning as pl
import os
import sys
plt.style.use('default')
sys.path.append("../src")  # 将 src 目录添加到 Python 路径
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from lightning.pytorch.loggers import WandbLogger
from dataset import GraphContrastiveDataModule, SpatialGraphDataset
from torch_geometric.loader import DataLoader
from model import GraphContrastiveModel


HCC = sc.read_h5ad('../data/MERFISH/HC1_processed.h5ad')
TCGA_LIHC = pd.read_table('../data/TCGA/TCGA-LIHC.htseq_counts_clean.tsv', index_col=0)
common_genes = np.intersect1d(HCC.var_names, TCGA_LIHC.index.values)
TCGA_LIHC = TCGA_LIHC.loc[common_genes]
exp_data = TCGA_LIHC.T
clinical_data = pd.read_table('../data/TCGA/TCGA-LIHC.GDC_phenotype.tsv', index_col=0)
survival_data = pd.read_table('../data/TCGA/TCGA-LIHC.survival.tsv', index_col=0)
clinical_data = survival_data.merge(clinical_data, left_index=True, right_index=True)
common_sample = np.intersect1d(exp_data.index, clinical_data.index)
exp_data = exp_data.loc[common_sample]
clinical_data = clinical_data.loc[common_sample]
clinical_data.columns.to_series().to_csv('../data/TCGA/Processed/clinical_data.csv', index=False)

bulk_TCGA_adata = sc.AnnData(X=exp_data.values, obs=clinical_data)
bulk_TCGA_adata.write_h5ad('../data/TCGA/Processed/TCGA_LIHC.h5ad')

use_region = sc.read_h5ad('../data/MERFISH/HC1_region_sampled.h5ad')
use_region = use_region[:, common_genes]
sc.pl.embedding(use_region, basis='spatial',  color='sub_cell_type')
use_region

sc.pp.neighbors(use_region, n_neighbors=20, use_rep='spatial')
import scipy.sparse as sp
use_region.obsp['spatial_connectivities'] = sp.coo_matrix((use_region.obsp['distances'] > 0).astype(int))
use_region.obsp['spatial_distances'] = use_region.obsp['distances']
graph_dataset = SpatialGraphDataset(use_region, name='HC1_region_sampled_hop2', hops=2)
graph_dataset = SpatialGraphDataset(use_region, name='HC1_region_sampled_hop3', hops=3)
print(f'Length of graph dataset 0 is {len(graph_dataset)}')


use_region = sc.read_h5ad('../data/MERFISH/HC_2sample_combined_sampled.h5ad')
use_region = use_region[:, common_genes]
sc.pl.embedding(use_region, basis='spatial',  color='sub_cell_type')
use_region

sc.pp.neighbors(use_region, n_neighbors=20, use_rep='spatial')
import scipy.sparse as sp
use_region.obsp['spatial_connectivities'] = sp.coo_matrix((use_region.obsp['distances'] > 0).astype(int))
use_region.obsp['spatial_distances'] = use_region.obsp['distances']
graph_dataset = SpatialGraphDataset(use_region, batch_key='batch', name='HC2combined_sampled_hop2', hops=2)
graph_dataset = SpatialGraphDataset(use_region, batch_key='batch', name='HC2combined_sampled_hop3', hops=3)
print(f'Length of graph dataset 0 is {len(graph_dataset)}')