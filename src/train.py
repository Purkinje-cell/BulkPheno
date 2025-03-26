import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scipy.sparse as sp
import scvi
import seaborn as sns
import squidpy as sq
import torch
import torch.nn
from dataset import BulkDataModule, BulkDataset
from matplotlib.pylab import f
from model import BulkEncoderModel, EmbeddingStore, GraphContrastiveModel
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.dataloaders import AnnDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_scatter import scatter_add, scatter_mean

plt.style.use('default')
import re

from dataset import GraphContrastiveDataModule, SpatialGraphDataset
from lightning.pytorch.loggers import WandbLogger
from model import GraphContrastiveModel
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def collate_fn_new(batch):
    """Custom collate to handle mixed real/pseudo batches"""
    is_real = batch[0]['is_real']
    expressions = torch.stack([item['expression'] for item in batch])
    
    if is_real:
        return {'expression': expressions, 'is_real': True}
    else:
        graph_indices = [item['graph_idx'] for item in batch]
        return {
            'expression': expressions,
            'graph_indices': graph_indices,
            'is_real': False
        }

bulk_TCGA_adata = sc.read_h5ad('../data/TCGA/Processed/TCGA_LIHC.h5ad')
use_region = sc.read_h5ad('../data/MERFISH/HC1_region_3_test_pipe.h5ad')
common_genes = np.intersect1d(use_region.var_names, bulk_TCGA_adata.var_names)
use_region = use_region[:, common_genes].copy()

bulk_TCGA_adata.layers['count'] = bulk_TCGA_adata.X.copy()
sc.pp.normalize_total(bulk_TCGA_adata, target_sum=1e4)
sc.pp.log1p(bulk_TCGA_adata)
sc.pp.scale(bulk_TCGA_adata)
sc.pp.pca(bulk_TCGA_adata)
print(use_region)
sc.pl.embedding(use_region, basis='spatial', color='sub_cell_type')
plt.savefig('../figures/HC1_region_3_test_sub_cell_type_spatial.png')
sc.pp.neighbors(use_region, n_neighbors=20, use_rep='spatial')

graph_dataset = SpatialGraphDataset(use_region, name='HC1_region3_test', hops=2)
print(f'Length of graph dataset 0 is {len(graph_dataset)}')

train_idx = np.random.choice(int(graph_dataset.len()), int(graph_dataset.len() * 0.8), replace=False)
val_idx = np.setdiff1d(graph_dataset.indices(), train_idx)
train_dataset = graph_dataset[train_idx]
val_dataset = graph_dataset[val_idx]
print(f'Length of train dataset is {len(train_dataset)}')
print(f'Length of val dataset is {len(val_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, follow_batch=['x', 'mean_expression'])
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, follow_batch=['x', 'mean_expression'])

wandb_logger = WandbLogger(project="SpatialGCL")

graph_model = GraphContrastiveModel(input_dim=499, latent_dim=32, recon_weight=0.8, margin=1.0)
graph_trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, log_every_n_steps=1)
graph_trainer.fit(graph_model, train_loader, val_loader)

bulk_encode_model = BulkEncoderModel(input_dim=499, gcl_model=graph_model, latent_dim=64, hidden_dims=[256, 128])
train_loader = TorchDataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4, collate_fn=collate_fn_new)
val_loader = TorchDataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4, collate_fn=collate_fn_new)
bulk_encode_model.set_graph_dataset(train_dataset)
wandb_logger = WandbLogger(project="SpatialGCL")
trainer = pl.Trainer(max_epochs=20, log_every_n_steps=5, logger=wandb_logger)
trainer.fit(bulk_encode_model, train_loader)

graph_loader = DataLoader(graph_dataset, batch_size=512, shuffle=False, follow_batch=['x', 'mean_expression'])

graph_embeddings = []
graph_model = graph_model.to('cuda')
graph_model.eval()
with torch.no_grad():
    for batch in graph_loader:
        batch = batch.to('cuda')
        x, edge_index, edge_attr, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        z = graph_model.encode(x, edge_index, edge_attr, batch)
        graph_embeddings.append(z.cpu().numpy())

# Concatenate results
graph_embeddings = np.concatenate(graph_embeddings, axis=0)
graph_embeddings.shape

use_region = use_region[:, common_genes].copy()

use_survival_data = bulk_TCGA_adata[(bulk_TCGA_adata.obs['OS'] == 1) & (bulk_TCGA_adata.obs['sample_type.samples'] != "Solid Tissue Normal")].copy()
use_survival_data.obs['Median_OS'] = use_survival_data.obs['OS.time'].median()
use_survival_data.obs['Large_OS'] = use_survival_data.obs['OS.time'] > use_survival_data.obs['Median_OS']
use_survival_data.obs['Large_OS'] = use_survival_data.obs['Large_OS'].map({True: 'Large', False: 'Small'}).astype('category')
use_survival_data.obs['Large_OS'].value_counts()

bulk_encode_model = bulk_encode_model.to('cuda')
bulk_embedding = bulk_encode_model.get_latent_embedding(bulk_TCGA_adata)
bulk_labels = bulk_TCGA_adata.obs['sample_type.samples'].values
# bulk_embedding = bulk_encode_model.get_latent_embedding(use_survival_data)
# bulk_labels = use_survival_data.obs['Large_OS'].values
bulk_ids = np.arange(len(bulk_labels))
print(len(bulk_embedding), len(bulk_labels))
# bulk_embedding[0]
# query_res = graph_model.embedding_store.query(bulk_embedding, k=5000)
# query_res = query_res
# query_res['distances'] = query_res['distances'].flatten()
# query_res['indices'] = query_res['indices'].flatten()
# query_res['sample_ids'] = query_res['sample_ids'].flatten()
# sns.displot(query_res['distances'], kde=True)
# del query_res['labels']
# query_res = pd.DataFrame(query_res).query('distances > 0.7')
# query_res
# val_dataset[0].center_node_idx

bulk_embedding_store = EmbeddingStore(metric='cosine')
bulk_embedding_store.build_index(bulk_embedding, bulk_labels, bulk_ids)

query_res = bulk_embedding_store.query(graph_embeddings, k=200)

distance_tensor = torch.Tensor(query_res['distances'])
label_tensor = torch.Tensor(query_res['labels'].codes).to(torch.long)
sim_tensor = scatter_mean(distance_tensor, label_tensor, dim=1)
sim = sim_tensor.numpy()
sim_df = pd.DataFrame(sim, columns=bulk_TCGA_adata.obs['sample_type.samples'].cat.categories)
# sim_df = pd.DataFrame(sim, columns=use_survival_data.obs['Large_OS'].cat.categories)
sim_df

rank_tensor = torch.argsort(distance_tensor, dim=1, descending=False)
rank_reciprocal_tensor = 1 / (rank_tensor + 1)
rank_sum_tensor = scatter_mean(rank_reciprocal_tensor, label_tensor, dim=1)
rank_sum = rank_sum_tensor.numpy()
rank_sum_df = pd.DataFrame(rank_sum, columns=use_survival_data.obs['Large_OS'].cat.categories)
sim_df = rank_sum_df
sim_df

use_region.obsp['spatial_connectivities'] = sp.csr_matrix(use_region.obsp['spatial_connectivities'])
sim_new = use_region.obsp['spatial_connectivities'] @ sim / 20
sim_df = pd.DataFrame(sim_new, columns=use_survival_data.obs['Large_OS'].cat.categories)
# sim_df = pd.DataFrame(sim_new, columns=bulk_TCGA_adata.obs['sample_type.samples'].cat.categories)
sim_df

sim_df.index = use_region[graph_dataset[sim_df.index.values].center_node_idx.numpy()].obs_names
use_region.obs = use_region.obs.drop(columns=['Primary Tumor', 'Solid Tissue Normal', 'Recurrent Tumor'])
# use_region.obs = use_region.obs.drop(columns=['Large', 'Small'])
use_region.obs = use_region.obs.merge(sim_df, left_index=True, right_index=True)

use_region.obs['Prediction'] = use_region.obs[['Primary Tumor', 'Solid Tissue Normal', 'Recurrent Tumor']].idxmax(axis=1)
# use_region.obs['Prediction'] = use_region.obs[['Large', 'Small']].idxmax(axis=1)
# use_region.obs['Prediction_Threshold'] = use_region.obs[['Primary Tumor', 'Solid Tissue Normal', 'Recurrent Tumor']].max(axis=1)
# use_region.obs['Prediction'] = np.where(use_region.obs['Prediction_Threshold'] > 0.7, use_region.obs['Prediction'], 'Uncertain')
use_region.obs['Prediction'].value_counts()

cancer_region = use_region[(use_region.obs['cell_type'] == 'Cancer cell') & (use_region.obs['Prediction'] != "Solid Tissue Normal")].copy()
sc.tl.rank_genes_groups(cancer_region, groupby='Prediction', method='wilcoxon')

diff_genes = cancer_region.uns['rank_genes_groups']['names']
diff_pval_adj = cancer_region.uns['rank_genes_groups']['pvals_adj']
diff_foldchange = cancer_region.uns['rank_genes_groups']['logfoldchanges']
diff_genes=pd.DataFrame(diff_genes).melt(var_name='Prediction', value_name='gene')
diff_pval_adj=pd.DataFrame(diff_pval_adj).melt(var_name='Prediction', value_name='pval_adj')
diff_foldchange=pd.DataFrame(diff_foldchange).melt(var_name='Prediction', value_name='foldchange')
diff_res = pd.concat([diff_genes, diff_pval_adj['pval_adj'], diff_foldchange['foldchange']], axis=1)
diff_res = diff_res.query('pval_adj < 0.05 & foldchange > 1 & Prediction != "Solid Tissue Normal"')
diff_res['log_pval_adj'] = -np.log10(diff_res['pval_adj'])
# diff_res.loc[diff_res['Prediction'] == 'Primary Tumor', 'foldchange'] = - diff_res.loc[diff_res['Prediction'] == 'Primary Tumor', 'foldchange']
diff_res.loc[diff_res['Prediction'] == 'Large', 'foldchange'] = - diff_res.loc[diff_res['Prediction'] == 'Large', 'foldchange']

plt.figure(figsize=(10, 10))
sns.scatterplot(data=diff_res, x = 'foldchange', y='log_pval_adj', hue='Prediction')
for i, row in diff_res.iterrows():
    if abs(row['foldchange']) > 2:
        plt.text(row['foldchange'], row['log_pval_adj'], row['gene'], fontsize=9, ha='right')
plt.savefig('diff_genes_OS.png')
plt.show()

# for x in ['cell_type', 'Large', 'Small', 'Prediction']:
for x in ['cell_type', 'Primary Tumor', 'Solid Tissue Normal', 'Recurrent Tumor', 'Prediction']:
    sc.pl.embedding(use_region, basis='spatial', color=x, save=f'{x}_region_3_OS.pdf')

cell_prop=use_region.obs[['Prediction', 'cell_type']].groupby('Prediction').value_counts(['cell_type'],normalize=True).sort_index().reset_index()
cell_prop.columns=['Prediction', 'cell_type', 'prop']
sns.barplot(data=cell_prop, x='Prediction', y='prop', hue='cell_type')
plt.savefig('cell_prop_region_3a_OS.png')