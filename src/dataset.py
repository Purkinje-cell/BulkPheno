import pandas as pd
import torch
import numpy as np
import scanpy as sc
import anndata as ad
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils import one_hot


class TCGADataset(Dataset):
    """
    Dataset for TCGA bulk expression data
    """

    def __init__(self, expression_mtx_path, label_path, label_type):
        self.expression_mtx = pd.read_csv(
            expression_mtx_path, index_col=0
        )  # (n_sample, n_gene)
        self.label = pd.read_csv(label_path, index_col=0)
        self.target_label = self.label[label_type].astype("category")
        self.n_cat = len(self.target_label.cat.categories)
        self.target_label = torch.tensor(
            self.target_label.cat.codes.values, dtype=torch.long
        )
        self.data = torch.tensor(
            self.expression_mtx.values, dtype=torch.float32
        )  # (n_sample, n_gene)
        self.batch_index = torch.tensor(
            self.label["batch_number"].astype("category").cat.codes.values,
            dtype=torch.long,
        )
        self.n_batch = len(self.label["batch_number"].unique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.target_label[idx]
        batch_index = self.batch_index[idx]
        return x, y, batch_index


def collate_fn(batch):
    data_batch, target_batch, batch_index = zip(*batch)

    batch_index = torch.tensor(batch_index, dtype=torch.float32).view(-1, 1)
    data_batch = torch.stack(data_batch, dim=0)
    target_batch = torch.stack(target_batch, dim=0)
    library_sizes = torch.log1p(torch.mean(data_batch, dim=1))

    mean = torch.mean(library_sizes).expand(data_batch.size(0))
    variance = torch.var(library_sizes).expand(data_batch.size(0))

    return data_batch, target_batch, batch_index, mean, variance


class ContrastiveBulkDataset(Dataset):
    """Dataset for contrastive learning with bulk RNA-seq AnnData"""
    def __init__(self, adata: ad.AnnData, label_key: str = 'cell_type', layer: str = None):
        """
        Args:
            adata: AnnData object containing bulk RNA data
            label_key: Key in adata.obs containing biological labels
            layer: Layer in AnnData to use (default: .X)
        """
        self.adata = adata
        self.label_key = label_key
        self.layer = layer
        
        # Convert labels to categorical codes
        self.labels = pd.Categorical(adata.obs[label_key]).codes
        self.n_classes = len(pd.Categorical(adata.obs[label_key]).categories)

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        # Get expression data
        if self.layer:
            expr = self.adata.layers[self.layer][idx]
        else:
            expr = self.adata.X[idx]
        
        # Handle sparse matrices
        if hasattr(expr, 'toarray'):
            expr = expr.toarray().squeeze()
            
        return {
            'expression': torch.tensor(expr, dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class ContrastiveDataModule(pl.LightningDataModule):
    """Lightning DataModule for contrastive bulk RNA experiments"""
    def __init__(self, adata_path: str, label_key: str = 'cell_type', layer: str = None,
                 batch_size: int = 128, num_workers: int = 4,
                 val_size: float = 0.1, test_size: float = 0.1):
        super().__init__()
        self.adata_path = adata_path
        self.label_key = label_key
        self.layer = layer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.test_size = test_size

    def prepare_data(self):
        # Load the full AnnData once
        self.adata = ad.read(self.adata_path)
        
        # Basic preprocessing
        sc.pp.filter_genes(self.adata, min_counts=1)
        sc.pp.normalize_total(self.adata)
        sc.pp.log1p(self.adata)

    def setup(self, stage=None):
        # Split indices
        indices = np.arange(self.adata.shape[0])
        train_idx, test_idx = train_test_split(indices, test_size=self.test_size)
        train_idx, val_idx = train_test_split(train_idx, test_size=self.val_size/(1-self.test_size))
        
        # Create subset AnnData objects
        self.train_data = self.adata[train_idx]
        self.val_data = self.adata[val_idx]
        self.test_data = self.adata[test_idx]

    def train_dataloader(self):
        return DataLoader(
            ContrastiveBulkDataset(self.train_data, self.label_key, self.layer),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            ContrastiveBulkDataset(self.val_data, self.label_key, self.layer),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            ContrastiveBulkDataset(self.test_data, self.label_key, self.layer),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
