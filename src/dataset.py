import pandas as pd
import torch
import numpy as np
import scanpy as sc
import anndata as ad
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TripletSelector:
    """Precomputes triplets for contrastive learning"""
    def __init__(self, margin: float = 1.0, samples_per_anchor: int = 5):
        self.margin = margin
        self.samples_per_anchor = samples_per_anchor

    def generate_triplets(self, labels: np.ndarray):
        """Generate anchor-positive-negative triplets"""
        triplets = []
        class_indices = {}
        
        # Group indices by class
        for idx, label in enumerate(labels):
            class_indices.setdefault(label, []).append(idx)
            
        # Generate triplets for each anchor
        for class_id, indices in class_indices.items():
            # Get negative class IDs
            neg_classes = [c for c in class_indices if c != class_id]
            
            for anchor_idx in indices:
                # Generate multiple positives/negatives per anchor
                for _ in range(self.samples_per_anchor):
                    # Select positive from same class (excluding self)
                    positive_idx = np.random.choice([i for i in indices if i != anchor_idx])
                    
                    # Select negative from different class
                    neg_class = np.random.choice(neg_classes)
                    negative_idx = np.random.choice(class_indices[neg_class])
                    
                    triplets.append((anchor_idx, positive_idx, negative_idx))
                    
        return np.array(triplets)


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
    """Dataset for precomputed triplets"""
    def __init__(self, adata: ad.AnnData, triplet_indices: np.ndarray, 
                 label_key: str = 'cell_type', layer: str = None):
        """
        Args:
            adata: AnnData object containing bulk RNA data
            triplet_indices: Precomputed triplet indices (anchor, positive, negative)
            label_key: Key in adata.obs containing biological labels
            layer: Layer in AnnData to use (default: .X)
        """
        self.adata = adata
        self.triplet_indices = triplet_indices
        self.label_key = label_key
        self.layer = layer
        
        # Convert labels to categorical codes
        self.labels = pd.Categorical(adata.obs[label_key]).codes
        self.n_classes = len(pd.Categorical(adata.obs[label_key]).categories)

    def __len__(self):
        return len(self.triplet_indices)

    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_idx = self.triplet_indices[idx]
        
        def get_expression(index):
            if self.layer:
                expr = self.adata.layers[self.layer][index]
            else:
                expr = self.adata.X[index]
            
            # Handle sparse matrices
            if hasattr(expr, 'toarray'):
                expr = expr.toarray().squeeze()
                
            return torch.tensor(expr, dtype=torch.float32)
        
        return {
            'anchor': get_expression(anchor_idx),
            'positive': get_expression(pos_idx),
            'negative': get_expression(neg_idx),
            'label': torch.tensor(self.labels[anchor_idx], dtype=torch.long)
        }


class ContrastiveDataModule(pl.LightningDataModule):
    """Lightning DataModule for contrastive bulk RNA experiments with in-memory AnnData"""
    def __init__(self, 
                 adata: ad.AnnData,
                 label_key: str = 'cell_type',
                 layer: str = None,
                 batch_size: int = 128,
                 num_workers: int = 4,
                 val_size: float = 0.1,
                 test_size: float = 0.0,
                 filter_genes: bool = False,
                 normalize: bool = False,
                 log1p: bool = False,
                 triplet_margin: float = 1.0,
                 samples_per_anchor: int = 5):
        super().__init__()
        self.adata = adata
        self.label_key = label_key
        self.layer = layer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.test_size = test_size
        self.filter_genes = filter_genes
        self.normalize = normalize
        self.log1p = log1p
        self.triplet_margin = triplet_margin
        self.samples_per_anchor = samples_per_anchor

    def prepare_data(self):
        # Create a copy to avoid modifying original object
        self.adata = self.adata.copy()
        
        # Basic preprocessing
        if self.filter_genes:
            sc.pp.filter_genes(self.adata, min_counts=1)
        if self.normalize:
            sc.pp.normalize_total(self.adata, target_sum=1e4)
        if self.log1p:
            sc.pp.log1p(self.adata)

    def setup(self, stage=None):
        indices = np.arange(self.adata.shape[0])
        test_size = self.test_size if self.test_size > 0 else 0
        
        # First split test if needed
        if test_size > 0:
            train_idx, test_idx = train_test_split(
                indices, 
                test_size=test_size,
                stratify=self.adata.obs[self.label_key]
            )
            self.test_data = self.adata[test_idx]
            indices = train_idx  # Remaining indices are for train/val
        else:
            self.test_data = None

        # Split remaining into train/val
        val_ratio = self.val_size / (1 - test_size) if test_size > 0 else self.val_size
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=val_ratio,
            stratify=self.adata[indices].obs[self.label_key]
        )
        
        self.train_data = self.adata[train_idx]
        self.val_data = self.adata[val_idx]
        
        # Generate triplets after splitting
        selector = TripletSelector(margin=self.triplet_margin,
                                  samples_per_anchor=self.samples_per_anchor)
        
        # Process each split
        self.train_triplets = selector.generate_triplets(
            pd.Categorical(self.train_data.obs[self.label_key]).codes
        )
        self.val_triplets = selector.generate_triplets(
            pd.Categorical(self.val_data.obs[self.label_key]).codes
        )
        if self.test_data is not None:
            self.test_triplets = selector.generate_triplets(
                pd.Categorical(self.test_data.obs[self.label_key]).codes
            )

    def train_dataloader(self):
        return DataLoader(
            ContrastiveBulkDataset(self.train_data, self.train_triplets, 
                                  self.label_key, self.layer),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            ContrastiveBulkDataset(self.val_data, self.val_triplets,
                                  self.label_key, self.layer),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        if self.test_data is not None:
            return DataLoader(
                ContrastiveBulkDataset(self.test_data, self.test_triplets,
                                      self.label_key, self.layer),
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
        return None
