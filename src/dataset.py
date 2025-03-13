import pandas as pd
import torch
import numpy as np
import scanpy as sc
import anndata as ad
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph

# TripletSelector class removed as we now use online hard mining


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
    """Dataset for contrastive learning with individual samples"""
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
            'expression': get_expression(idx),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
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
                 log1p: bool = False):
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
        if self.test_data is not None:
            return DataLoader(
                ContrastiveBulkDataset(self.test_data, self.label_key, self.layer),
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
        return None


class PhenotypeDataModule(pl.LightningDataModule):
    """Handles bulk data with precomputed single-cell embeddings"""
    def __init__(
        self,
        bulk_adata: ad.AnnData,
        sc_embeddings: torch.Tensor,
        label_key: str = 'phenotype',
        layer: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.bulk_adata = bulk_adata
        self.sc_embeddings = sc_embeddings
        self.label_key = label_key
        self.layer = layer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Create bulk dataset with phenotype labels
        self.bulk_dataset = ContrastiveBulkDataset(
            self.bulk_adata,
            label_key=self.label_key,
            layer=self.layer
        )

    def train_dataloader(self):
        return DataLoader(
            self.bulk_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        expressions = [item['expression'] for item in batch]
        labels = [item['label'] for item in batch]
        return (
            torch.stack(expressions),
            torch.stack(labels)
        )


class SpatialGraphDataset(InMemoryDataset):
    """Dataset for spatial transcriptomics data represented as graphs"""
    def __init__(self, adata, hops=2, transform=None):
        """
        Args:
            adata: AnnData object with spatial information
            hops: Number of hops for subgraph extraction
            transform: PyG transforms to apply
        """
        self.adata = adata
        self.hops = hops
        super().__init__(transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['spatial_graph_data.pt']

    def process(self):
        graphs = []
        for node_idx in range(self.adata.shape[0]):
            # Extract k-hop subgraph with edge attributes
            subset, edge_index, _, edge_mask = k_hop_subgraph(
                node_idx, 
                self.hops, 
                self.adata.obsp['spatial_connectivities'],
                num_nodes=self.adata.shape[0]
            )
            
            # Get spatial coordinates and features
            spatial_coords = torch.tensor(self.adata.obsm['spatial'][subset], dtype=torch.float)
            features = torch.tensor(self.adata.X[subset].toarray(), dtype=torch.float)
            
            # Calculate reconstruction target
            mean_expression = features.mean(dim=0)
            
            # Create graph data object
            graph = Data(
                x=features,
                edge_index=edge_index,
                edge_attr=torch.tensor(
                    self.adata.obsp['spatial_distances'][edge_mask].data, 
                    dtype=torch.float
                ),
                pos=spatial_coords,
                center_node_idx=node_idx,
                mean_expression=mean_expression
            )
            graphs.append(graph)
        
        torch.save(self.collate(graphs), self.processed_paths[0])


class GraphContrastiveDataModule(pl.LightningDataModule):
    """DataModule for spatial graph contrastive learning"""
    def __init__(self, adata, hops=2, batch_size=64, num_workers=4,
                 feature_drop_rate=0.1, edge_drop_rate=0.2, permute_prob=0.2):
        super().__init__()
        self.adata = adata
        self.hops = hops
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_drop_rate = feature_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.permute_prob = permute_prob  # Add new parameter
        self.dataset = None

    def prepare_data(self):
        # Create the dataset
        self.dataset = SpatialGraphDataset(self.adata, hops=self.hops)

    def setup(self, stage=None):
        if self.dataset is None:
            self.prepare_data()
            
        # 90-10 train-val split
        train_size = int(0.9 * len(self.dataset))
        self.train_data = self.dataset[:train_size]
        self.val_data = self.dataset[train_size:]

    def _augment_graph(self, graph):
        """Apply biologically plausible augmentations"""
        # Create a clone to avoid modifying original
        g = graph.clone()
        
        # Feature permutation (row-wise shuffle)
        if self.permute_prob > 0 and torch.rand(1) < self.permute_prob:
            perm = torch.randperm(g.x.size(0))  # Get random permutation of node indices
            g.x = g.x[perm]  # Shuffle node features while keeping features intact
        
        # Feature dropout
        if self.feature_drop_rate > 0:
            drop_mask = torch.rand(g.x.size(1)) < self.feature_drop_rate
            g.x[:, drop_mask] = 0
            
        # Edge dropping
        if self.edge_drop_rate > 0 and g.edge_index.size(1) > 0:
            num_edges = g.edge_index.size(1)
            keep_mask = torch.rand(num_edges) > self.edge_drop_rate
            g.edge_index = g.edge_index[:, keep_mask]
            if g.edge_attr is not None:
                g.edge_attr = g.edge_attr[keep_mask]
            
        return g

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
