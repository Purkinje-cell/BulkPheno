from re import T
from matplotlib import pyplot as plt
import os
import pandas as pd
import torch
import numpy as np
import scanpy as sc
import anndata as ad
import pytorch_lightning as pl
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.utils import k_hop_subgraph, from_scipy_sparse_matrix, remove_self_loops
from tqdm import tqdm

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

    def __init__(
        self, adata: ad.AnnData, label_key: str = "cell_type", layer: str = None
    ):
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
            if hasattr(expr, "toarray"):
                expr = expr.toarray().squeeze()

            return torch.tensor(expr, dtype=torch.float32)

        return {
            "expression": get_expression(idx),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class ContrastiveDataModule(pl.LightningDataModule):
    """Lightning DataModule for contrastive bulk RNA experiments with in-memory AnnData"""

    def __init__(
        self,
        adata: ad.AnnData,
        label_key: str = "cell_type",
        layer: str = None,
        batch_size: int = 128,
        num_workers: int = 4,
        val_size: float = 0.1,
        test_size: float = 0.0,
        filter_genes: bool = False,
        normalize: bool = False,
        log1p: bool = False,
    ):
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
                indices, test_size=test_size, stratify=self.adata.obs[self.label_key]
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
            stratify=self.adata[indices].obs[self.label_key],
        )

        self.train_data = self.adata[train_idx]
        self.val_data = self.adata[val_idx]

    def train_dataloader(self):
        return DataLoader(
            ContrastiveBulkDataset(self.train_data, self.label_key, self.layer),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            ContrastiveBulkDataset(self.val_data, self.label_key, self.layer),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.test_data is not None:
            return DataLoader(
                ContrastiveBulkDataset(self.test_data, self.label_key, self.layer),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        return None


class PhenotypeDataModule(pl.LightningDataModule):
    """Handles bulk data with precomputed single-cell embeddings"""

    def __init__(
        self,
        bulk_adata: ad.AnnData,
        sc_embeddings: torch.Tensor,
        label_key: str = "phenotype",
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
            self.bulk_adata, label_key=self.label_key, layer=self.layer
        )

    def train_dataloader(self):
        return DataLoader(
            self.bulk_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        expressions = [item["expression"] for item in batch]
        labels = [item["label"] for item in batch]
        return (torch.stack(expressions), torch.stack(labels))
class SpatialGraphDataset(Dataset):
    """
    Memory-efficient dataset for spatial transcriptomics data represented as graphs
    Stores individual graphs on disk and loads them on demand
    """

    def __init__(
        self, 
        adata: ad.AnnData, 
        name: str, 
        batch_key=None, 
        hops=2, 
        root="../data",
        downsample_num=None,
        transform=None):
        """
        Args:
            adata: AnnData object with spatial information
            name: Unique name for this dataset
            batch_key: Key in adata.obs for batch information
            hops: Number of hops for subgraph extraction
            root: Root directory for storing processed data
            transform: PyG transforms to apply
        """
        self.adata = adata
        self.name = name
        self.hops = hops
        self.batch_key = batch_key
        self.root = root
        self.transform = transform
        self.processed_dir = os.path.join(root, "processed")
        self.graph_dir = os.path.join(self.processed_dir, f"graphs_{name}")
        self.downsample_num = downsample_num
        
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        
        # Load or create the index of valid graphs
        self.valid_indices = self._load_or_create_index()
        self.valid_indices = np.array(self.valid_indices) if not isinstance(self.valid_indices, np.ndarray) else self.valid_indices

    def _load_or_create_index(self):
        """Load existing index or create a new one if not found"""
        meta_path = os.path.join(self.processed_dir, f"graph_index_{self.name}.pt")
        if os.path.exists(meta_path):
            return torch.load(meta_path)
        
        # Process and create index if not exists
        valid_indices = self._preprocess()
        torch.save(valid_indices, meta_path)
        valid_indices = np.array(valid_indices)
        return valid_indices
    
    def process_graph(self, node_idx, edge_index, edge_attr):
        
        graph_path = os.path.join(self.graph_dir, f"graph_{node_idx}.pt")
        # Extract k-hop subgraph with edge attributes
        subset, edge_index_sub, _, edge_mask = k_hop_subgraph(
            node_idx,
            self.hops,
            edge_index,
            num_nodes=self.adata.shape[0],
            relabel_nodes=True,
        )
        
        # Skip nodes with no neighbors
        if subset.shape[0] <= 1:
            print(f'Node {node_idx} has no neighbors')
            return False
            
        # Get spatial coordinates and features
        spatial_coords = torch.tensor(
            self.adata.obsm["spatial"][subset], dtype=torch.float
        )
        
        if hasattr(self.adata.X, "toarray"):
            features = torch.tensor(
                self.adata.X[subset].toarray(), dtype=torch.float
            )
        else:
            features = torch.tensor(self.adata.X[subset], dtype=torch.float)

        # Calculate reconstruction target
        mean_expression = features.mean(dim=0)
        edge_attr_sub = edge_attr[edge_mask]
        
        # Create graph data object
        if self.batch_key:
            batch = torch.tensor(self.adata.obs[self.batch_key][node_idx]).reshape(-1)
            graph = Data(
                x=features,
                edge_index=edge_index_sub,
                edge_attr=edge_attr_sub,
                pos=spatial_coords,
                center_node_idx=node_idx,
                mean_expression=mean_expression,
                batch=batch,
            )
        else:
            graph = Data(
                x=features,
                edge_index=edge_index_sub,
                edge_attr=edge_attr_sub,
                pos=spatial_coords,
                center_node_idx=node_idx,
                mean_expression=mean_expression,
                batch=torch.zeros(1, dtype=torch.long)
            )
            
        # Save individual graph to disk with compression
        torch.save(graph, graph_path, _use_new_zipfile_serialization=True)
        return True

    def _preprocess(self):
        """Process and save individual graphs to disk"""
        valid_indices = []
        
        # Create graph structure from spatial distances
        edge_index, edge_attr = from_scipy_sparse_matrix(
            self.adata.obsp["distances"]
        )
        mask = edge_attr < 50
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        
        # Visualize edge length distribution
        sns.displot(edge_attr, bins=100)
        plt.savefig(f'../figures/{self.name}_edge_length.png')

        remained_nodes = np.arange(self.adata.shape[0])
        if self.downsample_num is not None:
            remained_nodes = np.random.choice(
                remained_nodes, size=self.downsample_num, replace=False
            )
        remained_nodes = remained_nodes.tolist()
        
        for node_idx in tqdm(remained_nodes, desc="Processing graphs"):
            # Skip if already processed
            graph_path = os.path.join(self.graph_dir, f"graph_{node_idx}.pt")
            if os.path.exists(graph_path):
                valid_indices.append(node_idx)
                continue
            # Process and save graph
            if self.process_graph(node_idx, edge_index, edge_attr):
                valid_indices.append(node_idx)
            else:
                print(f"Node {node_idx} has no neighbors")

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Load a single graph from disk on demand"""
        node_idx = self.valid_indices[idx]
        graph_path = os.path.join(self.graph_dir, f"graph_{node_idx}.pt")
        data = torch.load(graph_path)
        return data if self.transform is None else self.transform(data)

    @property
    def processed_file_names(self):
        """Return the index file name"""
        return [f"graph_index_{self.name}.pt"]




class BulkDataset(Dataset):
    """Dataset for bulk RNA with pseudo-bulk generation from spatial data"""

    def __init__(
        self, 
        adata: ad.AnnData | None = None, 
        pheno_label: str | None = None, 
        spatial_graph_dataset: SpatialGraphDataset | None = None, 
        hops: int = 2
    ):
        if adata is not None:
            # Real bulk RNA mode
            self.mode = "real"
            self.expressions = (
                torch.tensor(adata.X.toarray(), dtype=torch.float32)
                if hasattr(adata.X, "toarray")
                else torch.tensor(adata.X, dtype=torch.float32)
            )
            if pheno_label:
                self.phenotype_label = torch.tensor(
                    adata.obs[pheno_label].cat.codes.values, dtype=torch.long
                )
            else:
                self.phenotype_label = torch.zeros(len(adata), dtype=torch.long)
        elif spatial_graph_dataset is not None:
            # Pseudo-bulk mode
            self.mode = "pseudo"
            self.graph_dataset = spatial_graph_dataset
            self.hops = hops
            self._preprocess_pseudo_bulk()
        else:
            raise ValueError("Must provide either adata or spatial_graph_dataset")

    def _preprocess_pseudo_bulk(self):
        """Generate pseudo-bulk from spatial graph data"""
        self.pseudo_expressions = []
        self.graph_indices = []

        for idx in range(len(self.graph_dataset)):
            graph = self.graph_dataset[idx]
            self.pseudo_expressions.append(graph.mean_expression)
            self.graph_indices.append(idx)

        self.pseudo_expressions = torch.stack(self.pseudo_expressions)

    def __len__(self):
        return (
            len(self.expressions)
            if self.mode == "real"
            else len(self.pseudo_expressions)
        )

    def __getitem__(self, idx):
        if self.mode == "real":
            return {
                "expression": self.expressions[idx], 
                "phenotype": self.phenotype_label[idx],
                "is_real": True}
        else:
            return {
                "expression": self.pseudo_expressions[idx],
                "graph_idx": self.graph_indices[idx],
                "is_real": False,
            }


class BulkDataModule(pl.LightningDataModule):
    """Handles real and pseudo-bulk data loading"""

    def __init__(
        self,
        bulk_adata: ad.AnnData = None,
        spatial_adata: ad.AnnData = None,
        spatial_hops: int = 2,
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        super().__init__()
        self.bulk_adata = bulk_adata
        self.spatial_adata = spatial_adata
        self.spatial_hops = spatial_hops
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.spatial_graph_dataset = None

    def prepare_data(self):
        if self.spatial_adata is not None:
            self.spatial_graph_dataset = SpatialGraphDataset(
                self.spatial_adata, name="pseudo_bulk_2c", hops=self.spatial_hops
            )

    def setup(self, stage=None):
        if self.bulk_adata is not None:
            indices = np.arange(self.bulk_adata.shape[0])
            train_idx, test_idx = train_test_split(indices, test_size=self.test_split)
            train_idx, val_idx = train_test_split(train_idx, test_size=self.val_split)
            self.train_idx = train_idx
            self.val_idx = val_idx
            self.test_idx = test_idx

            self.real_train = BulkDataset(adata=self.bulk_adata[train_idx])
            self.real_val = BulkDataset(adata=self.bulk_adata[val_idx])
            self.real_test = BulkDataset(adata=self.bulk_adata[test_idx])

        if self.spatial_graph_dataset is not None:
            indices = np.arange(len(self.spatial_graph_dataset))
            train_idx, test_idx = train_test_split(
                indices, test_size=self.test_split, random_state=42
            )
            train_idx, val_idx = train_test_split(
                train_idx, test_size=self.val_split, random_state=42
            )
            self.train_idx = train_idx
            self.val_idx = val_idx
            self.test_idx = test_idx

            self.pseudo_train = BulkDataset(
                spatial_graph_dataset=Subset(self.spatial_graph_dataset, train_idx)
            )
            self.pseudo_val = BulkDataset(
                spatial_graph_dataset=Subset(self.spatial_graph_dataset, val_idx)
            )
            self.pseudo_test = BulkDataset(
                spatial_graph_dataset=Subset(self.spatial_graph_dataset, test_idx)
            )

    def train_dataloader(self):
        if hasattr(self, "real_train"):
            real_loader = DataLoader(
                self.real_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
            )
            return real_loader

        if hasattr(self, "pseudo_train"):
            pseudo_loader = DataLoader(
                self.pseudo_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn,
            )
            return pseudo_loader

    def _collate_fn(self, batch):
        """Custom collate to handle mixed real/pseudo batches"""
        is_real = batch[0]["is_real"]
        expressions = torch.stack([item["expression"] for item in batch])

        if is_real:
            return {"expression": expressions, "is_real": True}
        else:
            graph_indices = [item["graph_idx"] for item in batch]
            return {
                "expression": expressions,
                "graph_indices": graph_indices,
                "is_real": False,
            }


class GraphContrastiveDataModule(pl.LightningDataModule):
    """DataModule for spatial graph contrastive learning"""

    def __init__(self, adata, hops=2, batch_size=64, num_workers=4, val_split=0.2):
        super().__init__()
        self.adata = adata
        self.hops = hops
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.dataset = None

    def prepare_data(self):
        # Create the dataset
        self.dataset = SpatialGraphDataset(self.adata, name="graph_cl", hops=self.hops)

    def setup(self, stage=None):
        if self.dataset is None:
            self.prepare_data()

        # Create train/val splits with random permutation
        indices = np.random.permutation(len(self.dataset))
        train_size = int((1 - self.val_split) * len(indices))
        self.train_idx = indices[:train_size]
        self.val_idx = indices[train_size:]

    def train_dataloader(self):
        from torch.utils.data import Subset

        return PyGLoader(
            Subset(self.dataset, self.train_idx),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            follow_batch=["x", "mean_expression"],
        )

    def val_dataloader(self):
        from torch.utils.data import Subset

        return PyGLoader(
            Subset(self.dataset, self.val_idx),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            follow_batch=["x", "mean_expression"],
        )
