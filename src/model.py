import collections
from typing import Callable, Iterable, Literal, Union, List, Dict, Any


import anndata as ad
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import squidpy as sq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
try:
    import faiss
except ImportError:
    faiss = None
from torch.utils.data import DataLoader, Dataset
from anndata import AnnData
from networkx import subgraph
from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.distributions import NegativeBinomial, Normal, ZeroInflatedNegativeBinomial
from scvi.model.base import (
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
    EmbeddingMixin,
    ArchesMixin,
)
from scvi.module import VAE
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Decoder, DecoderSCVI, Embedding, Encoder, FCLayers
from scvi.utils import setup_anndata_dsp
from torch.distributions import Categorical, Distribution, Normal
from torch.distributions import kl_divergence
from torch.distributions import kl_divergence as kl
from torch.nn.functional import one_hot
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    PNAConv,
    SAGEConv,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
from torch_scatter import scatter_add

# from module import BulkVAEModule
# from training_mixin import BasicTrainingMixin
# from utils import broadcast_labels, one_hot
from dataset import ContrastiveBulkDataset


class FullCrossAttention(nn.Module):
    """Attention using all single-cell embeddings simultaneously"""
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, bulk_embeddings, sc_embeddings):
        """
        Args:
            bulk_embeddings: (batch_size, embed_dim)
            sc_embeddings: (num_sc_cells, embed_dim) - fixed for all batches
        """
        # Expand sc embeddings for batch processing
        sc_embeddings = sc_embeddings.unsqueeze(0).repeat(bulk_embeddings.size(0), 1, 1)  # (B, N, D)
        
        # Attention computation
        attn_output, attn_weights = self.attention(
            query=bulk_embeddings.unsqueeze(1),
            key=sc_embeddings,
            value=sc_embeddings
        )
        
        return attn_output.squeeze(1), attn_weights  # (B, D), (B, 1, N)


class EmbeddingStore:
    """Stores and indexes latent embeddings for similarity queries"""
    def __init__(self, metric: str = 'cosine'):
        self.metric = metric
        self.embeddings = None
        self.index = None
        self.labels = None
        self.sample_ids = None
        
    def build_index(self, embeddings: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray):
        """Build similarity search index"""
        self.embeddings = embeddings
        self.labels = labels
        self.sample_ids = sample_ids
        
        if faiss is not None:
            # FAISS implementation for production use
            dim = embeddings.shape[1]
            if self.metric == 'cosine':
                self.index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(embeddings)
            else:  # L2
                self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
        else:
            # Fallback to sklearn implementation
            n_neighbors = min(100, len(embeddings)-1)
            self.index = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=self.metric
            ).fit(embeddings)

    def query(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        label_filter: str = None
    ) -> dict:
        """Find similar samples in the embedding space"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first")
            
        # Convert to numpy if needed
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
            
        if faiss is not None:
            if self.metric == 'cosine':
                faiss.normalize_L2(query_embedding)
            distances, indices = self.index.search(query_embedding, k)
        else:
            distances, indices = self.index.kneighbors(query_embedding, n_neighbors=k)
            
        return {
            'indices': indices,
            'distances': distances,
            'sample_ids': self.sample_ids[indices],
            'labels': self.labels[indices]
        }


class TripletLoss(nn.Module):
    """Online hard triplet mining with PyTorch's TripletMarginLoss"""

    def __init__(self, margin=1.0, distance="euclidean"):
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, embeddings, labels):
        """Compute loss using hardest triplets in batch"""
        triplets = self._get_triplets(embeddings, labels)
        if triplets is None or len(triplets) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return self.loss_fn(*triplets)
        
    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(a, b, dim=-1)
    
    def euclidean_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.pairwise_distance(a, b, p=2)

    def _get_triplets(self, embeddings, labels):
        # Update distance calculation to use cosine similarity
        pairwise_dist = 1 - self.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0)
        )

        labels = labels.view(-1, 1)
        mask_same = labels == labels.t()
        mask_diff = ~mask_same

        triplets = []
        for i in range(len(embeddings)):
            # Hardest positive (furthest)
            pos_mask = mask_same[i]
            pos_mask[i] = False  # Exclude self
            if not pos_mask.any():
                continue
            hardest_pos = torch.argmax(pairwise_dist[i][pos_mask]).item()
            pos_idx = torch.where(pos_mask)[0][hardest_pos]

            # Hardest negative (closest)
            neg_mask = mask_diff[i]
            if not neg_mask.any():
                continue
            hardest_neg = torch.argmin(pairwise_dist[i][neg_mask]).item()
            neg_idx = torch.where(neg_mask)[0][hardest_neg]

            triplets.append((i, pos_idx, neg_idx))

        if not triplets:
            return None

        indices = torch.tensor(triplets, device=embeddings.device)
        return (
            embeddings[indices[:, 0]],
            embeddings[indices[:, 1]],
            embeddings[indices[:, 2]],
        )


# class BulkVAE(UnsupervisedTrainingMixin, VAEMixin, BaseModelClass):
#     def __init__(
#         self,
#         adata: AnnData,
#         n_latent: int = 10,
#         **model_kwargs,
#     ):
#         super().__init__(adata)

#         self.module = BulkVAEModule(
#             n_input=self.summary_stats["n_vars"],
#             n_batch=self.summary_stats["n_batch"],
#             n_latent=n_latent,
#             **model_kwargs,
#         )
#         self._model_summary_string = (
#             f"BulkVAE Model with the following params: \nn_latent: {n_latent}"
#         )
#         self.init_params_ = self._get_init_params(locals())

#     @classmethod
#     def setup_anndata(
#         cls,
#         adata: AnnData,
#         batch_key: str | None = None,
#         layer: str | None = None,
#         **kwargs,
#     ) -> AnnData | None:
#         setup_method_args = cls._get_setup_method_args(**locals())
#         anndata_fields = [
#             LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
#             CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
#             # Dummy fields required for VAE class.
#             CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
#             NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
#             CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
#             NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
#         ]
#         adata_manager = AnnDataManager(
#             fields=anndata_fields, setup_method_args=setup_method_args
#         )
#         adata_manager.register_fields(adata, **kwargs)
#         cls.register_manager(adata_manager)


class Encoder(nn.Module):
    def __init__(
        self, input_dim: int, latent_dim: int, hidden_dims: list, dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self, latent_dim: int, output_dim: int, hidden_dims: list, dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class ContrastiveAE(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = [512, 256],
        margin: float = 1.0,
        recon_weight: float = 1.0,
        lr: float = 1e-3,
        dropout: float = 0.2,
        label_key: str = "cell_type",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.triplet_loss = TripletLoss(margin=margin)
        self.recon_loss = nn.MSELoss()
        
        # Add embedding store to the model
        self.embedding_store = EmbeddingStore(metric='cosine')

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x = batch["expression"]
        labels = batch["label"]

        # Get embeddings
        z = self.encoder(x)

        # Calculate losses
        triplet_loss = self.triplet_loss(z, labels)
        recon = self.decoder(z)
        recon_loss = self.recon_loss(recon, x)

        total_loss = self.hparams.recon_weight * recon_loss + triplet_loss

        self.log_dict(
            {
                "train_loss": total_loss,
                "train_recon_loss": recon_loss,
                "train_triplet_loss": triplet_loss,
            }
        )
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )

    def validation_step(self, batch, batch_idx):
        x = batch["expression"]
        labels = batch["label"]

        # Get embeddings
        z = self.encoder(x)

        # Calculate losses
        triplet_loss = self.triplet_loss(z, labels)
        recon = self.decoder(z)
        recon_loss = self.recon_loss(recon, x)

        total_loss = self.hparams.recon_weight * recon_loss + triplet_loss

        self.log_dict(
            {
                "val_loss": total_loss,
                "val_recon_loss": recon_loss,
                "val_triplet_loss": triplet_loss,
            }
        )
        return total_loss

    def on_train_end(self):
        """Automatically build index after training"""
        if self.trainer.datamodule is not None:
            train_adata = self.trainer.datamodule.train_data
            self.build_similarity_index(train_adata)
            
    def build_similarity_index(self, adata: ad.AnnData):
        """Build similarity index from training data"""
        embeddings = self.get_latent_embedding(adata)
        self.embedding_store.build_index(
            embeddings=embeddings,
            labels=adata.obs[self.hparams.label_key].values,
            sample_ids=adata.obs_names.values
        )
    
    def query_similar(
        self,
        query: Union[ad.AnnData, torch.Tensor],
        k: int = 5,
        label_filter: str = None
    ) -> dict:
        """
        Query similar samples from the trained model
        
        Args:
            query: Either AnnData object or tensor of shape (n_samples, n_features)
            k: Number of similar samples to retrieve
            label_filter: Only return samples with this label
            
        Returns:
            Dictionary containing similar samples' information
        """
        if isinstance(query, ad.AnnData):
            query_embedding = self.get_latent_embedding(query)
        else:
            self.eval()
            with torch.no_grad():
                query_embedding = self.encoder(query.to(self.device)).cpu().numpy()
                
        return self.embedding_store.query(query_embedding, k, label_filter)

    def get_latent_embedding(
        self,
        adata: ad.AnnData,
        layer: str = None,
        label_key: str = "cell_type",
        batch_size: int = 256,
        store_in_adata: bool = True,
    ):
        """
        Generate latent embeddings for an AnnData object

        Args:
            adata: AnnData containing expression data
            layer: Layer to use (default: .X)
            label_key: Key for pseudo-labels (will create dummy if missing)
            batch_size: Batch size for inference
            store_in_adata: Whether to store embeddings in adata.obsm['X_latent']

        Returns:
            numpy array of latent embeddings (cells x latent_dim)
        """
        # Create copy to avoid modifying original data
        temp_adata = adata.copy()

        # Add dummy labels if needed
        if label_key not in temp_adata.obs:
            temp_adata.obs[label_key] = "dummy"

        # Create a simple dataset for inference (not triplet-based)
        class SimpleDataset(Dataset):
            def __init__(self, adata, layer=None):
                self.adata = adata
                self.layer = layer

            def __len__(self):
                return self.adata.shape[0]

            def __getitem__(self, idx):
                if self.layer:
                    expr = self.adata.layers[self.layer][idx]
                else:
                    expr = self.adata.X[idx]

                # Handle sparse matrices
                if hasattr(expr, "toarray"):
                    expr = expr.toarray().squeeze()

                return torch.tensor(expr, dtype=torch.float32)

        # Create dataset and dataloader
        dataset = SimpleDataset(temp_adata, layer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # No parallel workers for in-memory data
        )

        # Collect embeddings
        embeddings = []
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                x = batch.to(self.device)
                z = self.encoder(x)
                embeddings.append(z.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        
        if store_in_adata:
            adata.obsm['X_latent'] = embeddings
            
        return embeddings


class PhenotypeAttentionModel(pl.LightningModule):
    """Identifies phenotype-associated niches using full attention"""
    def __init__(
        self,
        input_dim: int,
        sc_embed_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = [512, 256],
        margin: float = 1.0,
        recon_weight: float = 1.0,
        lr: float = 1e-3,
        dropout: float = 0.2,
        num_attention_heads: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Bulk processing components
        self.bulk_encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
        # Attention mechanism
        self.attention = FullCrossAttention(
            embed_dim=latent_dim,
            num_heads=num_attention_heads
        )
        
        # Reconstruction decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
        # Loss components
        self.triplet_loss = TripletLoss(margin=margin)
        self.recon_loss = nn.MSELoss()
        
        # Will be initialized during setup
        self.sc_embeddings = None
        
    def load_pretrained_components(self, contrastive_ae: ContrastiveAE):
        """
        Initialize encoder/decoder from pretrained ContrastiveAE model
        Args:
            contrastive_ae: Pretrained ContrastiveAE model instance
        """
        # Load encoder weights
        self.bulk_encoder.load_state_dict(contrastive_ae.encoder.state_dict())
        
        # Load decoder weights
        self.decoder.load_state_dict(contrastive_ae.decoder.state_dict())
        
        print("Successfully loaded pretrained encoder and decoder weights")

    def forward(self, bulk_x):
        bulk_z = self.bulk_encoder(bulk_x)
        attended_z, attn_weights = self.attention(bulk_z, self.sc_embeddings)
        recon_x = self.decoder(attended_z)
        return attended_z, recon_x, attn_weights

    def setup_sc_embeddings(self, sc_embeddings: torch.Tensor):
        """Register single-cell embeddings as buffer"""
        self.register_buffer('sc_embeddings', sc_embeddings)

    def training_step(self, batch, batch_idx):
        bulk_x, phenotype_labels = batch
        
        # Forward pass
        attended_z, recon_x, _ = self(bulk_x)
        
        # Reconstruction loss
        recon_loss = self.recon_loss(recon_x, bulk_x)
        
        # Triplet loss on phenotype labels
        triplet_loss = self.triplet_loss(attended_z, phenotype_labels)
        
        total_loss = self.hparams.recon_weight * recon_loss + triplet_loss
        
        self.log_dict({
            "train_loss": total_loss,
            "train_recon_loss": recon_loss,
            "train_triplet_loss": triplet_loss,
        })
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=1e-5
        )

    def get_attention_weights(self, bulk_x):
        """Retrieve attention weights for interpretation"""
        with torch.no_grad():
            _, _, attn_weights = self(bulk_x)
        return attn_weights.squeeze(1)  # (batch_size, num_sc_cells)


class GraphContrastiveModel(pl.LightningModule):
    """Contrastive learning model for spatial transcriptomics graphs"""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 256, 
        latent_dim: int = 128, 
        temperature: float = 0.1, 
        recon_weight: float = 0.5,
        feature_drop_rate: float = 0.1,
        edge_drop_rate: float = 0.2,
        lr: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        # GNN encoder
        self.conv1 = GATConv(input_dim, hidden_dim, edge_dim=1)
        self.conv2 = GATConv(hidden_dim, latent_dim, edge_dim=1)
        
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Embedding store for similarity search
        self.embedding_store = EmbeddingStore(metric='cosine')

    def encode(self, x, edge_index, edge_attr, batch=None):
        """Encode graph data to latent space"""
        # First graph convolution
        x = self.conv1(x, edge_index).relu()
        
        # Second graph convolution
        x = self.conv2(x, edge_index)
        
        # If batch indices provided, pool to graph-level embeddings
        if batch is not None:
            return global_mean_pool(x, batch)
        return x

    def forward(self, data):
        """Process a batch of graphs"""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        return self.encode(x, edge_index, edge_attr, batch)

    # def _augment_graph(self, data):
    #     """Apply random augmentations to graph"""
    #     # Create a clone to avoid modifying original
    #     data = data.clone()
        
    #     # Feature dropout (10%)
    #     feature_mask = torch.rand(data.x.size()) > 0.1
    #     data.x = data.x * feature_mask
        
    #     # Edge dropout (20%)
    #     if data.edge_index.size(1) > 0:
    #         edge_mask = torch.rand(data.edge_index.size(1)) > 0.2
    #         data.edge_index = data.edge_index[:, edge_mask]
    #         if data.edge_attr is not None:
    #             data.edge_attr = data.edge_attr[edge_mask]
        
    #     return data

    def _augment_graph(self, graph):
        """Apply biologically plausible augmentations"""
        # Create a clone to avoid modifying original
        g = graph.clone()
        
        # Feature permutation (row-wise shuffle)
        # if self.hparams.feature_drop_rate > 0 and torch.rand(1) < 0.5:
        perm = torch.randperm(g.x.size(0))  # Get random permutation of node indices
        g.x = g.x[perm]  # Shuffle node features while keeping features intact
        
        # # Feature dropout
        # if self.hparams.feature_drop_rate > 0:
        #     drop_mask = torch.rand(g.x.size(1)) < self.hparams.feature_drop_rate
        #     g.x[:, drop_mask] = 0
            
        # # Edge dropping
        # if self.hparams.edge_drop_rate > 0 and g.edge_index.size(1) > 0:
        #     num_edges = g.edge_index.size(1)
        #     keep_mask = torch.rand(num_edges) > self.hparams.edge_drop_rate
        #     g.edge_index = g.edge_index[:, keep_mask]
        #     if g.edge_attr is not None:
        #         g.edge_attr = g.edge_attr[keep_mask]
            
        return g
        
    def _augment_batch(self, batch):
        """Apply augmentations to entire batch"""
        if isinstance(batch, Batch):
            return Batch.from_data_list([self._augment_graph(g) for g in batch.to_data_list()])
        return self._augment_graph(batch)

    def contrastive_loss(self, z1, z2):
        """InfoNCE loss between two views"""
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        logits = torch.mm(z1, z2.T) / self.hparams.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(z1.size(0), device=self.device)
        
        # Compute cross entropy loss (both directions)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2

    def training_step(self, batch, batch_idx):
        # Generate two augmented views
        view1 = self._augment_batch(batch)
        view2 = self._augment_batch(batch)
        
        # Get embeddings
        z1 = self(view1)
        z2 = self(view2)
        
        # Contrastive loss
        cl_loss = self.contrastive_loss(z1, z2)
        
        # Reconstruction loss
        recon = self.decoder(z1)
        recon = recon.reshape(-1)
        recon_loss = F.mse_loss(recon, batch.mean_expression)
        
        # Total loss
        total_loss = cl_loss + self.hparams.recon_weight * recon_loss
        # total_loss = cl_loss
        
        self.log_dict({
            'train_loss': total_loss,
            'contrastive_loss': cl_loss,
            'recon_loss': recon_loss
        })
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Get embeddings
        z = self(batch)
        
        # Reconstruction loss
        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, batch.mean_expression)
        
        self.log('val_recon_loss', recon_loss)
        return recon_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=1e-5
        )
    
    def build_similarity_index(self, dataloader):
        """Build similarity index from a dataloader"""
        self.eval()
        embeddings = []
        labels = []
        sample_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                z = self(batch)
                embeddings.append(z.cpu().numpy())
                
                # Extract center node labels and IDs
                if hasattr(batch, 'y'):
                    labels.append(batch.y.cpu().numpy())
                else:
                    # Use dummy labels if not available
                    labels.append(np.zeros(len(batch)))
                    
                if hasattr(batch, 'center_node_idx'):
                    sample_ids.append(batch.center_node_idx.cpu().numpy())
                else:
                    sample_ids.append(np.arange(len(batch)))
        
        # Concatenate results
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        sample_ids = np.concatenate(sample_ids, axis=0)
        
        # Build index
        self.embedding_store.build_index(embeddings, labels, sample_ids)
        print(f"Built similarity index with {len(embeddings)} embeddings")
