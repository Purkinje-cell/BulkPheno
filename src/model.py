import collections
from typing import Callable, Iterable, Literal, Union, List, Dict, Any


import anndata as ad
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

try:
    import faiss
except ImportError:
    faiss = None
from torch.utils.data import DataLoader, Dataset
from anndata import AnnData
from scvi.nn import Decoder, DecoderSCVI, Embedding, Encoder, FCLayers
from scvi.utils import setup_anndata_dsp
from torch.distributions import Categorical, Distribution, Normal
from torch.distributions import kl_divergence
from torch.distributions import kl_divergence as kl
from torch.nn.functional import one_hot
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GATConv,
    Sequential,
    GCNConv,
    GINConv,
    PNAConv,
    SAGEConv,
    global_add_pool,
    ASAPooling,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix, k_hop_subgraph, subgraph, from_scipy_sparse_matrix, remove_self_loops
from torch_scatter import scatter_add

# from module import BulkVAEModule
# from training_mixin import BasicTrainingMixin
# from utils import broadcast_labels, one_hot
from dataset import ContrastiveBulkDataset, BulkDataset


class FullCrossAttention(nn.Module):
    """Attention using all single-cell embeddings simultaneously"""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, bulk_embeddings, sc_embeddings):
        """
        Args:
            bulk_embeddings: (batch_size, embed_dim)
            sc_embeddings: (num_sc_cells, embed_dim) - fixed for all batches
        """
        # Expand sc embeddings for batch processing
        sc_embeddings = sc_embeddings.unsqueeze(0).repeat(
            bulk_embeddings.size(0), 1, 1
        )  # (B, N, D)

        # Attention computation
        attn_output, attn_weights = self.attention(
            query=bulk_embeddings.unsqueeze(1), key=sc_embeddings, value=sc_embeddings
        )

        return attn_output.squeeze(1), attn_weights  # (B, D), (B, 1, N)


class EmbeddingStore:
    """Stores and indexes latent embeddings for similarity queries"""

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.embeddings = None
        self.index = None
        self.labels = None
        self.sample_ids = None

    def build_index(
        self, embeddings: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray
    ):
        """Build similarity search index"""
        self.embeddings = embeddings
        self.labels = labels
        self.sample_ids = sample_ids

        if faiss is not None:
            # FAISS implementation for production use
            dim = embeddings.shape[1]
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(embeddings)
            else:  # L2
                self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
        else:
            # Fallback to sklearn implementation
            n_neighbors = min(100, len(embeddings) - 1)
            self.index = NearestNeighbors(
                n_neighbors=n_neighbors, metric=self.metric
            ).fit(embeddings)

    def query(
        self, query_embedding: np.ndarray, k: int = 5, label_filter: str = None
    ) -> dict:
        """Find similar samples in the embedding space"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first")

        # Convert to numpy if needed
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()

        if faiss is not None:
            if self.metric == "cosine":
                faiss.normalize_L2(query_embedding)
            distances, indices = self.index.search(query_embedding, k)
        else:
            distances, indices = self.index.kneighbors(query_embedding, n_neighbors=k)

        return {
            "indices": indices,
            "distances": distances,
            "sample_ids": self.sample_ids[indices],
            "labels": self.labels[indices],
        }


class TripletLoss(nn.Module):
    """
    Triplet loss with hard mining using L2 distance
    Parameters
    ----------
    margin: float
        Margin for triplet loss
    
    Forward Inputs
    
    embeddings: torch.Tensor
        Embeddings to compute loss on
    labels: torch.Tensor
        Pseudo-labels for embeddings
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, embeddings, labels):
        """Compute loss using hardest triplets in batch"""
        triplets = self._get_triplets(embeddings, labels)
        if triplets is None or len(triplets) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return self.loss_fn(*triplets)

    def pairwise_distance(self, x, y=None):
        """Compute pairwise L2 distance between tensors"""
        if y is None:
            y = x
        return torch.cdist(x, y, p=2)

    def _get_triplets(self, embeddings, labels):
        # Use L2 distance for mining
        pairwise_dist = self.pairwise_distance(embeddings)

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


class TripletGCLLoss(nn.Module):
    """Triplet loss with hard mining using L2 distance"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


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
        self.embedding_store = EmbeddingStore(metric="cosine")

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
            sample_ids=adata.obs_names.values,
        )

    def query_similar(
        self,
        query: Union[ad.AnnData, torch.Tensor],
        k: int = 5,
        label_filter: str = None,
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
            adata.obsm["X_latent"] = embeddings

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
            embed_dim=latent_dim, num_heads=num_attention_heads
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
        for param in self.bulk_encoder.parameters():
            param.requires_grad = False

        # Load decoder weights
        self.decoder.load_state_dict(contrastive_ae.decoder.state_dict())
        for param in self.decoder.parameters():
            param.requires_grad = False

        print("Successfully loaded pretrained encoder and decoder weights")

    def forward(self, bulk_x):
        bulk_z = self.bulk_encoder(bulk_x)
        attended_z, attn_weights = self.attention(bulk_z, self.sc_embeddings)
        recon_x = self.decoder(attended_z)
        return attended_z, recon_x, attn_weights

    def setup_sc_embeddings(self, sc_embeddings: torch.Tensor):
        """Register single-cell embeddings as buffer"""
        self.sc_embeddings = sc_embeddings

    def training_step(self, batch, batch_idx):
        bulk_x, phenotype_labels = batch

        # Forward pass
        attended_z, recon_x, _ = self(bulk_x)

        # Reconstruction loss
        recon_loss = self.recon_loss(recon_x, bulk_x)

        # Triplet loss on phenotype labels
        triplet_loss = self.triplet_loss(attended_z, phenotype_labels)

        total_loss = self.hparams.recon_weight * recon_loss

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

    def get_attention_weights(self, bulk_x):
        """Retrieve attention weights for interpretation"""
        self.eval()
        with torch.no_grad():
            _, _, attn_weights = self(bulk_x)
        return attn_weights.squeeze(1)  # (batch_size, num_sc_cells)


class BulkEncoderModel(pl.LightningModule):
    """Bulk RNA encoder aligned with GCL model's latent space"""

    def __init__(
        self,
        input_dim: int,
        gcl_model: "GraphContrastiveModel",
        latent_dim: int = 128,
        hidden_dims: list = [512, 256],
        recon_weight: float = 1.0,
        align_weight: float = 0.5,
        lr: float = 1e-3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["gcl_model"])

        # Encoder network
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Initialize decoder from GCL model and freeze

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, gcl_model.hparams.hidden_dim),
            nn.BatchNorm1d(gcl_model.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gcl_model.hparams.hidden_dim, gcl_model.hparams.hidden_dim),
            nn.BatchNorm1d(gcl_model.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gcl_model.hparams.hidden_dim, input_dim),
        )
        
        # Copy weights from the GCL model's decoder
        with torch.no_grad():
            # Get state dict from GCL model's decoder
            gcl_decoder_state_dict = gcl_model.decoder.state_dict()
            
            # Initialize our decoder with these weights
            # Need to ensure compatible architecture
            decoder_state_dict = self.decoder.state_dict()
            
            # Only copy parameters with matching names and shapes
            for name, param in gcl_decoder_state_dict.items():
                if name in decoder_state_dict and decoder_state_dict[name].shape == param.shape:
                    decoder_state_dict[name].copy_(param)
            
            # Load the modified state dict
            self.decoder.load_state_dict(decoder_state_dict)
            
            # Freeze decoder parameters
            for param in self.decoder.parameters():
                param.requires_grad = False


        # Store reference to GCL model
        # self.gcl_encoder = gcl_model
        # self.gcl_encoder.eval()
        # for param in self.gcl_encoder.parameters():
        #     param.requires_grad = False

        # Loss components
        self.recon_loss = nn.MSELoss()
        self.align_loss = nn.MSELoss()
        self.contrastive_loss = TripletLoss(margin=1.0)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

    def training_step(self, batch, batch_idx):
        if batch["is_real"]:
            return self._real_batch_step(batch)
        else:
            return self._pseudo_batch_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        if batch["is_real"]:
            return self._real_batch_step(batch)
        else:
            return self._pseudo_batch_step(batch, stage="val")

    def set_graph_embedding(self, graph_embedding: torch.Tensor):
        self.graph_embedding = graph_embedding

    def _real_batch_step(self, batch):
        x = batch["expression"]
        label = batch["phenotype"]
        z, recon = self(x)
        triplet_loss = self.contrastive_loss(z, label)
        recon_loss = self.recon_loss(recon, x) * self.hparams.recon_weight
        loss = triplet_loss + recon_loss

        self.log_dict(
            {
                "train_real_recon_loss": recon_loss,
                "train_real_triplet_loss": triplet_loss,
                "train_real_total_loss": loss,
            }
        )
        return loss

    def _pseudo_batch_step(self, batch, stage="train"):
        x = batch["expression"]
        graph_indices = batch["graph_indices"]

        # Get bulk embeddings
        z_bulk, recon = self(x)

        # Get corresponding GCL embeddings
        with torch.no_grad():
            z_gcl = [self.graph_embedding[idx] for idx in graph_indices]
            z_gcl = torch.stack(z_gcl)
            z_gcl = z_gcl.to(self.device)

        # Calculate losses
        recon_loss = self.recon_loss(recon, x) * self.hparams.recon_weight
        align_loss = self.align_loss(z_bulk, z_gcl) * self.hparams.align_weight
        # triplet_loss = self.contrastive_loss(z_bulk, batch["label"])
        total_loss = recon_loss + align_loss
        # total_loss = recon_loss

        self.log_dict(
            {
                f"{stage}_pseudo_recon_loss": recon_loss,
                f"{stage}_align_loss": align_loss,
                f"{stage}_pseudo_total_loss": total_loss,
            }
        )
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.encoder.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )

    def get_latent_embedding(self, adata: ad.AnnData):
        """Get latent embeddings for bulk RNA data"""
        self.eval()
        dataset = BulkDataset(adata=adata)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        embeddings = []
        with torch.no_grad():
            for batch in loader:
                x = batch["expression"].to(self.device)
                z, _ = self(x)
                embeddings.append(z.cpu())

        return torch.cat(embeddings, dim=0).numpy()

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Dot product similarity
        sim = torch.einsum("nc,nc->n", z_i, z_j) / self.temperature

        # Loss calculation
        loss = -torch.log(
            torch.exp(sim) / (torch.exp(sim).sum() + 1e-6)
        ).mean()  # Numerically stable

        return loss

class GraphContrastiveModel(pl.LightningModule):
    """Contrastive learning model for spatial transcriptomics graphs"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        margin: float = 1.0,
        recon_weight: float = 0.5,
        feature_drop_rate: float = 0.1,
        edge_drop_rate: float = 0.2,
        conv_layer: str = "GAT",
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        # GNN encoder
        self.conv_layer = conv_layer
        self.automatic_optimization = False
        if conv_layer == "GAT":
            self.conv1 = GATConv(input_dim, hidden_dim, edge_dim=1)
            self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=1)
        elif conv_layer == "GCN":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif conv_layer == "GIN":
            self.conv1 = GINConv(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.conv2 = GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        elif conv_layer == "PNA":
            self.conv1 = PNAConv(
                input_dim, hidden_dim, aggregators=["mean", "min", "max", "std"]
            )
            self.conv2 = PNAConv(
                hidden_dim, hidden_dim, aggregators=["mean", "min", "max", "std"]
            )

        self.project = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
            
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

        self.regressor = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Triplet loss with hard mining
        self.triplet_loss = TripletGCLLoss(margin=margin)

        # Embedding store for similarity search
        self.embedding_store = EmbeddingStore(metric="cosine")
        self.info_nce_loss = InfoNCELoss(temperature=0.5)

    def encode(self, x, edge_index, edge_attr, batch=None):
        """Encode graph data to latent space"""
        # First graph convolution
        if self.conv_layer in ['GCN', 'GIN']:
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
        else: 
            x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
            # Second graph convolution
            x = self.conv2(x, edge_index, edge_attr=edge_attr).relu()

        x = self.project(x)

        # If batch indices provided, pool to graph-level embeddings
        if batch is not None:
            return global_mean_pool(x, batch)
        return x

    def forward(self, data):
        """Process a batch"""
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        z = self.encode(x, edge_index, edge_attr, batch)
        return z

    def _augment_graph(self, graph):
        """Augmentation preserving center node ID"""
        g = graph.clone()

        # Feature permutation (preserve center node)
        # if torch.rand(1) < 0.5:
        perm = torch.randperm(g.x.size(0))
        # Keep center node in first position
        g.x = g.x[perm]

        # # Feature dropout
        drop_mask = torch.rand_like(g.x) < self.hparams.feature_drop_rate
        g.x[drop_mask] = 0

        # # Edge dropping
        # if g.edge_index.size(1) > 0:
        #     keep_mask = torch.rand(g.edge_index.size(1)) > self.hparams.edge_drop_rate
        #     g.edge_index = g.edge_index[:, keep_mask]
        #     if g.edge_attr is not None:
        #         g.edge_attr = g.edge_attr[keep_mask]

        return g

    def _augment_batch(self, batch):
        """Apply augmentations to entire batch"""
        if isinstance(batch, Batch):
            return Batch.from_data_list(
                [self._augment_graph(g) for g in batch.to_data_list()]
            )
        return self._augment_graph(batch)

    def _get_hard_triplets(self, embeddings, center_nodes):
        """Mine hard triplets based on center node identity using L2 distance"""
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        triplets = []
        for i in range(len(embeddings)):
            # Positive: same center node (original + augmented)
            pos_mask = center_nodes == center_nodes[i]
            pos_mask[i] = False  # Exclude self

            if not pos_mask.any():
                continue

            # Hardest positive (furthest)
            hardest_pos = torch.argmax(pairwise_dist[i][pos_mask]).item()
            pos_idx = torch.where(pos_mask)[0][hardest_pos].item()

            # Negative: different center node
            neg_mask = center_nodes != center_nodes[i]
            if not neg_mask.any():
                continue

            # Hardest negative (closest)
            hardest_neg = torch.argmin(pairwise_dist[i][neg_mask]).item()
            neg_idx = torch.where(neg_mask)[0][hardest_neg].item()

            triplets.append((i, pos_idx, neg_idx))

        if not triplets:
            return None

        indices = torch.tensor(triplets, device=embeddings.device)
        return (
            embeddings[indices[:, 0]],
            embeddings[indices[:, 1]],
            embeddings[indices[:, 2]],
        )

    def sample_graphs(self, data:Data, ratio:float=0.8):
        """Sample a batch of graphs for triplet mining"""
        # 确保sample_indices在与data.edge_index相同的设备上
        device = data.edge_index.device
        sample_indices = torch.randperm(data.num_nodes, device=device)
        split_idx = int(ratio * len(sample_indices))
        sample_indices = sample_indices[:split_idx]
        
        # 确保所有张量在同一设备上
        edge_index, edge_attr = subgraph(sample_indices, data.edge_index, data.edge_attr, relabel_nodes=True)
        x = data.x[sample_indices]
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        

    def training_step(self, batch: Batch, batch_idx):
        # Generate augmented view
        # augmented_batch = self._augment_batch(batch)
        regressor_optimizer, graph_encoder_decoder_optimizer = self.optimizers()
        data_list = batch.to_data_list()
        global_views_1 = []
        global_views_2 = []
        local_views_1 = []
        local_views_2 = []
        for data in data_list:
            global_view_1 = self.sample_graphs(data)
            global_view_2 = self.sample_graphs(data)
            local_view_1 = self.sample_graphs(data, ratio=0.2)
            local_view_2 = self.sample_graphs(data, ratio=0.2)
            global_views_1.append(global_view_1)
            global_views_2.append(global_view_2)
            local_views_1.append(local_view_1)
            local_views_2.append(local_view_2)
        
        global_views_1 = Batch.from_data_list(global_views_1)
        global_views_2 = Batch.from_data_list(global_views_2)
        local_views_1 = Batch.from_data_list(local_views_1)
        local_views_2 = Batch.from_data_list(local_views_2)
        z_global_1 = self(global_views_1)
        z_global_2 = self(global_views_2)
        z_local_1 = self(local_views_1) # (batch_size, latent_dim)
        z_local_2 = self(local_views_2) # (batch_size, latent_dim)
        z_local_2_neg = z_local_2[torch.randperm(z_local_2.size(0))]
        z_local_mix = torch.cat([z_local_1, z_local_2], dim=1) # (batch_size, 2 * latent_dim)
        z_local_mix_neg = torch.cat([z_local_1, z_local_2_neg], dim=1) # (batch_size, 2 * latent_dim)
        mix_score = torch.mean(self.regressor(z_local_mix)) # (1)
        mix_neg_score = torch.mean(self.regressor(z_local_mix_neg)) # (1)
        loss_ll = -(mix_score - mix_neg_score)
        regressor_optimizer.zero_grad()
        self.manual_backward(loss_ll)
        regressor_optimizer.step()
        

        # Get embeddings for both original and augmented
        z_orig = self(batch)
        # z_aug = self(augmented_batch)

        # Combine embeddings and center nodes
        # combined_z = torch.cat([z_orig, z_aug], dim=0)
        # combined_centers = torch.cat(
        #     [batch.center_node_idx, batch.center_node_idx], dim=0
        # )

        z_local_1 = self(local_views_1) # (batch_size, latent_dim)
        z_local_2 = self(local_views_2) # (batch_size, latent_dim)
        z_local_mix = torch.cat([z_local_1, z_local_2], dim=1) # (batch_size, 2 * latent_dim)
        ll_loss = torch.mean(self.regressor(z_local_mix), dim=0) # (1)
        gg_loss = self.info_nce_loss(z_global_1, z_global_2)
        gl_loss = (self.info_nce_loss(z_global_1, z_local_2) + self.info_nce_loss(z_global_2, z_local_1) + self.info_nce_loss(z_global_1, z_local_1) + self.info_nce_loss(z_global_2, z_local_2)) / 4

        # Mine hard triplets
        # triplets = self._get_hard_triplets(combined_z, combined_centers)

        # Calculate triplet loss
        # if triplets is None:
        #     cl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        # else:
        #     cl_loss = self.triplet_loss(*triplets)

        # Reconstruction loss
        recon = self.decoder(z_orig)
        recon = recon.reshape(-1)
        recon_loss = F.mse_loss(recon, batch.mean_expression)
        infor_loss = gg_loss + 0.8 * gl_loss + 0.4 * ll_loss

        # Total loss
        total_loss = infor_loss + self.hparams.recon_weight * recon_loss

        graph_encoder_decoder_optimizer.zero_grad()
        self.manual_backward(total_loss)
        graph_encoder_decoder_optimizer.step()
        self.log_dict(
            {
                "train_loss": total_loss,
                "info_loss": infor_loss,
                # "triplet_loss": cl_loss,
                "recon_loss": recon_loss,
            }
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Generate augmented view
        augmented_batch = self._augment_batch(batch)

        # Get embeddings for both original and augmented
        z_orig = self(batch)
        z_aug = self(augmented_batch)

        # Combine embeddings and center nodes
        combined_z = torch.cat([z_orig, z_aug], dim=0)
        combined_centers = torch.cat(
            [batch.center_node_idx, batch.center_node_idx], dim=0
        )

        # Mine hard triplets
        triplets = self._get_hard_triplets(combined_z, combined_centers)

        # Calculate triplet loss
        if triplets is None:
            cl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            cl_loss = self.triplet_loss(*triplets)

        # Reconstruction loss
        recon = self.decoder(z_orig)
        recon = recon.reshape(-1)
        recon_loss = F.mse_loss(recon, batch.mean_expression)

        # Total loss
        total_loss = cl_loss + self.hparams.recon_weight * recon_loss

        self.log_dict(
            {
                "val_loss": total_loss,
                "val_triplet_loss": cl_loss,
                "val_recon_loss": recon_loss,
            }
        )
        return total_loss

    def configure_optimizers(self):
        regressor_optim = torch.optim.Adam(
            self.regressor.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )
        graph_encoder_decoder_optim = torch.optim.Adam(
            list(self.conv1.parameters()) + list(self.conv2.parameters()) + list(self.project.parameters()) + list(self.decoder.parameters()),
            lr=self.hparams.lr,
            weight_decay=1e-5,
        )
        return [regressor_optim, graph_encoder_decoder_optim]
    
    def get_graph_embedding(self, adata: ad.AnnData, hops):
        """Get graph embeddings for bulk RNA data"""
        self.eval()

        edge_index, edge_attr = from_scipy_sparse_matrix(
            adata.obsp["distances"]
        )
        mask = edge_attr < 50
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        embeddings = []
        valid_idx = []
        with torch.no_grad():
            for node_idx in tqdm(range(adata.shape[0]), desc = "Processing nodes"):
                subset, edge_index_sub, _, edge_mask = k_hop_subgraph(
                    node_idx,
                    hops,
                    edge_index,
                    num_nodes=adata.shape[0],
                    relabel_nodes=True,
                )
                
                # Skip nodes with no neighbors
                if subset.shape[0] <= 1:
                    print(f'Node {node_idx} has no neighbors')
                    embeddings.append(torch.zeros(1, self.hparams.latent_dim))
                    
                if hasattr(adata.X, "toarray"):
                    features = torch.tensor(
                        adata.X[subset].toarray(), dtype=torch.float
                    )
                else:
                    features = torch.tensor(adata.X[subset], dtype=torch.float)
                
                # Ensure features are on the same device as the model
                features = features.to(self.device)
                edge_index_sub = edge_index_sub.to(self.device)
                edge_attr_sub = edge_attr[edge_mask].to(self.device)
                
                # Get graph embedding
                z = self.encode(features, edge_index_sub, edge_attr_sub)
                z = torch.mean(z, dim=0, keepdim=True)  # (1, latent_dim)
                embeddings.append(z.cpu())
                valid_idx.append(node_idx)
        embeddings = torch.concat(embeddings, dim=0) # (num_nodes, latent_dim)
        print(f"Graph embeddings shape: {embeddings.shape}")
        return embeddings, valid_idx # 

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
                if batch.y is not None:
                    labels.append(batch.y.cpu().numpy())
                else:
                    # Use dummy labels if not available
                    labels.append(np.zeros(len(batch)))

                if hasattr(batch, "center_node_idx"):
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
