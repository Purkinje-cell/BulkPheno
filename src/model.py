import collections
from typing import Callable, Iterable, Literal


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
from torch_geometric.data import Data
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

from distribution import NegativeBinomial
from module import BulkVAEModule
from training_mixin import BasicTrainingMixin
from utils import broadcast_labels, one_hot


class BulkVAE(UnsupervisedTrainingMixin, VAEMixin, BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.module = BulkVAEModule(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_latent=n_latent,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"BulkVAE Model with the following params: \nn_latent: {n_latent}"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        layer: str | None = None,
        **kwargs,
    ) -> AnnData | None:
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.net(z)

class TripletSelector(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        pairwise_dist = F.pairwise_distance(
            embeddings.unsqueeze(1), 
            embeddings.unsqueeze(0), 
            p=2
        )
        
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        eye_mask = ~torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
        
        pos_dist = pairwise_dist[label_mask & eye_mask]
        neg_dist = pairwise_dist[~label_mask]
        
        if len(pos_dist) == 0 or len(neg_dist) == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        hardest_pos = pos_dist.max()
        hardest_neg = neg_dist.min()
        return F.relu(hardest_pos - hardest_neg + self.margin)

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
        augment_params: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Default augmentation parameters
        self.augment_params = augment_params or {
            'noise_scale': 0.1,
            'mixup_alpha': 0.3,
            'subsample_prob': 0.05,
            'perturb_scale': 0.2,
            'use_augmentation': True,
            'augment_prob': 0.7  # Probability of applying augmentation
        }
        
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self.triplet_selector = TripletSelector(margin=margin)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def gaussian_noise_augmentation(self, x):
        """Add log-normal noise to simulate technical variability"""
        device = x.device
        noise_scale = self.augment_params['noise_scale']
        noise = torch.exp(torch.randn_like(x) * noise_scale)
        return x * noise

    def depth_subsampling(self, x):
        """Simulate library size variation through binomial subsampling"""
        subsample_prob = self.augment_params['subsample_prob']
        # Ensure x is non-negative for binomial sampling
        x_pos = torch.clamp(x, min=0)
        return torch.distributions.Binomial(total_count=x_pos, probs=1-subsample_prob).sample()

    def biological_mixup(self, x, labels):
        """Create convex combinations of samples to simulate mixed cell populations"""
        alpha = self.augment_params['mixup_alpha']
        indices = torch.randperm(x.size(0))
        shuffled_x = x[indices]
        
        # Generate mixing coefficient from beta distribution
        lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
        mixed_x = lam * x + (1 - lam) * shuffled_x
        
        # For metric learning, keep original labels
        return mixed_x, labels

    def pathway_perturbation(self, x, gene_sets=None):
        """Up/downregulate groups of genes in coordinated pathways"""
        # If no gene sets provided, simulate random pathways
        if gene_sets is None:
            n_genes = x.shape[1]
            n_pathways = min(5, n_genes // 10)  # Create a few random pathways
            gene_sets = []
            for _ in range(n_pathways):
                pathway_size = torch.randint(5, max(6, n_genes // 5), (1,)).item()
                gene_indices = torch.randperm(n_genes)[:pathway_size]
                gene_sets.append(gene_indices.tolist())
        
        perturb_scale = self.augment_params['perturb_scale']
        for pathway_genes in gene_sets:
            mask = torch.zeros_like(x)
            mask[:, pathway_genes] = 1
            # Generate coherent perturbation direction for the pathway
            perturbation = 1 + torch.randn(1).to(x.device) * perturb_scale
            x = x * (1 + mask * perturbation)
        return x

    def apply_augmentations(self, x, labels):
        """Apply a series of biologically-inspired augmentations"""
        if not self.augment_params.get('use_augmentation', True):
            return x, labels
            
        # Apply augmentations with some probability
        if torch.rand(1).item() < self.augment_params.get('augment_prob', 0.7):
            # Apply in order of biological plausibility
            x = self.gaussian_noise_augmentation(x)
            
            # Apply depth subsampling only if data is count-like (non-negative)
            if torch.all(x >= 0):
                x = self.depth_subsampling(x)
                
            # Apply mixup augmentation
            x, labels = self.biological_mixup(x, labels)
            
            # Apply pathway perturbation
            x = self.pathway_perturbation(x)
            
        return x, labels

    def training_step(self, batch, batch_idx):
        x, labels = batch
        
        # Apply biologically-inspired augmentations
        x_aug, labels_aug = self.apply_augmentations(x, labels)
        
        # Use augmented data for training
        z = self.encoder(x_aug)
        x_hat = self.decoder(z)
        
        # Calculate loss against original data to maintain biological fidelity
        recon_loss = F.mse_loss(x_hat, x)
        triplet_loss = self.triplet_selector(z, labels_aug)
        total_loss = self.hparams.recon_weight * recon_loss + triplet_loss
        
        self.log_dict({
            'train_loss': total_loss,
            'train_recon_loss': recon_loss,
            'train_triplet_loss': triplet_loss
        })
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Store the last batch for potential latent interpolation"""
        self.last_batch = batch

    def latent_interpolation(self, batch_size=8):
        """Generate new samples via latent space interpolation"""
        # Get a batch of data from the training set
        if not hasattr(self, 'last_batch'):
            return None, None
            
        x, labels = self.last_batch
        if len(x) < 2:
            return None, None
            
        with torch.no_grad():
            z = self.encoder(x)
            
            # Create interpolation points between consecutive latent vectors
            interp_weights = torch.rand(len(z)-1, 1, device=z.device)
            z_interp = torch.lerp(z[:-1], z[1:], interp_weights)
            
            # Generate interpolated samples
            x_interp = self.decoder(z_interp)
            
            # For labels, take the first sample's label (arbitrary choice)
            labels_interp = labels[:-1]
            
        return x_interp, labels_interp

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        label_key: str,
        layer: str = None,
        **kwargs,
    ):
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, label_key),
        ]
        
        adata_manager = AnnDataManager(
            fields=anndata_fields, 
            setup_method_args={'layer': layer, 'label_key': label_key}
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
