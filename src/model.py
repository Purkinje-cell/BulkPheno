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


class ContrastiveAE(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = [512, 256],
        margin: float = 1.0,
        recon_weight: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder network
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder network
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)
        
        # Triplet loss
        triplet_loss = self._compute_triplet_loss(z, labels)
        
        # Total loss
        total_loss = self.hparams.recon_weight * recon_loss + triplet_loss
        
        self.log('train_loss', total_loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_triplet_loss', triplet_loss)
        return total_loss

    def _compute_triplet_loss(self, embeddings, labels):
        pairwise_dist = F.pairwise_distance(
            embeddings.unsqueeze(1), 
            embeddings.unsqueeze(0), 
            p=2
        )
        
        # Create mask for positive and negative pairs
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        eye_mask = ~torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)
        
        # Hard negative mining
        pos_dist = pairwise_dist[label_mask & eye_mask]
        neg_dist = pairwise_dist[~label_mask]
        
        if len(pos_dist) == 0 or len(neg_dist) == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        hardest_pos = pos_dist.max()
        hardest_neg = neg_dist.min()
        return F.relu(hardest_pos - hardest_neg + self.hparams.margin)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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
