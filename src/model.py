import collections
from typing import Iterable

from networkx import subgraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_scatter import scatter_add

from distribution import NegativeBinomial
from utils import broadcast_labels, one_hot


class FCLayers(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_relu: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out, bias=bias),
                            nn.BatchNorm1d(n_out) if use_batch_norm else None,
                            nn.ReLU() if use_relu else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int):
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(
            cat_list
        ), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (
                n_cat and cat is None
            ), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


class Encoder(nn.Module):
    """
    Bulk RNA seq data encoder
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
    ):
        super(Encoder, self).__init__()

        self.encoder = FCLayers(
            n_input, n_hidden, n_cat_list, n_layers, n_hidden, dropout_rate
        )
        self.mean_encoder = nn.Linear(n_hidden, n_latent)
        self.var_encoder = nn.Linear(n_hidden, n_latent)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, *cat_list):
        q = self.encoder(x, *cat_list)
        mu = self.mean_encoder(q)
        var = torch.exp(self.var_encoder(q)) + 1e-4
        z = self.reparameterize(mu, var)
        return z, mu, var


class Decoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_output: int = 128,
        n_layers: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
    ):
        super(Decoder, self).__init__()
        self.n_cat_list = n_cat_list
        self.decoder = FCLayers(
            n_input, n_hidden, n_cat_list, n_layers, n_hidden, dropout_rate=0
        )

        # mean gamma for Negative Binomial
        self.gamma_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion for Negative Binomial
        self.dispersion_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, z: torch.Tensor, library_size: torch.Tensor, *cat_list: torch.Tensor
    ):
        hidden = self.decoder(z, *cat_list)
        gamma = self.gamma_decoder(hidden)
        rate = torch.exp(library_size) * gamma
        dispersion = None
        return gamma, dispersion, rate


class Classifier(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int = 32,
        n_labels: int = 5,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            FCLayers(
                n_input, n_hidden, None, n_layers, n_hidden, dropout_rate=dropout_rate
            ),
            nn.Linear(n_hidden, n_labels),
        )

    def forward(self, x):
        return self.classifier(x)


class SpatialEncoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class BulkVAE(nn.Module):
    """
    Bulk RNA seq data VAE model, consider batch id and library size for bulk RNA seq data, also consider label for semi-supervised learning
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        log_norm: bool = True,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
    ):
        super(BulkVAE, self).__init__()
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate
        self.log_norm = log_norm
        self.dispersion = nn.Parameter(torch.randn(n_input))

        self.z_encoder = Encoder(
            n_input, n_hidden, n_latent, n_layers, None, dropout_rate
        )
        self.decoder = Decoder(
            n_latent, n_hidden, n_input, n_layers, [n_batch], dropout_rate
        )
        self.classifier = Classifier(
            n_latent, n_hidden, n_labels, n_layers, dropout_rate
        )
        self.l_encoder = Encoder(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=1,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
        )

    def sample_z(self, x, y=None):
        if self.log_norm:
            x = torch.log1p(x)
        z, mu, log_var = self.z_encoder(x)
        return z

    def get_latents(self, x, y=None):
        if self.log_norm:
            x = torch.log1p(x)
        z, mu_z, log_var_z = self.z_encoder(x)
        return z

    def recon_loss(self, x, dispersion, rate):
        reconst_loss = (
            -NegativeBinomial(mu=rate, theta=dispersion).log_prob(x).sum(dim=-1)
        )
        return reconst_loss

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x
        if self.log_norm:
            x_ = torch.log1p(x)

        z, qz_m, qz_v = self.z_encoder(x_)
        l, ql_m, ql_v = self.l_encoder(x_)
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.shape[0], qz_m.shape[1]))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.shape[0], qz_v.shape[1]))
            z = Normal(qz_m, qz_v.sqrt()).sample()
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.shape[0], ql_m.shape[1]))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.shape[0], ql_v.shape[1]))
            l = Normal(ql_m, ql_v.sqrt()).sample()

        gamma, dispersion, rate = self.decoder(z, l, batch_index)

        dispersion = self.dispersion
        dispersion = torch.exp(dispersion)

        return dict(
            gamma=gamma,
            dispersion=dispersion,
            rate=rate,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            l=l,
        )

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        """
        Args:
            x: torch.Tensor, shape (batch_size, n_input)
            local_l_mean: torch.Tensor, shape (batch_size, 1), this is the local mean of library size for each sample
            local_l_var: torch.Tensor, shape (batch_size, 1), this is the local variance of library size for each sample
            batch_index: torch.Tensor, shape (batch_size, n_batch), this is the batch index for each sample
            y: torch.Tensor, shape (batch_size, n_labels), this is the label for each sample
        """
        is_labelled = y is not None
        outputs = self.inference(x, batch_index)
        dispersion = outputs["dispersion"]
        rate = outputs["rate"]
        recon_loss = self.recon_loss(x, dispersion, rate)
        z = outputs["z"]

        # KL Divergence for z and l
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        mean = torch.zeros_like(qz_m)
        std = torch.ones_like(qz_v)
        kl_z = kl_divergence(Normal(qz_m, qz_v.sqrt()), Normal(mean, std)).sum(dim=1)

        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        kl_l = kl_divergence(
            Normal(ql_m, ql_v.sqrt()), Normal(local_l_mean, local_l_var)
        ).sum(dim=1)

        logits = self.classifier(z)
        probs = F.softmax(logits, dim=-1)
        if not is_labelled:
            return recon_loss, kl_z, kl_l, probs

        # classification_loss = F.cross_entropy(logits, y, reduction="sum", weight=torch.tensor([10, 1], dtype=torch.float32))
        classification_loss = F.cross_entropy(logits, y, reduction="sum")

        return recon_loss, kl_z, kl_l, classification_loss, probs


class CrossAttention(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(CrossAttention, self).__init__()
        self.fc_q = nn.Linear(n_input, n_hidden)
        self.fc_k = nn.Linear(n_input, n_hidden)
        self.fc_v = nn.Linear(n_input, n_hidden)
        self.fc_o = nn.Linear(n_hidden, n_output)

    def forward(self, x, y):
        q = self.fc_q(x)
        k = self.fc_k(y)
        v = self.fc_v(y)
        attention = F.softmax(
            torch.bmm(q, k.transpose(1, 2))
            / torch.sqrt(torch.tensor(k.size(-1)).float()),
            dim=-1,
        )
        out = torch.bmm(attention, v)
        out = self.fc_o(out)
        return out


class FCLayers(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_relu: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out, bias=bias),
                            nn.BatchNorm1d(n_out) if use_batch_norm else None,
                            nn.ReLU() if use_relu else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int):
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(
            cat_list
        ), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (
                n_cat and cat is None
            ), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


class Encoder(nn.Module):
    """
    Bulk RNA seq data encoder
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
    ):
        super(Encoder, self).__init__()

        self.encoder = FCLayers(
            n_input, n_hidden, n_cat_list, n_layers, n_hidden, dropout_rate
        )
        self.mean_encoder = nn.Linear(n_hidden, n_latent)
        self.var_encoder = nn.Linear(n_hidden, n_latent)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, *cat_list):
        q = self.encoder(x, *cat_list)
        mu = self.mean_encoder(q)
        var = torch.exp(self.var_encoder(q)) + 1e-4
        z = self.reparameterize(mu, var)
        return z, mu, var


class Decoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_output: int = 128,
        n_layers: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
    ):
        super(Decoder, self).__init__()
        self.n_cat_list = n_cat_list
        self.decoder = FCLayers(
            n_input, n_hidden, n_cat_list, n_layers, n_hidden, dropout_rate=0
        )

        # mean gamma for Negative Binomial
        self.gamma_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion for Negative Binomial
        self.dispersion_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, z: torch.Tensor, library_size: torch.Tensor, *cat_list: torch.Tensor
    ):
        hidden = self.decoder(z, *cat_list)
        gamma = self.gamma_decoder(hidden)
        rate = torch.exp(library_size) * gamma
        dispersion = None
        return gamma, dispersion, rate


class Classifier(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int = 32,
        n_labels: int = 5,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            FCLayers(
                n_input, n_hidden, None, n_layers, n_hidden, dropout_rate=dropout_rate
            ),
            nn.Linear(n_hidden, n_labels),
        )

    def forward(self, x):
        return self.classifier(x)


class SpatialEncoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class BulkVAE(nn.Module):
    """
    Bulk RNA seq data VAE model, consider batch id and library size for bulk RNA seq data, also consider label for semi-supervised learning
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        log_norm: bool = True,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
    ):
        super(BulkVAE, self).__init__()
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate
        self.log_norm = log_norm
        self.dispersion = nn.Parameter(torch.randn(n_input))

        self.z_encoder = Encoder(
            n_input, n_hidden, n_latent, n_layers, None, dropout_rate
        )
        self.decoder = Decoder(
            n_latent, n_hidden, n_input, n_layers, [n_batch], dropout_rate
        )
        self.classifier = Classifier(
            n_latent, n_hidden, n_labels, n_layers, dropout_rate
        )
        self.l_encoder = Encoder(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=1,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
        )

    def sample_z(self, x, y=None):
        if self.log_norm:
            x = torch.log1p(x)
        z, mu, log_var = self.z_encoder(x)
        return z

    def get_latents(self, x, y=None):
        if self.log_norm:
            x = torch.log1p(x)
        z, mu_z, log_var_z = self.z_encoder(x)
        return z

    def recon_loss(self, x, dispersion, rate):
        reconst_loss = (
            -NegativeBinomial(mu=rate, theta=dispersion).log_prob(x).sum(dim=-1)
        )
        return reconst_loss

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x
        if self.log_norm:
            x_ = torch.log1p(x)

        z, qz_m, qz_v = self.z_encoder(x_)
        l, ql_m, ql_v = self.l_encoder(x_)
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.shape[0], qz_m.shape[1]))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.shape[0], qz_v.shape[1]))
            z = Normal(qz_m, qz_v.sqrt()).sample()
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.shape[0], ql_m.shape[1]))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.shape[0], ql_v.shape[1]))
            l = Normal(ql_m, ql_v.sqrt()).sample()

        gamma, dispersion, rate = self.decoder(z, l, batch_index)

        dispersion = self.dispersion
        dispersion = torch.exp(dispersion)

        return dict(
            gamma=gamma,
            dispersion=dispersion,
            rate=rate,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            l=l,
        )

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        """
        Args:
            x: torch.Tensor, shape (batch_size, n_input)
            local_l_mean: torch.Tensor, shape (batch_size, 1), this is the local mean of library size for each sample
            local_l_var: torch.Tensor, shape (batch_size, 1), this is the local variance of library size for each sample
            batch_index: torch.Tensor, shape (batch_size, n_batch), this is the batch index for each sample
            y: torch.Tensor, shape (batch_size, n_labels), this is the label for each sample
        """
        is_labelled = y is not None
        outputs = self.inference(x, batch_index)
        dispersion = outputs["dispersion"]
        rate = outputs["rate"]
        recon_loss = self.recon_loss(x, dispersion, rate)
        z = outputs["z"]

        # KL Divergence for z and l
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        mean = torch.zeros_like(qz_m)
        std = torch.ones_like(qz_v)
        kl_z = kl_divergence(Normal(qz_m, qz_v.sqrt()), Normal(mean, std)).sum(dim=1)

        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        kl_l = kl_divergence(
            Normal(ql_m, ql_v.sqrt()), Normal(local_l_mean, local_l_var)
        ).sum(dim=1)

        logits = self.classifier(z)
        probs = F.softmax(logits, dim=-1)
        if not is_labelled:
            return recon_loss, kl_z, kl_l, probs

        # classification_loss = F.cross_entropy(logits, y, reduction="sum", weight=torch.tensor([10, 1], dtype=torch.float32))
        classification_loss = F.cross_entropy(logits, y, reduction="sum")

        return recon_loss, kl_z, kl_l, classification_loss, probs


class CrossAttention(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(CrossAttention, self).__init__()
        self.fc_q = nn.Linear(n_input, n_hidden)
        self.fc_k = nn.Linear(n_input, n_hidden)
        self.fc_v = nn.Linear(n_input, n_hidden)
        self.fc_o = nn.Linear(n_hidden, n_output)

    def forward(self, x, y):
        q = self.fc_q(x)
        k = self.fc_k(y)
        v = self.fc_v(y)
        attention = F.softmax(
            torch.bmm(q, k.transpose(1, 2))
            / torch.sqrt(torch.tensor(k.size(-1)).float()),
            dim=-1,
        )
        out = torch.bmm(attention, v)
        out = self.fc_o(out)
        return out


class SAGE(nn.Module):
    def __init__(self, gcn_first, gcn_second, fc_1, fc_2, number_of_features, gnn):
        super(SAGE, self).__init__()
        self.gnn = gnn
        self.gcn_first = gcn_first
        self.gcn_second = gcn_second
        self.fc_1 = fc_1
        self.fc_2 = fc_2
        self.number_of_features = number_of_features
        self._setup()
        self.mseloss = nn.MSELoss()
        self.relu = nn.ReLU()

    def _setup(self):
        if self.gnn == "GCN":
            self.graph_convolution_1 = GCNConv(self.number_of_features, self.gcn_first)
            self.graph_convolution_2 = GCNConv(self.gcn_first, self.gcn_second)
        elif self.gnn == "GIN":
            self.graph_convolution_1 = GINConv(
                nn.Sequential(
                    nn.Linear(self.number_of_features, self.gcn_first),
                    nn.ReLU(),
                    nn.Linear(self.gcn_first, self.gcn_first),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.gcn_first),
                ),
                train_eps=False,
            )
            self.graph_convolution_2 = GINConv(
                nn.Sequential(
                    nn.Linear(self.gcn_first, self.gcn_second),
                    nn.ReLU(),
                    nn.Linear(self.gcn_second, self.gcn_second),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.gcn_second),
                ),
                train_eps=False,
            )
        elif self.gnn == "GAT":
            self.graph_convolution_1 = GATConv(self.number_of_features, self.gcn_first, heads=2)
            self.graph_convolution_2 = GATConv(2 * self.gcn_first, self.gcn_second)
        elif self.gnn == "SAGE":
            self.graph_convolution_1 = SAGEConv(self.number_of_features, self.gcn_first)
            self.graph_convolution_2 = SAGEConv(self.gcn_first, self.gcn_second)

        self.fully_connected_1 = nn.Linear(self.gcn_second, self.fc_1)
        self.fully_connected_2 = nn.Linear(self.fc_1, self.fc_2)

    def forward(self, data):
        edge_idx = data.edge_index
        features = data.x
        batch = data.batch
        epsilon = 1e-6

        node_features_1 = F.relu(self.graph_convolution_1(features, edge_idx))
        node_features_2 = self.graph_convolution_2(node_features_1, edge_idx)
        graph_feature = global_add_pool(node_features_2, batch)
        num_nodes = node_features_2.size(0)

        static_node_feature = node_features_2.clone().detach()
        node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)

        abstract_features_1 = F.tanh(self.fully_connected_1(node_features_2))
        assignment = F.softmax(self.fully_connected_2(abstract_features_1), dim=1)
        
        gumbel_assignment = self.gumbel_softmax(assignment)

        node_feature_mean = node_feature_mean.repeat(num_nodes, 1)
        lambda_pos = gumbel_assignment[:, 0].unsqueeze(dim=1)
        lambda_neg = gumbel_assignment[:, 1].unsqueeze(dim=1)

        subgraph_representation = scatter_add(lambda_pos * node_features_2, batch, dim=0)

        noisy_node_feature_mean = lambda_pos * node_features_2 + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std
        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
        noisy_graph_feature = scatter_add(noisy_node_feature, batch, dim=0)

        KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2) + torch.sum(
            ((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2,
            dim=0,
        )
        KL_Loss = torch.mean(KL_tensor)

        values = torch.ones(edge_idx.size(1), device=edge_idx.device)
        num_nodes = assignment.size(0)
        Adj_sparse = torch.sparse_coo_tensor(edge_idx, values, (num_nodes, num_nodes))

        temp = torch.sparse.mm(Adj_sparse, assignment)
        new_adj = torch.mm(assignment.t(), temp)

        row_sum = new_adj.sum(dim=1, keepdim=True)
        normalize_new_adj = new_adj / row_sum

        norm_diag = torch.diag(normalize_new_adj)
        eye = torch.ones_like(norm_diag)
        pos_penalty = self.mseloss(norm_diag, eye)

        preserve_rate = (assignment[:, 0] > 0.5).sum().item() / assignment.size(0)
        return graph_feature, noisy_graph_feature, subgraph_representation, pos_penalty, KL_Loss, preserve_rate

    
    def subgraph_embedding(self, data):
        edge_idx = data.edge_index
        features = data.x
        batch = data.batch

        node_features_1 = F.relu(self.graph_convolution_1(features, edge_idx))
        node_features_2 = self.graph_convolution_2(node_features_1, edge_idx)

        abstract_features_1 = F.tanh(self.fully_connected_1(node_features_2))
        assignment = F.softmax(self.fully_connected_2(abstract_features_1), dim=1)
        preserve_rate = (assignment[:, 0] > 0.5).sum().item() / assignment.size(0)
        assignment = torch.argmax(assignment, dim=1).unsqueeze(dim=1)

        subgraph_representation = scatter_add(assignment * node_features_2, batch, dim=0)


        return subgraph_representation, assignment, preserve_rate

    def gumbel_softmax(self, prob):
        return F.gumbel_softmax(prob, tau=1, dim=-1)


class Subgraph(nn.Module):
    def __init__(self, gcn_first, gcn_second, fc_1, fc_2, cls_hidden, number_of_features, con_weight=5):
        super(Subgraph, self).__init__()
        self.gcn_first = gcn_first
        self.gcn_second = gcn_second
        self.fc_1 = fc_1
        self.fc_2 = fc_2
        self.cls_hidden = cls_hidden
        self.con_weight = con_weight
        self.number_of_features = number_of_features
        self.mse_criterion = nn.MSELoss(reduction="mean")
        self.bce_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.graph_level_model = SAGE(self.gcn_first, self.gcn_second, self.fc_1, self.fc_2, self.number_of_features, 'GAT')
        self.classify = nn.Sequential(
            nn.Linear(self.gcn_second, self.cls_hidden),
            nn.ReLU(),
            nn.Linear(self.cls_hidden, 2),
        )

    def forward(self, data):
        graph_feature, noisy_graph_feature, subgraph_repr, pos_penalty, kl_loss, preserve_rate = self.graph_level_model(data)

        concat_feature = torch.cat((graph_feature, noisy_graph_feature), dim=0)
        pred = self.classify(concat_feature)
        label = torch.cat((data.y, data.y), dim=0)
        cls_loss = F.cross_entropy(pred, label)
        pred = torch.argmax(pred, dim=1)
        correct_num = torch.sum(pred == label).item()
        return graph_feature, noisy_graph_feature, subgraph_repr, kl_loss, cls_loss, pos_penalty, preserve_rate, correct_num

    def test(self, data):
        self.eval()
        with torch.no_grad():
            subgraph_repr, assignment, preserve_rate = self.graph_level_model.subgraph_embedding(data)
            
            pred = self.classify(subgraph_repr)
            cls_loss = F.cross_entropy(pred, data.y)
            pred = torch.argmax(pred, dim=1)
            correct_num = torch.sum(pred == data.y).item()
        return correct_num, cls_loss, preserve_rate

import torch_geometric.nn as gnn
class BaselineGNN(nn.Module):
    def __init__(self, gcn_first, gcn_second, fc_1, fc_2, cls_hidden, number_of_features):
        super(BaselineGNN, self).__init__()

        self.gcn_first = gcn_first
        self.gcn_second = gcn_second
        self.fc_1 = fc_1
        self.fc_2 = fc_2
        self.cls_hidden = cls_hidden
        self.number_of_features = number_of_features
        self.gcn_layer_1 = GATConv(self.number_of_features, self.gcn_first, heads=2)
        self.gcn_layer_2 = GATConv(self.gcn_first * 2, self.gcn_second)
        self.classifier = nn.Sequential(
            nn.Linear(self.gcn_second, self.cls_hidden),
            # nn.BatchNorm1d(self.cls_hidden),
            nn.ReLU(),
            nn.Linear(self.cls_hidden, 2),
        )
        
    def forward(self, data):
        edge_idx = data.edge_index
        features = data.x
        node_features_1 = F.relu(self.gcn_layer_1(features, edge_idx))
        node_features_2 = self.gcn_layer_2(node_features_1, edge_idx)
        graph_embedding = global_add_pool(node_features_2, data.batch)
        logits = self.classifier(graph_embedding)
        cls_loss = F.cross_entropy(logits, data.y)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=1)
        correct_num = torch.sum(pred == data.y)
        return graph_embedding, cls_loss, correct_num, probs