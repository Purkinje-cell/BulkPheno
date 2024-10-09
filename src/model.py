from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Normal, kl_divergence
from torch_geometric.nn import GCNConv

def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""NB parameterizations conversion

        :param mu: mean of the NB distribution.
        :param theta: inverse overdispersion.
        :param eps: constant used for numerical log stability.
        :return: the number of failures until the experiment is stopped
            and the success probability.
    """
    assert (mu is None) == (
        theta is None
    ), "If using the mu/theta NB parameterization, both parameters must be specified"
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits


def _convert_counts_logits_to_mean_disp(total_count, logits):
    """NB parameterizations conversion

        :param total_count: Number of failures until the experiment is stopped.
        :param logits: success logits.
        :return: the mean and inverse overdispersion of the NB distribution.
    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta

class FCLayer(nn.Module):
    """
    Full connected layer for VAE
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        activation: nn.Module = None,
        dropout_rate: float = 0.1,
    ):
        super(FCLayer, self).__init__()
        self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        self.fc = nn.Linear(n_input + sum(self.n_cat_list), n_output)
        self.batch_norm = nn.BatchNorm1d(n_output)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        one_hot = torch.cat([F.one_hot(cat, num_classes=n_cat) for cat, n_cat in zip(cat_list, self.n_cat_list)], dim=1)
        x = torch.cat([x, one_hot], dim=1)
        x = self.fc(x)
        x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        x = self.dropout(x)
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
        
        self.encoder = nn.Sequential(
            *[FCLayer(n_input, n_hidden, n_cat_list, nn.ReLU(), dropout_rate) for _ in range(n_layers)],
            nn.Linear(n_hidden, 2 * n_latent),
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, *cat_list):
        mu, log_var = self.encoder(x, *cat_list).chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    

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
        self.decoder = nn.Sequential(
            *[FCLayer(n_input, n_hidden, n_cat_list, nn.ReLU(), dropout_rate) for _ in range(n_layers)],
        )

        # mean gamma for Negative Binomial
        self.gamma_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1)
        )

        # dispersion for Negative Binomial
        self.dispersion_decoder = nn.Linear(n_hidden, n_output)
        
    def forward(self, z: torch.Tensor, library_size: torch.Tensor, *cat_list: torch.Tensor):
        hidden = self.decoder(z, *cat_list)
        gamma = self.gamma_decoder(hidden)
        rate = torch.exp(library_size) * gamma
        dispersion = self.dispersion_decoder(hidden)
        return gamma, dispersion, rate

class Classifier(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_layers: int = 1,
        n_labels: int = 5,
        dropout_rate: float = 0.1,
    ):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            *[FCLayer(n_input, n_hidden, n_labels, nn.ReLU(), dropout_rate) for _ in range(n_layers)],
            nn.Linear(n_hidden, n_output),
            nn.Softmax()
        )
    
    def forward(self, x):
        return self.classifier(x)

class SpatialEncoder(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass

class BulkVAE(nn.Module):
    '''
    Bulk RNA seq data VAE model, considers batch id and library size for bulk RNA seq data, also consider label for semi-supervised learning
    '''
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
        dispersion: str = 'gene',
    ):
        super(BulkVAE, self).__init__()
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate
        self.log_norm = log_norm
        self.rate = nn.Parameter(torch.randn(n_input))

        self.z_encoder = Encoder(n_input, n_hidden, n_latent, n_layers, [n_labels], dropout_rate)
        self.decoder = Decoder(n_latent, n_hidden, n_input, n_layers, [n_batch, n_labels], dropout_rate)
        self.classifier = Classifier(n_latent, n_hidden, n_layers, n_labels, dropout_rate)
        self.l_encoder = Encoder(
                n_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
            )
    
    def sample_z(self, x, y=None):
        if self.log_norm:
            x = torch.log1p(x)
        z, mu, log_var = self.z_encoder(x, y)
        return z
        
    def sample_l(self, x, y=None):
        if self.log_norm:
            x = torch.log1p(x)
        l, mu, log_var = self.l_encoder(x, y)
        return l
        
    def classify(self, x):
        x = torch.log1p(x)
        return self.classifier(x)
    
    def recon_loss(self, x, dispersion, rate):
        total_count, logits = _convert_mean_disp_to_counts_logits(rate, dispersion)
        recon_loss = (
            -NegativeBinomial(total_count, logits=logits).log_prob(x).sum(dim=-1)
        )
        return recon_loss

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x
        if self.log_norm:
            x_ = torch.log1p(x)
        
        z, qz_m, qz_v = self.z_encoder(x_, y)
        l, ql_m, ql_v = self.l_encoder(x_, y)
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.shape[0], qz_m.shape[1]))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.shape[0], qz_v.shape[1]))
            z = Normal(qz_m, qz_v.sqrt()).sample()
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.shape[0], ql_m.shape[1]))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.shape[0], ql_v.shape[1]))
            l = Normal(ql_m, ql_v.sqrt()).sample()
        
        gamma, dispersion, rate = self.decoder(z, l, batch_index, y)

        rate = self.rate
        rate = torch.exp(rate)
        
        return dict(
            gamma=gamma,
            dispersion=dispersion,
            rate=rate,
            qz_m=qz_m,
            qz_v=qz_v,
            z = z,
            ql_m=ql_m,
            ql_v=ql_v,
            l = l
        )

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        outputs = self.inference(x, batch_index, y)
        gamma = outputs["gamma"]
        dispersion = outputs["dispersion"]
        rate = outputs["rate"]
        recon_loss = self.recon_loss(x, dispersion, rate)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        mean = torch.zeros_like(qz_m)
        std = torch.ones_like(qz_v)
        kl_z = kl_divergence(Normal(qz_m, qz_v.sqrt()), Normal(mean, std)).sum(dim=1)
        
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        kl_l = kl_divergence(Normal(ql_m, ql_v.sqrt()), Normal(local_l_mean, local_l_var)).sum(dim=1)

        return recon_loss, kl_z, kl_l
        
        
    
    