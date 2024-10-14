from typing import Iterable
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from distribution import NegativeBinomial
from utils import one_hot, broadcast_labels

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
                            nn.BatchNorm1d(n_out)
                            if use_batch_norm
                            else None,
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
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1)
        )

        # dispersion for Negative Binomial
        self.dispersion_decoder = nn.Linear(n_hidden, n_output)
        
    def forward(self, z: torch.Tensor, library_size: torch.Tensor, *cat_list: torch.Tensor):
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
            nn.Linear(n_hidden, n_labels)
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
    Bulk RNA seq data VAE model, consider batch id and library size for bulk RNA seq data, also consider label for semi-supervised learning
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
        self.dispersion = nn.Parameter(torch.randn(n_input))

        self.z_encoder = Encoder(n_input, n_hidden, n_latent, n_layers, None, dropout_rate)
        self.decoder = Decoder(n_latent, n_hidden, n_input, n_layers, [n_batch], dropout_rate)
        self.classifier = Classifier(n_latent, n_hidden, n_labels, n_layers, dropout_rate)
        self.l_encoder = Encoder(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=1,
            n_layers=n_layers,
            dropout_rate=dropout_rate
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
            l=l
        )

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        '''
        Args:
            x: torch.Tensor, shape (batch_size, n_input)
            local_l_mean: torch.Tensor, shape (batch_size, 1), this is the local mean of library size for each sample
            local_l_var: torch.Tensor, shape (batch_size, 1), this is the local variance of library size for each sample
            batch_index: torch.Tensor, shape (batch_size, n_batch), this is the batch index for each sample
            y: torch.Tensor, shape (batch_size, n_labels), this is the label for each sample
        '''
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
        kl_l = kl_divergence(Normal(ql_m, ql_v.sqrt()), Normal(local_l_mean, local_l_var)).sum(dim=1)

        logits = self.classifier(z)
        probs = F.softmax(logits, dim=-1)
        if not is_labelled:
            return recon_loss, kl_z, kl_l, probs

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
        attention = F.softmax(torch.bmm(q, k.transpose(1, 2))/torch.sqrt(torch.tensor(k.size(-1)).float()), dim=-1)
        out = torch.bmm(attention, v)
        out = self.fc_o(out)
        return out