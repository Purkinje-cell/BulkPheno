import torch
import torch_geometric
import logging
import __init__
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add, scatter_mean
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from gcn_lib.sparse.torch_vertex import GENConv
from gcn_lib.sparse.torch_nn import norm_layer, MLP
from tqdm import tqdm
from utils import forward_sub
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


# +
class Motif(nn.Module):
    def __init__(self, args):
        super(Motif, self).__init__()
        self.K = args.num_motifs
        self.motifs = nn.Parameter(F.normalize(torch.randn(self.K, args.proj_dim), dim=1), requires_grad=True)
    
    def _get_X(self, args, loader, encoder, sampler):
        with torch.no_grad():
            all_sub_proj = []
            for i, data in tqdm(enumerate(loader)):
                data_sub_ids, sub_whole_batch = sampler(data)
                
                data = data.to(args.device)
                node_emb, _, _, _ = encoder(data)
                sub_proj, _ = forward_sub(encoder, node_emb, data_sub_ids)
                sub_proj = F.normalize(sub_proj, dim=1)
                all_sub_proj += [sub_proj]

            X = torch.cat(all_sub_proj).detach().cpu().numpy()

        return X
    
    def initialize(self, args, loader, encoder, sampler):
        print("Initializing motifs ... ")
        X = self._get_X(args, loader, encoder, sampler)

#         gmm = GaussianMixture(n_components=self.K, covariance_type='spherical')
#         y = gmm.fit_predict(X)
#         self.centers.data = torch.from_numpy(gmm.means_).to(args.device).float()
        
        kmeans = KMeans(n_clusters=self.K)
        y = kmeans.fit_predict(X)
        self.motifs.data = torch.from_numpy(kmeans.cluster_centers_).to(args.device).float()

        return X, y


# -

'''
DeeperGCN implementation adapted from Guohao Li
https://github.com/lightaime/deep_gcns_torch/blob/master/examples/ogb/ogbg_mol/model.py
'''
class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block
        self.conv_encode_edge = args.conv_encode_edge
        self.add_virtual_node = args.add_virtual_node

        hidden_channels = args.emb_dim
        conv = args.conv
        aggr = args.gcn_aggr
        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        norm = args.norm
        mlp_layers = args.mlp_layers

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels]*3,
                                                     norm=norm))

        for layer in range(self.num_layers):
            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=self.conv_encode_edge, bond_encoder=True,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if not self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)
            

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

            
        h = self.atom_encoder(x)

        if self.add_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
            h = h + virtualnode_embedding[batch]

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = F.relu(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.add_virtual_node:
                    virtualnode_embedding_temp = global_add_pool(h2, batch) + virtualnode_embedding
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer-1](virtualnode_embedding_temp),
                        self.dropout, training=self.training)

                    h2 = h2 + virtualnode_embedding[batch]

                h = self.gcns[layer](h2, edge_index, edge_emb) + h

            h = self.norms[self.num_layers - 1](h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = F.relu(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')
        
        return h    

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))

# +
'''
GIN and GCN implementation adapted from Weihua Hu
https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
'''

class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            Note: additional BatchNorm in the message passing compared with regular GINConv!
        '''
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), nn.BatchNorm1d(2*emb_dim), nn.ReLU(), nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)    
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out


# -

class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

# +
# class GCNConv(MessagePassing):
#     def __init__(self, emb_dim):
#         super(GCNConv, self).__init__(aggr = "add")

#         self.emb_dim = emb_dim
#         self.linear = torch.nn.Linear(emb_dim, emb_dim)
#         self.bond_encoder = BondEncoder(emb_dim = emb_dim)
#         self.aggr = aggr

#     def norm(self, edge_index, num_nodes, dtype):
#         ### assuming that self-loops have been already added in edge_index
#         edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
#                                      device=edge_index.device)
#         row, col = edge_index
#         deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

#         return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


#     def forward(self, x, edge_index, edge_attr):
#         #add self loops in the edge space
#         edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

#         #add features corresponding to self-loop edges.
#         self_loop_attr = torch.zeros(x.size(0), 2)
#         self_loop_attr[:,0] = 4 #bond type for self-loop edge
#         self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
#         edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

#         edge_embedding = self.bond_encoder(edge_attr)    
        
# #         edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

#         norm = self.norm(edge_index, x.size(0), x.dtype)

#         x = self.linear(x)

#         return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm = norm)

#     def message(self, x_j, edge_attr, norm):
#         return norm.view(-1, 1) * (x_j + edge_attr)
# -

class GIN(torch.nn.Module):
    """
    GIN model with edge attributes
    """   
    def __init__(self, args):
        super(GIN, self).__init__()
        self.num_layers = args.num_layers
        self.drop_ratio = args.dropout
        self.JK = args.JK
        self.normalization = args.norm
        emb_dim = args.emb_dim
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.x_embed = AtomEncoder(emb_dim)
        
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            # With edge_attr
            self.gnns.append(GINConv(emb_dim))

            # Without edge_attr
            # nn_gin = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
            # conv_gin = GINConv(nn_gin)
            # self.gnns.append(conv_gin)
        
        self.norms = torch.nn.ModuleList()
        
        if self.normalization == 'batch':
            for layer in range(self.num_layers):
                self.norms.append(torch.nn.BatchNorm1d(emb_dim))
        elif self.normalization == 'layer':
            for layer in range(self.num_layers):
                self.norms.append(torch.nn.LayerNorm(emb_dim))

                        
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embed(x)

        h_list = [x]
        for layer in range(self.num_layers):
            
            # With edge_attr
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)

            # Without edge_attr
            # h = self.gnns[layer](h_list[layer], edge_index)
            
            if self.normalization != 'none':
                h = self.norms[layer](h)
    
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training) # remove relu for the last layer
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)
        
        
        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
        
        return node_representation


class GCN(torch.nn.Module):
    """
    GCN model with edge attributes
    """   
    def __init__(self, args):
        super(GCN, self).__init__()
        self.num_layers = args.num_layers
        self.drop_ratio = args.dropout
        self.JK = args.JK
        self.normalization = args.norm
        emb_dim = args.emb_dim
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.x_embed = AtomEncoder(emb_dim)
        
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            # With edge_attr
            self.gnns.append(GCNConv(emb_dim))

            # Without edge_attr
            # nn_gin = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
            # conv_gin = GINConv(nn_gin)
            # self.gnns.append(conv_gin)
        
        self.norms = torch.nn.ModuleList()
        
        if self.normalization == 'batch':
            for layer in range(self.num_layers):
                self.norms.append(torch.nn.BatchNorm1d(emb_dim))
        elif self.normalization == 'layer':
            for layer in range(self.num_layers):
                self.norms.append(torch.nn.LayerNorm(emb_dim))

                        
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embed(x)

        h_list = [x]
        for layer in range(self.num_layers):
            
            # With edge_attr
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)

            # Without edge_attr
            # h = self.gnns[layer](h_list[layer], edge_index)
            
            if self.normalization != 'none':
                h = self.norms[layer](h)
    
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training) # remove relu for the last layer
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)
        
        
        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
        
        return node_representation



class GNN(torch.nn.Module):
    """
    Wrapper of different GNN models
     
    """   
    def __init__(self, args):
        super(GNN, self).__init__()
        graph_pooling = args.graph_pooling
        emb_dim = args.emb_dim
        proj_dim = args.proj_dim
        proj_mode = args.proj_mode
        num_tasks = args.num_tasks if hasattr(args, 'num_tasks') else 1

        if args.model_type == "gin":
            self.gnn = GIN(args)
        elif args.model_type == "gcn":
            self.gnn = GCN(args)
        elif args.model_type == "dgcn":
            # For now use the same hidden_channels as the embedding dimension
            self.gnn = DeeperGCN(args)
        
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.graph_pool = global_add_pool
        elif graph_pooling == "mean":
            self.graph_pool = global_mean_pool
        elif graph_pooling == "max":
            self.graph_pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
              
        #One more MLP to project the final embedding
        if proj_mode == 'linear':
            self.projector = nn.Linear(emb_dim, proj_dim, bias=False)
        elif proj_mode == 'nonlinear':
            self.projector = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, proj_dim, bias=False)
            )

        #One more MLP for the downstream task
        self.task_pred = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, num_tasks)
        )
        
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_emb = self.gnn(x, edge_index, edge_attr)
        graph_emb = self.graph_pool(node_emb, batch)   
        graph_proj = self.projector(graph_emb)
        graph_pred = self.task_pred(graph_emb)
        return node_emb, graph_emb, graph_proj, graph_pred


