import torch
import torch.nn as nn
import random
import pytz
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import networkx as nx
import torch.nn.functional as F 
from torch.distributions.multinomial import Multinomial
from torch_cluster import random_walk
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_geometric.utils import softmax, degree, to_dense_adj, k_hop_subgraph, subgraph, to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import DataLoader, Data, Batch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from itertools import combinations, chain
from tqdm import tqdm
from datetime import datetime


# +
def timetz(*args):
    tz = pytz.timezone('US/Pacific')
    return datetime.now(tz).timetuple()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def save_model(output_model, epoch, model, encoder, matcher=None, output_path='saved_model_train/'):
    '''
    Save model while training
    '''
    train_model = output_path + output_model + f'_epoch_{epoch}'
    torch.save(model.state_dict(), train_model + ".pth")
    train_encoder = train_model + '_encoder'
    torch.save(encoder.state_dict(), train_encoder + ".pth")
    if matcher is not None:
        train_matcher = train_model + '_matcher'
        torch.save(matcher.state_dict(), train_matcher + ".pth")


# -

def pairwise_cosine_sim(embed1, embed2):
    return torch.mm(F.normalize(embed1, dim=1), F.normalize(embed2, dim=1).t())



def powerset(iterable):
    "Note: empty set and full set are excluded, unless iterable only contain one element"
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    "powerset([1]) --> (1,)"
    
    s = list(iterable)
    if len(s) == 0:
        raise ValueError("Empty input")
    elif len(s) == 1:
        return [(0,)]
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


# +
def forward_sub(encoder, node_emb, data_sub_ids):
    sub_emb = []
    node_sub_batch = []
    in_sub_sim = []
    N = 0
    for i, sub_ids in enumerate(data_sub_ids):
        for j, sub_id in enumerate(sub_ids):
            in_sub_node_emb = node_emb[sub_id]
            sub_emb += [in_sub_node_emb]
            node_sub_batch += [N + j] * sub_id.shape[0]
            in_sub_sim += [pairwise_cosine_sim(in_sub_node_emb, in_sub_node_emb).mean().view(1)]
            
        N += len(sub_ids)
    sub_emb = torch.cat(sub_emb)
    node_sub_batch = torch.LongTensor(node_sub_batch).to(node_emb.device)
    in_sub_node_sim = torch.cat(in_sub_sim)
    sub_proj = encoder.projector(encoder.graph_pool(sub_emb, node_sub_batch))
    return sub_proj, in_sub_node_sim


'''
Implementation of the sinkhorn function adopted from 
https://github.com/facebookresearch/swav/blob/master/main_swav.py
'''
def sinkhorn(S, num_iters, lamb=20):
    with torch.no_grad():
        Q = torch.exp(S*lamb)
        Q /= torch.sum(Q)
        u = torch.zeros(Q.shape[0]).to(Q.device)
        r = torch.ones(Q.shape[0]).to(Q.device) / Q.shape[0]
        c = torch.ones(Q.shape[1]).to(Q.device) / Q.shape[1]

        curr_sum = torch.sum(Q, dim=1)
        for it in range(num_iters):
            u = curr_sum
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim=1)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).float()



# -

def dataset_filter(dataset, min_nodes=4):
    print("Filtering dataset ...")
    filtered_dataset = []
    for d in tqdm(dataset):
        if d.edge_index.shape[1] == 0: # no edges
            continue
        if not np.isin(np.arange(d.x.shape[0]), d.edge_index).all(): # has isolated nodes
            continue
        if d.num_nodes < min_nodes:
            continue
        filtered_dataset.append(d)
    return filtered_dataset


# +
'''
Spectral clustering sampling utils
'''

def sc_sample_batch(args, data_batch, encoder):
    with torch.no_grad():
        data_batch = data_batch.to(args.device)
        node_emb, _, _, _ = encoder(data_batch)
        batch = data_batch.batch

        node_sim = pairwise_cosine_sim(node_emb, node_emb)
        node_sim[batch.view(1, -1) != batch.view(-1, 1)] = float('-inf')
        node_sim.fill_diagonal_(float('-inf'))
        aff_matrix = torch.softmax(node_sim / args.sc_temp, dim=1).cpu()

        edge_index = data_batch.edge_index.cpu()
        batch = batch.cpu()
        num_nodes = degree(batch).long()
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]]).cpu()

        all_subgraphs = []
        sub_whole_batch = []
        for i, single_num_nodes in enumerate(num_nodes):
            single_mask = batch == i
            single_graph_nodes = single_mask.nonzero(as_tuple=True)[0]
            single_edge_index = edge_index[:, np.isin(edge_index[0], single_graph_nodes)]
            single_aff = aff_matrix[single_mask][:, single_mask]
            cum_node = cum_nodes[i] 

            subgraphs = sc_single_graph(single_num_nodes, single_edge_index, single_aff, cum_node, args.min_nodes)

            all_subgraphs += [subgraphs]
            sub_whole_batch += [i] * len(subgraphs)
        sub_whole_batch = torch.LongTensor(sub_whole_batch)
        return all_subgraphs, sub_whole_batch

def sc_single_graph(single_num_nodes, single_edge_index, single_aff, cum_node, min_nodes):
    idxs = torch.arange(single_num_nodes)
    if single_num_nodes < min_nodes:
        return [idxs + cum_node]

    single_aff = (single_aff + single_aff.t()) / 2            
    num_clusters = int((single_num_nodes//min_nodes) ** 0.5) + 1
    sc = SpectralClustering(affinity='precomputed', n_clusters=num_clusters, n_jobs=8)
    pred = sc.fit_predict(single_aff.numpy())

    subgraphs = []
    for j in range(num_clusters):
        sampled_nodes = idxs[pred == j]
        if sampled_nodes.shape[0] >= min_nodes:
            sampled_edge_index, _ = subgraph(sampled_nodes + cum_node, single_edge_index)
            # only select the connected components
            if sampled_edge_index.shape[1] > 0:
                subgraphs += get_cc(sampled_edge_index, cum_node, min_nodes)

    if len(subgraphs) == 0:
        subgraphs = [sampled_nodes + cum_node] 
    
    return subgraphs

def get_cc(sampled_edge_index, cum_node, min_nodes):
    shifted_sampled_edge_index = sampled_edge_index - cum_node
    adj = to_dense_adj(shifted_sampled_edge_index)[0].numpy()
    cc_label = connected_components(adj)[1]

    unique_cc_label, unique_count = np.unique(cc_label, return_counts=True)
    select_cc_label = unique_cc_label[unique_count >= min_nodes]

    subgraphs = []
    for label in select_cc_label:
        select_nodes = (cc_label == label).nonzero()[0] 
        if select_nodes.shape[0] >= min_nodes:
            subgraphs += [torch.from_numpy(select_nodes) + cum_node]
    return subgraphs


# +
'''
Heuristic sampling utils
'''

def heuristic_sample_batch(args, data_batch):
    min_nodes = args.min_nodes
    data_list = data_batch.to_data_list()
    batch = data_batch.batch.cpu()
    data_num_nodes = degree(batch).long()
    data_cum_nodes = torch.cat([batch.new_zeros(1), data_num_nodes.cumsum(dim=0)[:-1]])

    all_subgraphs = []
    sub_whole_batch = []
    for idx, data in enumerate(data_list):
        if args.num_samples is not None:
            N = args.num_samples
        else:
            N = int((data.num_nodes / args.min_nodes) ** 0.5) + 1
        if args.sample_method == 'rw':
            subgraphs = rw_single_graph(data, N, args.rw_walk_len, args.min_nodes, data_cum_nodes[idx])
        elif args.sample_method == 'khop':
            subgraphs = khop_single_graph(data, N, args.khop_hops, args.min_nodes, data_cum_nodes[idx])
        else:
            raise ValueError("Unknown heuristic sample method")

        all_subgraphs += [subgraphs]
        sub_whole_batch += [idx] * len(subgraphs)

    sub_whole_batch = torch.LongTensor(sub_whole_batch)
    return all_subgraphs, sub_whole_batch

def rw_single_graph(data, num_samples=2, walk_len=None, min_nodes=4, cum_nodes=0):
    if walk_len is not None:
        min_walk, max_walk = walk_len
    else:
        min_walk, max_walk = min_nodes, data.num_nodes 
    
    start = torch.randperm(data.num_nodes)
    samples = []
    i = 0
    while len(samples) < num_samples and i < data.num_nodes:
        walk_length = np.random.randint(min_walk, max(min_walk + 1, max_walk))
        row, col = data.edge_index
        walk = random_walk(row, col, start[i:i+1], walk_length, coalesced=True)
        sampled_nodes = walk.unique()
        if sampled_nodes.shape[0] >= min_nodes:
            samples += [sampled_nodes + cum_nodes]
        i += 1
    
    if len(samples) == 0:
        samples += [sampled_nodes + cum_nodes]
        
    return samples


def khop_single_graph(data, num_samples=2, random_hops=None, min_nodes=4, cum_nodes=0):
    start = torch.randperm(data.num_nodes)
    samples = []
    i = 0
    while len(samples) < num_samples and i < data.num_nodes:
        if random_hops is not None:
            k = np.random.randint(random_hops[0], random_hops[1]+1)
        else:
            k = 2
        sampled_nodes, _, _, _ = k_hop_subgraph(node_idx=start[i].item(), num_hops=k, edge_index=data.edge_index)
        
        if sampled_nodes.shape[0] >= min_nodes:
            samples += [sampled_nodes + cum_nodes]  
        i += 1
        
    if len(samples) == 0:
        samples += [sampled_nodes + cum_nodes]       
    return samples
# -




