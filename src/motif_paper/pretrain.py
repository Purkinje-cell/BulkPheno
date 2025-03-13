import os
import argparse
import time
import sklearn
import torch
import ogb
import logging
import pytz
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F 
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean, scatter_sum
from datetime import datetime
from cuml import TSNE
from tqdm import tqdm
from models import GNN, Motif
from utils import *
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from loader import MoleculeDataset


tz = pytz.timezone('US/Pacific')
log_time = datetime.now(tz).strftime('%b%d_%H_%M_%S')
logging.Formatter.converter = timetz
logging.basicConfig(filename=f'./log/pretrain_{log_time}.log', level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%b%d %H-%M-%S')

parser = argparse.ArgumentParser(description='PyTorch implementation of MICRO-Graph')
parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='which dataset should be used for pre-training.')
parser.add_argument('--eval_metric', type=str, default='rocauc', help='graph task evaluate metric')   
parser.add_argument('--load_model', default=False, action="store_true", help='whether to load the trained model or not.')
parser.add_argument('--input_model_path', type=str, default = './saved_model/', help='filename to read the model (if there is any)')
parser.add_argument('--output_model_path', type=str, default = './saved_model/', help='filename to output the pre-trained model')
parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the pre-trained model')
parser.add_argument('--log_path', type=str, default = './runs/', help='log path')
parser.add_argument('--start_epoch', type=int, default=1, help='epoch to start training')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--seed', type=int, default = 0, help='random seed')

# +
'''
MICRO-Graph args
'''
# Sampling
parser.add_argument('--sample_method', type=str, default="sc",
                    help='which sampling strategy to use, one of [rw, khop, sc] (default: sc)')
parser.add_argument('--min_nodes', type=int, default=4, 
                    help='number of minimum nodes for filtering subgraph samples')
parser.add_argument('--sc_temp', type=int, default=0.2, help='temperature for node similarity computation')
parser.add_argument('--num_samples', type=int, default=None, 
                    help='number of samples. If None, the value will be decided according to graph size')
parser.add_argument('--rw_walk_len', type=tuple, default=(5, 40), help='range of walk length of random walk sampling')
parser.add_argument('--khop_hops', type=tuple, default=(1, 3), help='range of number of hops of khop sampling')

parser.add_argument('--num_motifs', type=int, default=20, 
                    help='number of motifs')
parser.add_argument('--sim_temp', type=float, default=0.05, 
                    help='temperature for calculating similarity between subgraphs, and subgraphs and motifs')

# Clustering
parser.add_argument('--assign_temp', type=float, default = 0.1, help='temperature calculating subgraph assignment')
parser.add_argument('--sinkhorn_lamb', type=float, default = 20, help='sinkhorn_lamb')
parser.add_argument('--sinkhorn_iterations', type=int, default = 5, help='number of iterations to run the sinkhorn algorithm')

# Contrastive
parser.add_argument('--filter_neg', default=False, action="store_true", help='whether to filter negative samples.')
parser.add_argument('--filter_thresh', type=float, default = 0.9, help='the threshold for filtering negative samples.')

# Loss
parser.add_argument('--lamb_cluster', type=float, default = 1, help='weight of clustering loss')
parser.add_argument('--lamb_contra', type=float, default = 0.05, help='weight of contrastive loss')
parser.add_argument('--lamb_sampler', type=float, default = 0.05, help='weight of sampler loss')


# -

'''
Optimization args
'''
parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0,
                    help='weight decay (default: 0)')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clip (default: 0.5)')   

'''
GNN architecture args
'''
parser.add_argument('--model_type', type=str, default='dgcn',
                    help='GNN model architecture, one of [gin, gcn, dgcn] (default: dgcn)')   
parser.add_argument('--num_layers', type=int, default=5,
                    help='number of GNN message passing layers (default: 5).')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='embedding dimensions (default: 300)')
parser.add_argument('--proj_dim', type=int, default=300,
                    help='projection dimensions (default: 300)')
parser.add_argument('--norm', type=str, default='batch',
                    help='normalization, one of [batch, layer, none] (defaul: batch)')   
parser.add_argument('--proj_mode', type=str, default='nonlinear',
                    help='projector mode, one of [linear, nonlinear] (default: nonlinear)')   
parser.add_argument('--graph_pooling', type=str, default="mean",
                    help='graph level pooling (sum, mean, max, set2set, attention)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout ratio (default: 0.2)')   


'''
GNN architecture args continued (DeeperGCN)
'''
parser.add_argument('--add_virtual_node', action='store_true')
# training & eval settings
parser.add_argument('--mlp_layers', type=int, default=1,
                    help='the number of layers of mlp in conv')
parser.add_argument('--block', default='res+', type=str,
                    help='graph backbone block type {res+, res, dense, plain}')
parser.add_argument('--conv', type=str, default='gen',
                    help='the type of GCNs')
parser.add_argument('--gcn_aggr', type=str, default='softmax',
                    help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
# learnable parameters
parser.add_argument('--t', type=float, default=1.0,
                    help='the temperature of SoftMax')
parser.add_argument('--p', type=float, default=1.0,
                    help='the power of PowerMean')
parser.add_argument('--learn_t', action='store_true')
parser.add_argument('--learn_p', action='store_true')
# message norm
parser.add_argument('--msg_norm', action='store_true')
parser.add_argument('--learn_msg_scale', action='store_true')
# encode edge in conv
parser.add_argument('--conv_encode_edge', action='store_true')


'''
GNN architecture args continued (GIN)
'''
parser.add_argument('--JK', type=str, default="last",
                    help='how the node features across layers are combined. last, sum, max or concat')



args = parser.parse_args()
args.device = torch.device('cuda:'+ str(args.device) if torch.cuda.is_available() else 'cpu')
logging.info(f"device: {args.device}")

set_seed(args.seed)

'''
Choose Dataset
'''
if args.dataset == 'Synthetic':
    args.num_tasks = 20
elif args.dataset == 'ogbg-molhiv':
    args.num_tasks = 1
elif args.dataset == 'ogbg-molbbbp':
    args.num_tasks = 1
elif args.dataset == 'ogbg-molbace':
    args.num_tasks = 1
elif args.dataset == 'ogbg-moltox21':
    args.num_tasks = 12
elif args.dataset == 'ogbg-moltoxcast':
    args.num_tasks = 617
elif args.dataset == 'ogbg-molsider':
    args.num_tasks = 27
elif args.dataset == 'ogbg-molclintox':
    args.num_tasks = 2
elif args.dataset == 'ogbg-molmuv':
    args.num_tasks = 17
    args.eval_metric = 'prcauc'
elif args.dataset == 'zinc_standard_agent':
    args.num_tasks = 1
else:
    raise ValueError("Dataset not recognized")


logging.info(f"Loading dataset: {args.dataset}")

set_seed(args.seed)
if 'Synthetic' in args.dataset:
    dataset = SyntheticDataset('data/'+ args.dataset).shuffle()
elif 'ogb' in args.dataset:
    dataset = PygGraphPropPredDataset(name=args.dataset).shuffle()
    logging.info(f"Dataset size before filtering {len(dataset)}")
    dataset = dataset_filter(dataset)
    logging.info(f"Dataset size after filtering {len(dataset)}")
elif args.dataset == 'zinc_standard_agent':
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset).shuffle()

'''
Create dataloaders
''' 
B = args.batch_size
loader = DataLoader(dataset, batch_size=B, shuffle=False)
initial_loader = DataLoader(dataset[:B*2], batch_size=B, shuffle=False)

logging.info(f'Number of batches: {len(loader)}')

'''
Create model, encoder, and sampler
''' 
model = Motif(args).to(args.device)
encoder = GNN(args)
if args.input_model_file == '':
    input_model = args.dataset + f'_{args.model_type}_{args.num_layers}_layers'
else:
    input_model = args.input_model_file
    logging.info(f"Loading model:\n{args.input_model_file}")
    encoder.load_state_dict(torch.load(args.input_model_path + args.input_model_file +".pth", map_location='cpu'))

encoder = encoder.to(args.device)

if args.sample_method == 'rw' or args.sample_method == 'khop':
    sampler = lambda data_batch: heuristic_sample_batch(args, data_batch)
elif args.sample_method == 'sc':
    sampler = lambda data_batch: sc_sample_batch(args, data_batch, encoder)


optim = torch.optim.AdamW([{'params': model.parameters(), 'lr': args.lr},
                         {'params': encoder.parameters(), 'lr': args.lr}])

pretrain_log = []
eval_log = []
log_titles = ["loss", "cluster_loss", "contra_loss",  "sampler_loss"]
output_model = input_model + f'_{args.num_motifs}_motifs_{args.sample_method}'
logging.info(f"output:  {args.output_model_path + output_model}")

criterion = torch.nn.BCEWithLogitsLoss(reduction = "mean")
X, y = model.initialize(args, initial_loader, encoder, sampler)

encoder.train()
num_batchs = len(loader)
sc_ids = [0] * num_batchs
for e in tqdm(range(args.start_epoch, args.start_epoch+args.num_epochs)):
    epoch_loss = 0
    for i, data in enumerate(loader):
        sc_ids[i] = sampler(data)

        data_sub_ids, sub_whole_batch = sc_ids[i]
        
        optim.zero_grad()
        data = data.to(args.device)

        node_emb, _, _, _ = encoder(data)
        whole_proj = encoder.projector(encoder.graph_pool(node_emb, data.batch))
        sub_proj, in_sub_node_sim = forward_sub(encoder, node_emb, data_sub_ids)


        S = pairwise_cosine_sim(model.motifs, sub_proj)

        with torch.no_grad():
            S_top10, _ = S.topk(k=int(0.1 * S.shape[1]), dim=1) # K x 0.1*N
            eta = S_top10[:, -1] # K x 1
            S_mask = (S > eta.view(-1,1)).any(dim=0) # 1 x N

            Q = sinkhorn(S, args.sinkhorn_iterations)
            Q_hard = F.one_hot(Q.argmax(dim=0), num_classes=model.K).t()


        sampler_loss = - in_sub_node_sim[S_mask].mean()

        cluster_loss = - (Q_hard * torch.log_softmax(S / args.sim_temp, dim=0)).sum(dim=0).mean()    


        num_subs = degree(sub_whole_batch)        
        blocks = [torch.ones(1, int(n)) for n in num_subs]
        W_mask = torch.block_diag(*blocks).to(args.device)


        W = pairwise_cosine_sim(whole_proj, sub_proj)

        if args.filter_neg:
            with torch.no_grad():
                sub_sub_blocks = [torch.ones(int(n), int(n)) for n in num_subs]
                sub_same_whole_mask = torch.block_diag(*sub_sub_blocks).bool().to(args.device)

                sub_sub_sim = pairwise_cosine_sim(sub_proj, sub_proj)

                # not from the same whole graph, and similarity is greater than the filter_threshold
                false_neg_mask = (~sub_same_whole_mask) & (sub_sub_sim > args.filter_thresh)

                # aggregate false_neg_mask, from N x N to M x N
                false_neg_mask = scatter_mean(false_neg_mask.float().t(), sub_whole_batch.to(args.device)).t()

                false_neg_mask = false_neg_mask.bool()
                W[false_neg_mask] = -10

        contra_loss = - (W_mask * torch.log_softmax(W / args.sim_temp, dim=1)).sum(dim=1).mean()


        loss = args.lamb_cluster*cluster_loss + args.lamb_contra*contra_loss + args.lamb_sampler*sampler_loss
        epoch_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optim.step()  
        pretrain_log += [[l.item() for l in [loss, cluster_loss, contra_loss, sampler_loss]]]
        
        if (i+1) % 50 == 0:
            logging.info(f"Batch {(e-1)*num_batchs+i} loss  : {loss.item() :.4f}")            
            save_model(output_model, (e-1)*num_batchs+i+1, model, encoder, output_path=args.output_model_path)    


