import os
import sys
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scipy.sparse as sp
import torch
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from dataset import GraphContrastiveDataModule, SpatialGraphDataset, BulkDataModule, BulkDataset
from model import GraphContrastiveModel, BulkEncoderModel, EmbeddingStore
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch.utils.data import DataLoader as TorchDataLoader
from torch_scatter import scatter_add, scatter_mean

def parse_args():
    parser = argparse.ArgumentParser(description="Spatial Graph Contrastive Learning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--spatial_dataset", type=str, help="Name of spatial dataset to use")
    parser.add_argument("--bulk_dataset", type=str, help="Name of bulk dataset to use")
    parser.add_argument("--query_column", type=str, help="Column name for query in bulk data")
    parser.add_argument("--query_k", type=int, help="Number of neighbors to query")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
    
    return config

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn_new(batch):
    """Custom collate to handle mixed real/pseudo batches"""
    is_real = batch[0]['is_real']
    expressions = torch.stack([item['expression'] for item in batch])
    
    if is_real:
        return {'expression': expressions, 'is_real': True}
    else:
        graph_indices = [item['graph_idx'] for item in batch]
        return {
            'expression': expressions,
            'graph_indices': graph_indices,
            'is_real': False
        }

def load_data(config):
    # Load bulk data
    bulk_data_path = os.path.join(config['data_dir'], config['bulk_dataset_path'])
    bulk_adata = sc.read_h5ad(bulk_data_path)
    
    # Load spatial data
    spatial_data_path = os.path.join(config['data_dir'], config['spatial_dataset_path'])
    spatial_adata = sc.read_h5ad(spatial_data_path)
    
    # Find common genes
    common_genes = np.intersect1d(spatial_adata.var_names, bulk_adata.var_names)
    spatial_adata = spatial_adata[:, common_genes].copy()
    bulk_adata = bulk_adata[:, common_genes].copy()
    
    # Preprocess bulk data
    bulk_adata.layers['count'] = bulk_adata.X.copy()
    sc.pp.normalize_total(bulk_adata, target_sum=1e4)
    sc.pp.log1p(bulk_adata)
    sc.pp.scale(bulk_adata)
    sc.pp.pca(bulk_adata)
    
    # Compute neighbors for spatial data if not already done
    if 'neighbors' not in spatial_adata.uns:
        sc.pp.neighbors(spatial_adata, n_neighbors=config['n_neighbors'], use_rep='spatial')
    
    return bulk_adata, spatial_adata, common_genes

def main():
    args = parse_args()
    config = load_config(args)
    set_seed(config['seed'])
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['figure_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Save config
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H-%M")

    config_path = os.path.join(config['output_dir'], config['gcl_conv_layer'] + '_' + time_str + '_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load data
    bulk_adata, spatial_adata, common_genes = load_data(config)
    
    # Visualize spatial data
    sc.pl.embedding(spatial_adata, basis='spatial', color=config['spatial_color_by'])
    plt.savefig(os.path.join(config['figure_dir'], f"{config['spatial_dataset']}_spatial.png"))
    
    # Create graph dataset
    graph_dataset = SpatialGraphDataset(
        spatial_adata, 
        name=config['spatial_dataset'], 
        hops=config['hops']
    )
    print(f'Length of graph dataset is {len(graph_dataset)}')
    
    # Split dataset
    train_idx = np.random.choice(
        int(graph_dataset.len()), 
        int(graph_dataset.len() * config['train_ratio']), 
        replace=False
    )
    val_idx = np.setdiff1d(graph_dataset.indices(), train_idx)
    train_dataset = graph_dataset[train_idx]
    val_dataset = graph_dataset[val_idx]
    print(f'Length of train dataset is {len(train_dataset)}')
    print(f'Length of val dataset is {len(val_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        follow_batch=['x', 'mean_expression']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        follow_batch=['x', 'mean_expression']
    )
    
    # Initialize WandB logger
    wandb_logger_gcl = WandbLogger(project=config['wandb_project'], name=f"{config['spatial_dataset']}_gcl", version=time_str)
    
    # Save hyperparameters to WandB
    wandb_logger_gcl.log_hyperparams(config)
    
    # Train graph contrastive model
    graph_model = GraphContrastiveModel(
        input_dim=len(common_genes),
        latent_dim=config['gcl_latent_dim'],
        hidden_dim=config['gcl_hidden_dim'],
        recon_weight=config['gcl_recon_weight'],
        margin=config['gcl_margin'],
        conv_layer=config['gcl_conv_layer'],
    )

    wandb_logger_gcl.watch(graph_model, log='all', log_freq=500)
    
    graph_trainer = pl.Trainer(
        max_epochs=config['gcl_max_epochs'],
        logger=wandb_logger_gcl,
        log_every_n_steps=config['log_every_n_steps']
    )
    
    graph_trainer.fit(graph_model, train_loader, val_loader)
    
    # Save graph model
    graph_model_path = os.path.join(config['model_dir'], f"{config['spatial_dataset']}_gcl.pt")
    torch.save(graph_model.state_dict(), graph_model_path)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        follow_batch=['x', 'mean_expression']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        follow_batch=['x', 'mean_expression']
    )
    graph_train_embeddings = []
    graph_val_embeddings = []
    graph_model = graph_model.to(config['device'])
    graph_model.eval()
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(config['device'])
            x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
            z = graph_model.encode(x, edge_index, edge_attr, batch_idx)
            graph_train_embeddings.append(z.detach().cpu())
        for batch in val_loader:
            batch = batch.to(config['device'])
            x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
            z = graph_model.encode(x, edge_index, edge_attr, batch_idx)
            graph_val_embeddings.append(z.detach().cpu())
    
    # Concatenate results
    graph_train_embeddings = torch.cat(graph_train_embeddings, dim=0)
    graph_val_embeddings = torch.cat(graph_val_embeddings, dim=0)
    graph_embeddings = torch.cat([graph_train_embeddings, graph_val_embeddings], dim=0)
    print(f"Graph embeddings shape: {graph_embeddings.shape}")
    # Initialize WandB logger for bulk encoder
    wandb_logger_bulk = WandbLogger(project=config['wandb_project'], name=f"{config['spatial_dataset']}_bulk", version=time_str)
    
    # Train bulk encoder model
    bulk_encode_model = BulkEncoderModel(
        input_dim=len(common_genes),
        gcl_model=graph_model,
        latent_dim=config['bulk_latent_dim'],
        hidden_dims=config['bulk_hidden_dims']
    )

    wandb_logger_bulk.watch(bulk_encode_model, log='all', log_freq=500)

    bulk_dataset = BulkDataset(spatial_graph_dataset=graph_dataset)
    bulk_train_dataset = Subset(bulk_dataset, train_idx)
    bulk_val_dataset = Subset(bulk_dataset, val_idx)
        
    
    bulk_train_loader = TorchDataLoader(
        bulk_train_dataset,
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'], 
        collate_fn=collate_fn_new
    )
    
    bulk_val_loader = TorchDataLoader(
        bulk_val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'], 
        collate_fn=collate_fn_new
    )
    
    bulk_encode_model.set_graph_embedding(graph_embeddings)
    
    bulk_trainer = pl.Trainer(
        max_epochs=config['bulk_max_epochs'],
        log_every_n_steps=config['log_every_n_steps'],
        logger=wandb_logger_bulk
    )
    
    bulk_trainer.fit(bulk_encode_model, bulk_train_loader, bulk_val_loader)
    
    # Save bulk model
    bulk_model_path = os.path.join(config['model_dir'], f"{config['spatial_dataset']}_bulk.pt")
    torch.save(bulk_encode_model.state_dict(), bulk_model_path)
    
    # Filter bulk data based on query column if needed
    if config.get('query_filter'):
        query_filter = config['query_filter']
        bulk_query = bulk_adata[eval(query_filter)].copy()
    else:
        bulk_query = bulk_adata.copy()
    
    # Get bulk embeddings
    bulk_encode_model = bulk_encode_model.to(config['device'])
    bulk_embedding = bulk_encode_model.get_latent_embedding(bulk_query)
    bulk_labels = bulk_query.obs[config['query_column']].values
    bulk_ids = np.arange(len(bulk_labels))
    print(f"Bulk embeddings shape: {len(bulk_embedding)}, labels: {len(bulk_labels)}")
    
    # Create embedding store and query
    bulk_embedding_store = EmbeddingStore(metric=config['distance_metric'])
    bulk_embedding_store.build_index(bulk_embedding, bulk_labels, bulk_ids)
    
    query_k = config.get('query_k', 200)
    query_res = bulk_embedding_store.query(graph_embeddings, k=query_k)
    
    # Process query results
    distance_tensor = torch.Tensor(query_res['distances'])
    label_tensor = torch.Tensor(query_res['labels'].codes).to(torch.long)
    sim_tensor = scatter_mean(distance_tensor, label_tensor, dim=1)
    sim = sim_tensor.numpy()
    sim_df = pd.DataFrame(sim, columns=bulk_query.obs[config['query_column']].cat.categories)
    
    # Calculate rank-based similarity
    rank_tensor = torch.argsort(distance_tensor, dim=1, descending=False)
    rank_reciprocal_tensor = 1 / (rank_tensor + 1)
    rank_sum_tensor = scatter_mean(rank_reciprocal_tensor, label_tensor, dim=1)
    rank_sum = rank_sum_tensor.numpy()
    rank_sum_df = pd.DataFrame(rank_sum, columns=bulk_query.obs[config['query_column']].cat.categories)
    
    # Spatial smoothing
    spatial_adata.obsp['spatial_connectivities'] = sp.csr_matrix(spatial_adata.obsp['spatial_connectivities'])
    sim_new = spatial_adata.obsp['spatial_connectivities'] @ sim / config['n_neighbors']
    sim_smoothed_df = pd.DataFrame(sim_new, columns=bulk_query.obs[config['query_column']].cat.categories)
    
    # Add results to spatial adata
    sim_smoothed_df.index = spatial_adata[graph_dataset[sim_smoothed_df.index.values].center_node_idx.numpy()].obs_names
    
    # Drop columns if they already exist
    cols = []
    for col in sim_smoothed_df.columns:
        if col in spatial_adata.obs.columns:
            cols.append(col)
    
    if cols:
        spatial_adata.obs = spatial_adata.obs.drop(columns=cols)

    spatial_adata.obs = spatial_adata.obs.merge(sim_smoothed_df, left_index=True, right_index=True)
    
    # Add prediction
    spatial_adata.obs['Prediction'] = spatial_adata.obs[sim_smoothed_df.columns.tolist()].idxmax(axis=1)
    
    # Save results
    spatial_adata.obs.to_csv(os.path.join(config['output_dir'], f"{config['spatial_dataset']}_with_predictions.csv"))
    
    # Visualize results
    for x in [config['spatial_color_by']] + sim_smoothed_df.columns.tolist() + ['Prediction']:
        sc.pl.embedding(spatial_adata, basis='spatial', color=x)
        plt.savefig(os.path.join(config['figure_dir'], f"{config['spatial_dataset']}_{x}.png"))
    
    # Cell type proportions by prediction
    if 'cell_type' in spatial_adata.obs.columns:
        cell_prop = spatial_adata.obs[['Prediction', 'cell_type']].groupby('Prediction').value_counts(['cell_type'], normalize=True).sort_index().reset_index()
        cell_prop.columns = ['Prediction', 'cell_type', 'prop']
        plt.figure(figsize=(10, 6))
        sns.barplot(data=cell_prop, x='Prediction', y='prop', hue='cell_type')
        plt.savefig(os.path.join(config['figure_dir'], f"{config['spatial_dataset']}_cell_prop.png"))
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
