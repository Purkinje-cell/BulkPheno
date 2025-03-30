import numpy as np
import scvi
import torch.nn
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.dataloaders import AnnDataLoader
import torch
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import squidpy as sq
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import pytorch_lightning as pl
import os
import sys
import argparse
plt.style.use('default')
sys.path.append("../src")  # 将 src 目录添加到 Python 路径
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from lightning.pytorch.loggers import WandbLogger
from dataset import GraphContrastiveDataModule, SpatialGraphDataset
from torch_geometric.loader import DataLoader
from model import GraphContrastiveModel


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess spatial transcriptomics data')
    parser.add_argument('--input', type=str, help='Path to input h5ad file')
    parser.add_argument('--input_dir', type=str, help='Directory containing h5ad files')
    parser.add_argument('--output_dir', type=str, default='../data/processed', 
                      help='Output directory for processed graphs')
    parser.add_argument('--hops', type=int, default=3,
                      help='Number of hops for neighborhood graph construction')
    parser.add_argument('--tcga_path', type=str, default='../data/TCGA/TCGA-LIHC.htseq_counts_clean.tsv',
                      help='Path to TCGA expression data')
    return parser.parse_args()


def process_adata(adata_path, output_dir, hops=3, tcga_path='../data/TCGA/TCGA-LIHC.htseq_counts_clean.tsv'):
    """Process a single AnnData file"""
    # Read and process TCGA data
    TCGA_LIHC = pd.read_table(tcga_path, index_col=0)
    
    # Load and process spatial data
    adata = sc.read_h5ad(adata_path)
    common_genes = np.intersect1d(adata.var_names, TCGA_LIHC.index.values)
    adata = adata[:, common_genes]
    
    # Create directory name from file stem
    file_stem = os.path.splitext(os.path.basename(adata_path))[0]
    graph_output_dir = os.path.join(output_dir, f"graphs_{file_stem}")
    
    # Create graph dataset
    graph_dataset = SpatialGraphDataset(
        adata, 
        name=file_stem,
        hops=hops,
        root=output_dir
    )
    
    print(f'Processed {adata_path} -> {graph_output_dir}')
    print(f'Graph dataset length: {len(graph_dataset)}')
    return graph_dataset
if __name__ == '__main__':
    args = parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process TCGA data first
    TCGA_LIHC = pd.read_table(args.tcga_path, index_col=0)
    exp_data = TCGA_LIHC.T
    clinical_data = pd.read_table('../data/TCGA/TCGA-LIHC.GDC_phenotype.tsv', index_col=0)
    survival_data = pd.read_table('../data/TCGA/TCGA-LIHC.survival.tsv', index_col=0)
    clinical_data = survival_data.merge(clinical_data, left_index=True, right_index=True)
    common_sample = np.intersect1d(exp_data.index, clinical_data.index)
    exp_data = exp_data.loc[common_sample]
    clinical_data = clinical_data.loc[common_sample]
    
    # Ensure output directory exists
    os.makedirs('../data/TCGA/Processed', exist_ok=True)
    clinical_data.columns.to_series().to_csv('../data/TCGA/Processed/clinical_data.csv', index=False)
    
    bulk_TCGA_adata = sc.AnnData(X=exp_data.values, obs=clinical_data)
    bulk_TCGA_adata.write_h5ad('../data/TCGA/Processed/TCGA_LIHC.h5ad')

    # Process spatial data
    if args.input:
        process_adata(args.input, args.output_dir, args.hops, args.tcga_path)
    elif args.input_dir:
        for fname in os.listdir(args.input_dir):
            if fname.endswith('.h5ad'):
                process_adata(
                    os.path.join(args.input_dir, fname),
                    args.output_dir,
                    args.hops,
                    args.tcga_path
                )
    else:
        # Default behavior - process HC1 file
        default_input = '../data/MERFISH/HC1_processed.h5ad'
        print(f"No input specified. Processing default file: {default_input}")
        
        use_region = sc.read_h5ad(default_input)
        common_genes = np.intersect1d(use_region.var_names, TCGA_LIHC.index.values)
        use_region = use_region[:, common_genes]
        sc.pl.embedding(use_region, basis='spatial', color='sub_cell_type')
        sc.pp.neighbors(use_region, n_neighbors=20, use_rep='spatial')
        import scipy.sparse as sp
        use_region.obsp['spatial_connectivities'] = sp.coo_matrix((use_region.obsp['distances'] > 0).astype(int))
        use_region.obsp['spatial_distances'] = use_region.obsp['distances']
        graph_dataset = SpatialGraphDataset(use_region, name='HC1_processed', hops=3, root=args.output_dir)
        print(f'Length of graph dataset is {len(graph_dataset)}')
