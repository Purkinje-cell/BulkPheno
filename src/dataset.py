import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import one_hot


class TCGADataset(Dataset):
    """
    Dataset for TCGA bulk expression data
    """

    def __init__(self, expression_mtx_path, label_path, label_type):
        self.expression_mtx = pd.read_csv(
            expression_mtx_path, index_col=0
        )  # (n_sample, n_gene)
        self.label = pd.read_csv(label_path, index_col=0)
        self.target_label = self.label[label_type].astype("category")
        self.n_cat = len(self.target_label.cat.categories)
        self.target_label = torch.tensor(
            self.target_label.cat.codes.values, dtype=torch.long
        )
        self.data = torch.tensor(
            self.expression_mtx.values, dtype=torch.float32
        )  # (n_sample, n_gene)
        self.batch_index = torch.tensor(
            self.label["batch_number"].astype("category").cat.codes.values,
            dtype=torch.long,
        )
        self.n_batch = len(self.label["batch_number"].unique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.target_label[idx]
        batch_index = self.batch_index[idx]
        return x, y, batch_index


def collate_fn(batch):
    data_batch, target_batch, batch_index = zip(*batch)

    batch_index = torch.tensor(batch_index, dtype=torch.float32).view(-1, 1)
    data_batch = torch.stack(data_batch, dim=0)
    target_batch = torch.stack(target_batch, dim=0)
    library_sizes = torch.log1p(torch.mean(data_batch, dim=1))

    mean = torch.mean(library_sizes).expand(data_batch.size(0))
    variance = torch.var(library_sizes).expand(data_batch.size(0))

    return data_batch, target_batch, batch_index, mean, variance
