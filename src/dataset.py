import torch
import pandas as pd
from torch.utils.data import Dataset

class TCGADataset(Dataset):
    def __init__(self, expression_mtx_path, label_path, label_type):
        self.expression_mtx = pd.read_csv(expression_mtx_path, index_col=0)
        self.label = pd.read_csv(label_path, index_col=0)
        self.data = torch.tensor(self.expression_mtx.values, dtype=torch.float32)
        self.target = torch.tensor(self.label[label_type].astype(bool).values, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.target[idx]
        return x, y