from torch_geometric.data import DataLoader, InMemoryDataset
from model import Subgraph
from torch_geometric.utils import from_scipy_sparse_matrix

import pandas as pd
import scanpy as sc
import torch
from torch_geometric.data import Data
class GraphDataset(InMemoryDataset):
    def __init__(self, root, adata_dir, label_dir, training=True, transform=None, pre_transform=None):
        self.adata_dir = adata_dir
        self.label_dir = label_dir
        self.training = training
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.label_dir]

    @property
    def processed_file_names(self):
        if self.training:
            return ['training.pt']
        else:
            return ['test.pt']

    def download(self):
        # Download your raw data here, if necessary.
        pass

    def process(self):
        # Read the labels
        labels = pd.read_csv(self.label_dir)
        labels = labels.query('isPerProtocol == True & Arm == "C&I" & BiopsyPhase != "Post-treatment"')
        labels['pCR'] = labels['pCR'].map({'pCR': 1, 'RD': 0})
        remained_images = labels['ImageID'].values
        
        # Create a list of Data objects
        data_list = []
        for image_id in remained_images:
            adata = sc.read_h5ad(f'{self.adata_dir}/{image_id}.h5ad')
            edge_idx, edge_attr = from_scipy_sparse_matrix(adata.obsp['spatial_connectivities'])
            y = labels.query(f'ImageID == "{image_id}"')['pCR'].values[0]
            feature = torch.tensor(adata.X, dtype=torch.float32)
            graph_data = Data(x=feature, edge_index=edge_idx, edge_attr=edge_attr, y=torch.tensor([y], dtype=torch.long))
            data_list.append(graph_data)

        # Process data and store it
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Ensure to adjust root, adata_dir, label_dir to your configurations
# dataset = GraphDataset(root='..', adata_dir='../data/TNBC_adata', label_dir='../data/Cleaned_data/TNBC/clinical_data.csv')
train_dataset = GraphDataset(root='.', adata_dir='data/TNBC_adata', label_dir='./data/TNBC_raw/train_labels.csv')
test_dataset = GraphDataset(root='.', adata_dir='./data/TNBC_adata', label_dir='./data/TNBC_raw/test_labels.csv', training=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Subgraph(gcn_first=16, gcn_second=16, fc_1=32, fc_2=2, cls_hidden=16, number_of_features=46)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1)

train_accs = []
test_accs = []
gradient_accumulation_steps = 8
for epoch in range(100):  # Example: 100 epochs
    model.train()
    total_loss = 0
    train_acc_num = 0
    test_acc_num = 0
    gradient_i = 0
    for data in train_dataset:
        data = data.to(device)
        _, _, _, kl_loss, cls_loss, positive_penalty, preserve_rate, correct_num = model(data)
        loss =  cls_loss + kl_loss * 0.001 + positive_penalty * 5
        train_acc_num += correct_num
        total_loss += loss
        gradient_i += 1
        if gradient_i % gradient_accumulation_steps == 0:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    train_acc = train_acc_num / len(train_dataset)
    for data in test_dataset:
        data = data.to(device)
        _, _, _, _, _, _, _, correct_num = model(data)
        test_acc_num += correct_num
    test_acc = test_acc_num / len(test_dataset)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f'Epoch {epoch}, Loss: {total_loss.item()}, Train Acc: {train_acc}, Test Acc: {test_acc}')
    
