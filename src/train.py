import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import BulkVAE
from dataset import TCGADataset, collate_fn
from trainer import VAETrainer
import lightning as pl


train_dataset = TCGADataset('data/Cleaned_data/Liver/train_exp.csv', 'data/Cleaned_data/Liver/train_label.csv', 'median_surv')
test_dateset = TCGADataset('data/Cleaned_data/Liver/test_exp.csv', 'data/Cleaned_data/Liver/test_label.csv', 'median_surv')

n_input = 398
n_hidden = 128
n_latent = 32
n_labels = 2
n_batch = len(train_dataset)
batch_size = 32
n_epoch = 100

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dateset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 定义模型
model = BulkVAE(
    n_input=n_input,
    n_batch=n_batch,
    n_hidden=n_hidden,
    n_latent=n_latent,
    n_labels=n_labels,
    n_layers=2,
    dropout_rate=0.1,
)

optimizer = Adam(model.parameters(), lr=1e-2)
for i in range(n_epoch):
    for batch in train_loader:
        optimizer.zero_grad()
        data_batch, target_batch, batch_index, mean, variance = batch
        recon_loss, kl_z, kl_l, classification_loss, probs = model(data_batch, mean, variance, batch_index, target_batch)
        if i < 20:
            kl_weight = 0
        else:
            kl_weight = 1
        total_loss = torch.mean(recon_loss + (kl_z + kl_l) * kl_weight) + classification_loss * 1000
        total_loss.backward()
        optimizer.step()
        print(total_loss)
# VAE = VAETrainer(model)
# trainer = pl.Trainer(max_epochs=n_epoch)
# trainer.fit(VAE, train_loader, test_loader)
# trainer.test(VAE, test_loader)

for batch in test_loader:
    data_batch, target_batch, batch_index, mean, variance = batch
    recon_loss, kl_z, kl_l, probs = model(data_batch, mean, variance, batch_index)
    print(probs)
    prediction = torch.argmax(probs, dim=1)
    # print(prediction == target_batch)
    print(prediction)
#accuracy = torch.sum(prediction == test_dateset.label['median_surv'].astype('category').cat.codes.values) / len(test_dateset.target_label)
