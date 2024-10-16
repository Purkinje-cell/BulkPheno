import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import BulkVAE
from dataset import TCGADataset, collate_fn
from trainer import VAETrainer
import lightning as pl


train_dataset = TCGADataset('data/Cleaned_data/Liver/train_exp.csv', 'data/Cleaned_data/Liver/train_label.csv', 'sample_type')
test_dateset = TCGADataset('data/Cleaned_data/Liver/test_exp.csv', 'data/Cleaned_data/Liver/test_label.csv', 'sample_type')

n_input = 398
n_hidden = 192
n_latent = 64
n_labels = 2
n_batch = train_dataset.n_batch
batch_size = 64
n_epoch = 200

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dateset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 定义模型
model = BulkVAE(
    n_input=n_input,
    n_batch=n_batch,
    n_hidden=n_hidden,
    n_latent=n_latent,
    n_labels=n_labels,
    n_layers=3,
    dropout_rate=0.2,
)

optimizer = Adam(model.parameters(), lr=1e-3)
for i in range(n_epoch):
    for batch in train_loader:
        optimizer.zero_grad()
        data_batch, target_batch, batch_index, mean, variance = batch
        recon_loss, kl_z, kl_l, classification_loss, probs = model(data_batch, mean, variance, batch_index, target_batch)
        mean_recon_loss = torch.mean(recon_loss)
        mean_kl_z = torch.mean(kl_z)
        mean_kl_l = torch.mean(kl_l)
        total_loss = mean_recon_loss + mean_kl_z + mean_kl_l + classification_loss * 10
        total_loss.backward()
        optimizer.step()
    print(f"Epoch: {i}, Loss: {total_loss}, Recon_loss: {mean_recon_loss}, KL_z: {mean_kl_z}, KL_l: {mean_kl_l}, Classification_loss: {classification_loss}, probs: {probs}")
    prediction = torch.argmax(probs, dim=1)
    print(prediction == target_batch)
    print(torch.sum(prediction == target_batch)/len(target_batch))
# VAE = VAETrainer(model)
# trainer = pl.Trainer(max_epochs=n_epoch)
# trainer.fit(VAE, train_loader, test_loader)
# trainer.test(VAE, test_loader)
torch.save(model, 'model.pth')
# total_latent = []
# for batch in train_loader:
#     data_batch, target_batch, batch_index, mean, variance = batch
#     latent = model.encode(data_batch, mean, variance, batch_index)
#     total_latent.append(latent)

for batch in test_loader:
    data_batch, target_batch, batch_index, mean, variance = batch
    recon_loss, kl_z, kl_l, probs = model(data_batch, mean, variance, batch_index)
    prediction = torch.argmax(probs, dim=1)
    print(prediction == target_batch)
    print(torch.sum(prediction == target_batch)/len(target_batch))
#accuracy = torch.sum(prediction == test_dateset.label['median_surv'].astype('category').cat.codes.values) / len(test_dateset.target_label)
