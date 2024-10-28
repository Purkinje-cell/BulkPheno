import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import TCGADataset, collate_fn
from model import BulkVAE
from trainer import VAETrainer

target_label = "tumor_stage_label"
train_dataset = TCGADataset(
    "data/Cleaned_data/Liver/train_exp.csv",
    "data/Cleaned_data/Liver/train_label.csv",
    target_label,
)
test_dateset = TCGADataset(
    "data/Cleaned_data/Liver/test_exp.csv",
    "data/Cleaned_data/Liver/test_label.csv",
    target_label,
)
# train_dataset = TCGADataset('data/Cleaned_data/Colon/train_exp.csv', 'data/Cleaned_data/Colon/train_label.csv', target_label)
# test_dateset = TCGADataset('data/Cleaned_data/Colon/test_exp.csv', 'data/Cleaned_data/Colon/test_label.csv', target_label)

print(len(train_dataset))
print(len(test_dateset))
n_input = 397
n_hidden = 128
n_latent = 64
n_labels = 5
n_batch = train_dataset.n_batch
batch_size = 64
n_epoch = 400

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dateset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

# 定义模型
model = BulkVAE(
    n_input=n_input,
    n_batch=n_batch,
    n_hidden=n_hidden,
    n_latent=n_latent,
    n_labels=n_labels,
    n_layers=2,
    dropout_rate=0.2,
)

optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
train_acc = []
test_acc = []
recon_losses = []
kl_zs = []
kl_ls = []
classification_losses = []
max_acc = 0
for i in range(n_epoch):
    total_correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        data_batch, target_batch, batch_index, mean, variance = batch
        recon_loss, kl_z, kl_l, classification_loss, probs = model(
            data_batch, mean, variance, batch_index, target_batch
        )
        mean_recon_loss = torch.mean(recon_loss)
        mean_kl_z = torch.mean(kl_z)
        mean_kl_l = torch.mean(kl_l)
        # total_loss = classification_loss * 10
        if i < 200:
            total_loss = mean_recon_loss + mean_kl_z + mean_kl_l * 0.1
        else:
            total_loss = (
                classification_loss * 50 + mean_recon_loss + mean_kl_z + mean_kl_l * 0.1
            )
        total_loss.backward()
        optimizer.step()
        total_correct += torch.sum(torch.argmax(probs, dim=1) == target_batch)
    recon_losses.append(mean_recon_loss.detach().numpy())
    kl_ls.append(mean_kl_l.detach().numpy())
    kl_zs.append(mean_kl_z.detach().numpy())
    classification_losses.append(classification_loss.detach().numpy())
    # print(f"Epoch: {i}, Loss: {total_loss}, Recon_loss: {mean_recon_loss}, KL_z: {mean_kl_z}, KL_l: {mean_kl_l}, Classification_loss: {classification_loss}")
    train_accu = total_correct
    print(f"Train Accuracy: {total_correct/len(train_dataset)}")

    train_acc.append(train_accu.detach().numpy() / len(train_dataset))

    total_correct = 0
    for batch in test_loader:
        data_batch, target_batch, batch_index, mean, variance = batch
        recon_loss, kl_z, kl_l, probs = model(data_batch, mean, variance, batch_index)
        prediction = torch.argmax(probs, dim=1)
        total_correct += torch.sum(prediction == target_batch)
    test_accu = total_correct
    print(f"Test Accuracy: {total_correct/len(test_dateset)}")
    if max_acc < total_correct / len(test_dateset):
        max_acc = total_correct / len(test_dateset)
        torch.save(model, "model.pth")
    test_acc.append(test_accu.detach().numpy() / len(test_dateset))
# VAE = VAETrainer(model)
# trainer = pl.Trainer(max_epochs=n_epoch)
# trainer.fit(VAE, train_loader, test_loader)
# trainer.test(VAE, test_loader)
print(max_acc)
# sns.lineplot(x=range(n_epoch), y=train_acc)
# sns.lineplot(x=range(n_epoch), y=test_acc)
# plt.show()

# recon_losses = np.array(recon_losses)
# kl_zs = np.array(kl_zs)
# kl_ls = np.array(kl_ls)
# classification_losses = np.array(classification_losses)
# plt.figure()
# sns.lineplot(x=range(20, n_epoch), y=recon_losses[20:], label='recon_loss')
# sns.lineplot(x=range(20, n_epoch), y=kl_zs[20:], label='kl_z')
# sns.lineplot(x=range(20, n_epoch), y=kl_ls[20:], label='kl_l')
# sns.lineplot(x=range(20, n_epoch), y=classification_losses[20:], label='classification_loss')
# plt.show()
# total_latent = []
# for batch in train_loader:
#     data_batch, target_batch, batch_index, mean, variance = batch
#     latent = model.encode(data_batch, mean, variance, batch_index)
#     total_latent.append(latent)

# for batch in test_loader:
#     data_batch, target_batch, batch_index, mean, variance = batch
#     recon_loss, kl_z, kl_l, probs = model(data_batch, mean, variance, batch_index)
#     prediction = torch.argmax(probs, dim=1)
#     print(prediction == target_batch)
#     print(torch.sum(prediction == target_batch)/len(target_batch))
# accuracy = torch.sum(prediction == test_dateset.label['median_surv'].astype('category').cat.codes.values) / len(test_dateset.target_label)
