import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from src.model import VAE, MLP
from src.dataset import TCGADataset

# 超参数
input_dim = 398
hidden_dim = 64
latent_dim = 32
batch_size = 64
epochs = 10
lr = 1e-5

# 加载数据
train_dataset = TCGADataset('Cleaned_data/Liver/train_exp.csv', 'Cleaned_data/Liver/train_label.csv', 'median_surv')
test_dateset = TCGADataset('Cleaned_data/Liver/test_exp.csv', 'Cleaned_data/Liver/test_label.csv', 'median_surv')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dateset, batch_size=batch_size, shuffle=False)

# 定义模型
model = VAE(input_dim, hidden_dim, latent_dim)
classifier = MLP(latent_dim, latent_dim, 2)

# 定义优化器
optimizer = Adam(model.parameters(), lr=lr)

# 训练
for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        x, y = data
        x = x.view(x.size(0), -1)
        print(x)
        
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss_vae = vae_loss(recon_x, x, mu, logvar)
        y_pred = classifier(mu)
        loss_classification = classification_loss(y_pred, y)
        loss = loss_vae + loss_classification

        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, len(train_loader), loss.item()))
            
# 保存模型
torch.save(model.state_dict(), 'vae.pth')

# 加载模型
model.load_state_dict(torch.load('vae.pth'))
model.eval()
