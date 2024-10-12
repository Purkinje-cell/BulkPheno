import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning as pl
from model import BulkVAE

class VAETrainer(pl.LightningModule):
    def __init__(self, VAE: BulkVAE):
        super(VAETrainer, self).__init__()
        self.VAE = VAE
    
    def training_step(self, batch):
        data_batch, target_batch, batch_index, mean, variance = batch
        if target_batch is not None:
            recon_loss, kl_z, kl_l, classification_loss, probs = self.VAE(data_batch, mean, variance, batch_index, target_batch)
            loss = (
                torch.mean(recon_loss + kl_z + kl_l)
            )
            loss += classification_loss
        else:
            recon_loss, kl_z, kl_l, probs = self.VAE(data_batch, mean, variance, batch_index)
            loss = torch.mean(recon_loss + kl_z + kl_l)
        return loss
    
    def test_step(self, batch):
        data_batch, target_batch, batch_index, mean, variance = batch
        recon_loss, kl_z, kl_l, classification_loss, probs = self.VAE(data_batch, mean, variance, batch_index, target_batch)
        loss = recon_loss + kl_z + kl_l + classification_loss
        return loss
    
    def prediction_step(self, batch):
        data_batch, target_batch, batch_index, mean, variance = batch
        _, _, _, classification_loss, probs = self.VAE(data_batch, mean, variance, batch_index, target_batch)
        return classification_loss, probs
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.VAE.parameters(), lr=1)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return [optimizer], [lr_scheduler]
