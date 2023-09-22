from turtle import forward
from typing import Any, Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from lightning import pytorch as pl

class ContractBlk(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_blk = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        res = self.conv_blk(x)
        out = self.maxpool(res)
        return out, res
    
class ExpandBlk(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_blk = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        #self.upscale = nn.Upsample(scale_factor=2)
        self.upscale = nn.ConvTranspose2d(out_channels, out_channels //2, kernel_size=2, stride=2)

    def forward(self, x, res):
        conv = self.expand_blk(x)
        out = torch.cat((self.upscale(conv), res), dim=1)
        return out

    
# class contract_path(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.contract_ladder = [64, 128, 256, 512]

#         prev_depth = 1

#         self.contract_path = nn.Sequential()
#         contract_blks = []
#         for c in self.contract_ladder:
#             self.contract_blks.append(contract_convblk(prev_depth, c))
#             prev_depth = c

#         self.contract_path = nn.Sequential(*contract_blks)


#     def forward(self, x):
#         return self.contract_path(x)
    

class UNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels, n_filters=32) -> None:
        super().__init__()
        self.contract1 = ContractBlk(in_channels, n_filters)
        self.contract2 = ContractBlk(n_filters, n_filters * 2)
        self.contract3 = ContractBlk(n_filters*2, n_filters*4)
        self.contract4 = ContractBlk(n_filters*4, n_filters*8)
        self.contract5 = ContractBlk(n_filters*8, n_filters*16)

        self.expand1 = ExpandBlk(n_filters*16, n_filters*8)
        self.expand2 = ExpandBlk(n_filters*8, n_filters*4)
        self.expand3 = ExpandBlk(n_filters*4, n_filters*2)
        self.expand4 = ExpandBlk(n_filters*2, n_filters)


    def forward(self, x):
        x, res1 = self.contract1(x)
        x, res2 = self.contract2(x)
        x, res3 = self.contract3(x)
        x, res4 = self.contract4(x)
        x, res5 = self.contract5(x)
        x = self.expand1(x, res4)
        x = self.expand2(x, res3)
        x = self.expand3(x, res2)
        x = self.expand4(x, res1)
        return x
    
    def data_prep(self) -> None:
        self.train_data = datasets.OxfordIIITPet(root="~/work/data/", download=True, split='trainval', target_types='segmentation', transform=transforms.ToTensor())
        self.val_data = datasets.OxfordIIITPet(root="~/work/data/", download=True, split='test' ,target_types='segmentation', transform=transforms.ToTensor())
        
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(self.train_data, batch_size=128, shuffle=True)
    
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.val_data, batch_size=128, shuffle=False)
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss}
    
    




