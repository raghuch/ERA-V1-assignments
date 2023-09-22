from typing import Any, Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from lightning import pytorch as pl

class ContractBlk(nn.Module):
    def __init__(self, in_channels, out_channels, use_maxpool=True, only_conv=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.only_conv = only_conv

        self.conv_blk = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        if use_maxpool:
            self.downscale = nn.MaxPool2d(2, 2)
        else:
            self.downscale = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=2, stride=2)
        #self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        res = self.conv_blk(x)
        # print("In contract block: ")
        # print(res.shape)
        if self.only_conv:
            out = res
        else:
            out = self.downscale(res)
        # print(out.shape)
        # print("____________________")
        return out, res
    
class ExpandBlk(nn.Module):
    def __init__(self, in_channels, out_channels, use_upsample=False) -> None:
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
        if use_upsample:
            self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.upscale = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

    def forward(self, x, res):
        x = self.upscale(x)
        conv = torch.cat((x, res), dim=1)
        out = self.conv_blk(conv)

        #conv = self.upscale(conv)
        #out = torch.cat((conv, res), dim=1)
        return out

    

class UNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels, n_filters=32, use_maxpool=True, use_upsample=False, calc_dice_loss=False) -> None:
        super().__init__()
        self.calc_dice_loss = calc_dice_loss
        self.contract1 = ContractBlk(in_channels, n_filters, use_maxpool)
        self.contract2 = ContractBlk(n_filters, n_filters * 2, use_maxpool)
        self.contract3 = ContractBlk(n_filters*2, n_filters*4, use_maxpool)
        self.contract4 = ContractBlk(n_filters*4, n_filters*8, use_maxpool)
        self.contract5 = ContractBlk(n_filters*8, n_filters*16, use_maxpool, only_conv=True)

        self.expand1 = ExpandBlk(n_filters*16, n_filters*8, use_upsample)
        self.expand2 = ExpandBlk(n_filters*8, n_filters*4, use_upsample)
        self.expand3 = ExpandBlk(n_filters*4, n_filters*2, use_upsample)
        self.expand4 = ExpandBlk(n_filters*2, n_filters, use_upsample)

        self.final_conv1 = nn.Conv2d(n_filters, in_channels, kernel_size=1)
        self.final_conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)


    def forward(self, x):
        x1, res1 = self.contract1(x)
        x2, res2 = self.contract2(x1)
        x3, res3 = self.contract3(x2)
        x4, res4 = self.contract4(x3)
        x5, res5 = self.contract5(x4)

        x6 = self.expand1(x5, res4)
        x7 = self.expand2(x6, res3)
        x8 = self.expand3(x7, res2)
        x9 = self.expand4(x8, res1)
        x9 = self.final_conv1(x9)
        return self.final_conv2(x9)
    
    # def data_prep(self) -> None:
    #     self.train_data = datasets.OxfordIIITPet(root="~/work/data/", download=True, split='trainval', target_types='segmentation', transform=transforms.ToTensor())
    #     self.val_data = datasets.OxfordIIITPet(root="~/work/data/", download=True, split='test' ,target_types='segmentation', transform=transforms.ToTensor())

    # def setup(self, stage: str) -> None:

    #     if stage == 'train':

        
    
    # def train_dataloader(self) -> TRAIN_DATALOADERS:
    #     return torch.utils.data.DataLoader(self.train_data, batch_size=128, shuffle=True)
    
    
    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return torch.utils.data.DataLoader(self.val_data, batch_size=128, shuffle=False)
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer
    
    def _dice_loss(self, y_true, y_pred, smooth=1e-3):
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = (y_true_f * y_pred_f).sum()
        union = y_pred_f.sum() + y_true_f.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        if self.calc_dice_loss:
            loss = self._dice_loss(y, y_hat)
        else:
            loss = nn.BCEWithLogitsLoss()(y_hat, y)
        #loss = F.cross_entropy(y_hat, y)
        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        if self.calc_dice_loss:
            loss = self._dice_loss(y, y_hat)
        else:
            loss = nn.BCEWithLogitsLoss()(y_hat, y)
        return {'val_loss': loss}
    
    




