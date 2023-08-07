import torch
import torch.nn as nn
import torch.nn.functional as F

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, padding=1) -> None:
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.stride = stride
#         self.padding = padding



class customResNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=10) -> None:
        super().__init__()
        self.in_channels = in_channels
        #self.out_channels = out_channels
        self.n_classes = n_classes

        self.prep_blk = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.X1 = self._make_X_block(in_channels=64, out_channels=128)
        self.R1 = self._make_res_block(in_channels=64, out_channels=128, stride=2)
        self.layer2 = self._make_X_block(in_channels=128, out_channels=256)
        self.X3 = self._make_X_block(in_channels=256, out_channels=512)
        self.R3 = self._make_res_block(in_channels=256, out_channels=512, stride=2)

        self.maxpool_f = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, self.n_classes)
        self.flatten = nn.Flatten()

        
    def forward(self, x):
        x = self.prep_blk(x)
        x1 = self.X1(x) 
        x1_res = self.R1(x)
        print(x1.shape)
        print(x1_res.shape)
        x1 = x1 + x1_res
        x2 = self.layer2(x1)
        x3 = self.X3(x2) 
        x3_res =  self.R3(x2)
        x3 = x3 + x3_res
        x4 = self.maxpool_f(x3)
        out = self.fc(self.flatten(x4))
        return out
    
    @staticmethod
    def _make_X_block(in_channels, out_channels, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),stride=stride, padding=padding),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    @staticmethod
    def _make_res_block(in_channels, out_channels, stride=1, padding=1):
        mid_channels = out_channels
        custom_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3,3), stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3,3), stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        return custom_block