import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

dropout_val = 0.15

class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        #Input Block
        self.conv_blk1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout_val)
        ) #output_sz = 26

        #Conv block 1
        self.conv_blk2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout_val)
        )  #output_sz = 24 

        #Transition Block
        self.conv_blk3 = nn.Sequential(
            nn.Conv2d(32, 20, kernel_size=1, padding=0, bias=False)
        ) #output_sz = 24
        self.maxpool1 = nn.MaxPool2d(2,2) #output_sz = 12

        #Conv block 2
        self.conv_blk4 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout_val)
        )  #output_sz = 10

        #Conv block 3
        self.conv_blk5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout_val)
        )  #output_sz = 8

        #Conv block 4
        self.conv_blk6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout_val)
        )  #output_sz = 6

        #Output block - global avg pooling + output conv
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )
        self.conv_blk7 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, padding=0, bias=False)
        )


    def forward(self, x):
        x = self.conv_blk1(x)
        x = self.conv_blk2(x)
        x = self.conv_blk3(x)
        x = self.maxpool1(x)
        x = self.conv_blk4(x)
        x = self.conv_blk5(x)
        x = self.conv_blk6(x)
        x = self.gap(x)
        x = self.conv_blk7(x)
        
        x = x.view(-1,10)

        return F.log_softmax(x, dim=-1)