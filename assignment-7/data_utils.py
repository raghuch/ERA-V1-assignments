import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

default_train_transforms = transforms.Compose([
                                        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))]) #None
default_test_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]) #None

def get_augmented_MNIST_dataset(data_root, train_tfms=default_train_transforms, test_tfms=default_test_transforms, batch_sz=128, shuffle=True):                                                                                                                 
    trainset = datasets.MNIST(data_root, train=True, download=True, transform=train_tfms)
    testset = datasets.MNIST(data_root, train=False, download=True, transform=test_tfms)
    use_cuda = torch.cuda.is_available()
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_sz, num_workers=4, pin_memory=True) if use_cuda else dict(shuffle=shuffle, batch_size=64)


    train_loader = DataLoader(trainset, **dataloader_args)
    test_loader = DataLoader(testset, **dataloader_args)

    return train_loader, test_loader


def get_default_mnist_transforms():
   train_transforms = transforms.Compose([
      transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
      transforms.Resize((28, 28)),
      transforms.RandomRotation((-15., 15.), fill=0),
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
    ])

 # Test data transformations
   test_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
   return train_transforms, test_transforms


def get_mnist_dataset():
   train_transforms, test_transforms = get_default_mnist_transforms()
   train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
   test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

   return train_data, test_data

