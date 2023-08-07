import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt

cifar_label_idx_to_name = ["airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"]

default_train_transforms = A.Compose([
    A.PadIfNeeded(min_height=40, min_width=40),
    A.RandomCrop(32, 32),
    A.HorizontalFlip(),
    A.CoarseDropout(1, 8,8, 1, 8,fill_value=0.473363, mask_fill_value=None)
])

default_test_transforms = A.Compose([
    A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
])

class AugmentedCIFAR10(Dataset):
    def __init__(self, img_lst, train=True, train_tfms=default_train_transforms, test_tfms=default_test_transforms):
        super().__init__()
        self.img_lst = img_lst
        self.train = train
        self.transforms = train_tfms

        self.norm = test_tfms

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx):
        img, label = self.img_lst[idx]

        if self.train:
            img = self.transforms(image=np.array(img))["image"]
        else:
            img = self.norm(image=np.array(img))["image"]

        img = np.transpose(img, (2,0,1)).astype(np.float32)

        return torch.tensor(img, dtype=torch.float), label
    

def get_augmented_cifar10_dataset(data_root, train_tfms=default_train_transforms, test_tfms=default_test_transforms, batch_sz=128, shuffle=True):
    trainset = datasets.CIFAR10(data_root, train=True, download=True) #, transform=train_transforms)
    testset = datasets.CIFAR10(data_root, train=False, download=True) #, transform=test_transforms)
    use_cuda = torch.cuda.is_available()
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_sz, num_workers=4, pin_memory=True) if use_cuda else dict(shuffle=shuffle, batch_size=64)

    train_loader = torch.utils.data.DataLoader(AugmentedCIFAR10(trainset, train=True, train_tfms=train_tfms), **dataloader_args)
    test_loader = torch.utils.data.DataLoader(AugmentedCIFAR10(testset, train=False, test_tfms=test_tfms), **dataloader_args)

    return train_loader, test_loader
    
def display_imgs(img_lst, label_lst, correct_label_lst):
    fig = plt.figure(figsize=(10, 6))
    rows = 5 
    cols = 4
    for idx in np.arange(1, rows*cols + 1):
        ax = fig.add_subplot(rows, cols, idx)
        ax.set_title(f"predicted: {cifar_label_idx_to_name[label_lst[idx].squeeze()]} (label: {cifar_label_idx_to_name[correct_label_lst[idx].squeeze()]})",
                     fontdict={'fontsize':8})
        ax.axis('off')
        img = img_lst[idx]
        img = img/2 + 0.5
        img = np.clip(img, 0, 1)
        plt.imshow(img.transpose(1,2,0))
    #plt.subplots
    plt.show()

def display_imgs_gradcam(img_lst, label_lst, correct_label_lst, vis_lst):
    fig = plt.figure(figsize=(10, 6))
    rows = 5 
    cols = 4
    for idx in np.arange(1, rows*cols + 1):
        ax = fig.add_subplot(rows, cols, idx)
        ax.set_title(f"predicted: {cifar_label_idx_to_name[label_lst[idx].squeeze()]} (label: {cifar_label_idx_to_name[correct_label_lst[idx].squeeze()]})",
                     fontdict={'fontsize':8})
        ax.axis('off')
        img = img_lst[idx]
        img = img/2 + 0.5
        img = np.clip(img, 0, 1)
        plt.imshow(img.transpose(1,2,0))
        plt.imshow(vis_lst[idx], alpha=0.5)
    #plt.subplots
    plt.show()


def get_misclassified_imgs(model, test_dataloader):
    incorrect_imgs = []
    incorrect_labels = []
    incorrect_preds = []

    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    for img, tgt in test_dataloader:
        img = img.to(device)
        tgt = tgt.to(device)

        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)

        for idx in range(img.shape[0]):
            if tgt[idx] != pred[idx]:
                incorrect_imgs.append(img[idx].cpu().numpy())
                incorrect_preds.append(tgt[idx].cpu().numpy())
                incorrect_labels.append(pred[idx].cpu().numpy())

    return incorrect_imgs, incorrect_labels, incorrect_preds
