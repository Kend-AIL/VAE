import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CIFAR100ImagesOnly(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx] 

        return {'image': image, 'label': label}


def get_data(batch_size):
    """
    获取 CIFAR-100 数据集
    :param batch_size: 每个批次的大小
    :param mask: 是否应用遮罩
    :param mask_ratio: 遮罩的像素比例
    :return: DataLoader 实例
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = CIFAR100ImagesOnly(root='../data', train=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
    val_dataset = CIFAR100ImagesOnly(root='../data', train=False, transform=transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,num_workers=16, shuffle=True)
    return train_dataloader,val_dataloader
