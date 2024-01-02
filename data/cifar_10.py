import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CIFAR10ImagesOnly(Dataset):
    def __init__(self, root, train=True, transform=None, mask=False, mask_ratio=0.5):
        """
        初始化函数
        :param root: 数据集的根目录
        :param train: 是否加载训练集（True）或测试集（False）
        :param transform: 应用于每个图像的转换
        :param mask: 是否应用遮罩
        :param mask_ratio: 遮罩的像素比例，取值范围 [0, 1]
        """
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.mask = mask
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # 忽略标签

        # 如果启用遮罩，则应用遮罩
        if self.mask:
            masked_image = self.apply_mask(image)
        else:
            masked_image = image

        return {'image': image, 'mask_image': masked_image}

    def apply_mask(self, image):
        """
        应用遮罩到图像
        :param image: 要遮罩的图像
        :return: 遮罩后的图像
        """
        image = np.array(image)  # 将 PIL 图像转换为 NumPy 数组
        mask_area = np.random.choice([0, 1], size=image.shape[:2], p=[self.mask_ratio, 1 - self.mask_ratio])
        mask_area = np.expand_dims(mask_area, axis=-1)
        return torch.tensor(mask_area * image, dtype=torch.float32)  # 将遮罩应用于图像并转换回 torch 张量

def get_cifar10_data(batch_size, mask=False, mask_ratio=0.5):
    """
    获取 CIFAR-10 数据集
    :param batch_size: 每个批次的大小
    :param mask: 是否应用遮罩
    :param mask_ratio: 遮罩的像素比例
    :return: DataLoader 实例
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = CIFAR10ImagesOnly(root='../data', train=True, transform=transform, mask=mask, mask_ratio=mask_ratio)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
    val_dataset = CIFAR10ImagesOnly(root='../data', train=False, transform=transform, mask=mask, mask_ratio=mask_ratio)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
    return train_dataloader, val_dataloader
