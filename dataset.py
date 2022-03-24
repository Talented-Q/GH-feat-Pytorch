import torchvision
from torchvision import transforms
from random import random
import torch.nn as nn
import torch
from torch.utils import data
from PIL import Image
from functools import partial
from pathlib import Path
EXTS = ['jpg', 'jpeg', 'png']

def exists(val):
    return val is not None

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else  # 根据概率，随机操作
        return fn(x)

def convert_rgb_to_transparent(image):
    # 将RGB转化为潜在代码
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image


def convert_transparent_to_rgb(image):
    # 将潜在代码转化为RGB
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

def resize_to_minimum_size(min_size, image):
    # 将图像调整到模型能输入的最小尺寸
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

def get_dataset(image_size, name='ffhq', BATCH=4, transparent=False, aug_prob=0.):

    convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
    num_channels = 3 if not transparent else 4

    trasnformers_train = transforms.Compose([
        transforms.Lambda(convert_image_fn),
        transforms.Lambda(partial(resize_to_minimum_size, image_size)),
        transforms.Resize(image_size),
        RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                    transforms.CenterCrop(image_size)),
        transforms.ToTensor(),
        transforms.Lambda(expand_greyscale(transparent))
    ])

    trasnformers_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    if name == 'mnist':
        train_ds = torchvision.datasets.MNIST(root=r'G:\Database',train=True,trasnform=trasnformers_train)
        test_ds = torchvision.datasets.MNIST(root=r'G:\Database',train=False,trasnform=trasnformers_test)
    elif name == 'ffhq':
        train_ds = torchvision.datasets.ImageFolder(r'C:\Dataset\Ffhq', transform=trasnformers_train)
        test_ds = torchvision.datasets.ImageFolder(r'C:\Dataset\Ffhq', transform=trasnformers_test)
    elif name == 'imagenet':
        train_ds = torchvision.datasets.ImageFolder(r'C:\ILSVRC2012\train', transform=trasnformers_train)
        test_ds = torchvision.datasets.ImageFolder(r'C:\ILSVRC2012\val', transform=trasnformers_test)
    elif name == 'cifar10':
        train_ds = torchvision.datasets.CIFAR10(root=r'G:\Database', train=True, trasnform=trasnformers_train)
        test_ds = torchvision.datasets.CIFAR10(root=r'G:\Database', train=False, trasnform=trasnformers_test)
    elif name == 'cifar100':
        train_ds = torchvision.datasets.CIFAR100(root=r'G:\Database', train=True, trasnform=trasnformers_train)
        test_ds = torchvision.datasets.CIFAR100(root=r'G:\Database', train=False, trasnform=trasnformers_test)
    elif name == 'bird':
        train_ds = torchvision.datasets.ImageFolder(r'C:\Users\Tom.riddle\.conda\envs\torch1.9\project\Dataset\bird\bird\train', transform=trasnformers_train)
        test_ds = torchvision.datasets.ImageFolder(r'C:\Users\Tom.riddle\.conda\envs\torch1.9\project\Dataset\bird\bird\val', transform=trasnformers_test)



    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCH, shuffle=True, drop_last=True)

    length_train = len(train_dl)
    length_test = len(test_dl)

    return train_dl, test_dl, length_train, length_test