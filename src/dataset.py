import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, num_workers=2, root="./data"):
    """
    Returns training and validation dataloaders for CIFAR-10.
    """
    training_transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    validation_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    training_set = datasets.CIFAR10(root=root, train=True, download=True, transform=training_transformer)
    validation_set = datasets.CIFAR10(root=root, train=False, download=True, transform=validation_transformer)

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return training_loader, validation_loader
