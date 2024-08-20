import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import os

def save_model(model, epoch, optimizer, save_dir):
    save_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(save_state, save_dir + "/ckpt_{}.pt".format(epoch))
    print("Model saved at epoch {} as {}".format(epoch-1, epoch))

def load_model(loadDir, loadEpoch, model, optimizer):
    loadDir = os.path.join(loadDir, "ckpt_{}.pt".format(loadEpoch))
    checkpoint = torch.load(loadDir)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("\nLoaded model from {}".format(loadDir))
    return epoch

def get_trainloader(trainBatch=256):
    transformTrain = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    trainDataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transformTrain
        )
    trainLoader = DataLoader(
        trainDataset, 
        batch_size=trainBatch, 
        shuffle=True, 
        num_workers=0,
        )
    return trainLoader

def get_testloader(testBatch=200, shuffle=False):
    transformTest = transforms.Compose([
        transforms.ToTensor(),
    ])
    testDataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transformTest
        )
    testLoader = DataLoader(
        testDataset,
        batch_size=testBatch, 
        shuffle=shuffle, 
        num_workers=0)
    return testLoader

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
