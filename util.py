import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import os

def save_model(model, epoch, optimizer, save_dir):
    save_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': self.optimizer.state_dict(),
    }
    torch.save(save_state, save_dir + "/ckpt_{}.pt".format(epoch))
    print("Model saved at epoch {}".format(epoch))

def load_model(load_dir, model, epoch):
    load_dir = os.path.join(load_dir, "ckpt_{}.pt".format(epoch))
    checkpoint = torch.load(load_dir)
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("\nLoaded model from {}".format(load_dir))
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