import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import RandomSampler, DataLoader

# import cProfile

from models import *
from defender import beta_adv_train
from test import pgd_test
from pgd import PGD
from util import save_model, load_model
from plot import Plot
from logger import Logger

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define DNN
    model = ResNet18().to(device)

    # Get TRAIN dataset
    train_batch = 256
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
        )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch, 
        shuffle=True, 
        num_workers=0,
        )
    
    # Get TEST dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=200, 
        shuffle=True, 
        num_workers=0)

    # Define model parameters
    lr = 0.05 # Default 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )

    # Define hyperparameters
    eps = 8/255
    atk_iter = 10
    train_iter = None
    train_epoch = 2
    save = False
    load = False
    saveDir = "saved_models"
    saveDir = os.path.join(saveDir, "logger_test")
    logger = Logger(saveDir)

    if load:
        saved_epoch = 48
        train_epoch = load_model(saveDir, model, saved_epoch) + 1
    if save and not load:
        if os.path.isdir(saveDir):
            print("save_dir already exists! Exiting...")
            exit()
        else:
            os.mkdir(saveDir)

    pgd_adv = PGD(device, model)

    torch.backends.cudnn.benchmark = True
    # Train model
    for e in range(train_epoch):
        print("\nTrain Epoch: {}".format(e))
        beta_adv_train(
            train_loader,
            eps,
            model,
            device,
            train_iter,
            atk_iter,
            criterion,
            optimizer,
            logger,
            )
        
        if e % 2 == 0:
            print("\nTest Epoch: {}".format(e))
            pgd_test(model,
                    device, 
                    test_loader, 
                    criterion, 
                    pgd_adv,
                    logger,
                    # stopIter=25
                    )

        if save and e % 100 == 0:
            print("\nSave Epoch: {}".format(e))
            save_model(model, e, optimizer, saveDir)

    # print("\nLAST TEST")
    # pgd_test(model,
    #             device, 
    #             test_loader, 
    #             criterion, 
    #             pgd_adv,
    #             )

    plotter = Plot(saveDir, logger)
    plotter.draw_figure_losses()
    plotter.draw_figure_margins()