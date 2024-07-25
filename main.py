import os
import torch
import torch.nn as nn
import torch.optim as optim

from multiprocessing import Process

# import cProfile

from models import *
from defender import beta_adv_train
from test import pgd_test
from pgd import PGD
from util import save_model, load_model, get_trainloader, get_testloader
from plot import Plot
from logger import Logger

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18().to(device)
    trainLoader = get_trainloader(trainBatch=512)
    testLoader = get_testloader(testBatch=400, shuffle=False)
    pgdAdv = PGD(model)

    # Define hyperparameters
    eps = 8/255
    lr = 0.1 # Default 0.1 (used 0.05)
    atkIter = 10
    trainEpochStart, trainEpochEnd = 0, 50
    save = True
    load = False
    loadEpoch = 50 # Only used if load == True
    saveDir = "saved_models"
    saveDir = os.path.join(saveDir, "new_bilevel")
    trainPGD = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )

    if load:
        load_model(saveDir, model, loadEpoch)
        trainEpochStart = loadEpoch
        trainEpochEnd += trainEpochStart + 50
        saveDir += "_" + str(trainEpochStart)
    if save or load:
        if os.path.isdir(saveDir):
            exit("saveDir already exists! Exiting...")
        os.mkdir(saveDir)
        print("Made model save directory at " + saveDir)
        logger = Logger(saveDir)
    else:
        logger = Logger(None)

    # Train model
    print("Training from {} to {}".format(trainEpochStart, trainEpochEnd))
    for e in range(trainEpochStart, trainEpochEnd):
        print("\nTrain Epoch: {}".format(e))
        beta_adv_train(
            trainLoader, eps, model, device,
            atkIter, criterion, optimizer, logger,
            trainPGD
            )
        
        if e % 1 == 0:
            print("\nTest Epoch: {}".format(e))
            pgd_test(
                model, testLoader, 
                criterion, pgdAdv, logger,
                # stopIter=25
                )
 
        if save and (e+1) % 10 == 0:
            print("\nSave Epoch: {}".format(e))
            save_model(model, e+1, optimizer, saveDir)

        if e == 99:
            print("\nChanged learning rate to 0.005\n")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / 10

    # print("\nLAST TEST")
    # pgd_test(model, testLoader, criterion, pgdAdv)

    # if save:
    #     print("\nSave Epoch: {}".format(trainEpochEnd-1))
    #     save_model(model, trainEpochEnd-1, optimizer, saveDir)

    # plotter = Plot(saveDir, logger)
    # plotter.draw_figure_losses()
    # plotter.draw_figure_margins()