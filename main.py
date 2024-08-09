import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
# import cProfile

from models import *
from defender import beta_adv_train
from test import pgd_test
from pgd import PGD
from util import save_model, load_model, get_trainloader, get_testloader
from logger import Logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=8/255, 
                        help="Epsilon value for perturbation size")
    parser.add_argument("--atkIter", type=int, default=10, 
                        help="Number of attack iterations for training")
    parser.add_argument("--trainBatchSize", type=int, default=500)
    parser.add_argument("--testBatchSize", type=int, default=500)
    parser.add_argument("--testEnabled", action="store_true", default=True,
                        help="Run tests between training")

    parser.add_argument("--useBETA", action="store_true", default=True,
                        help="Train model with PGD")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--numTrainEpoch", type=int, default=50,
                        help="Number of epochs to train")
    
    parser.add_argument("--saveEnabled", action="store_true",
                        help="Save model during training")
    parser.add_argument("--saveDir", type=str, default="", 
                        help="Base directory for where the saved models are saved. \
                            can NOT be existing directory")    
    
    parser.add_argument("--loadEnabled", action="store_true",
                        help="Load a saved model")
    parser.add_argument("--loadEpoch", type=int, default=0,
                        help="Epoch to load saved model. Used only when load set True.")
    parser.add_argument("--loadDir", type=str, default="",
                        help="Directory to load model from")

    # Only used for logger
    parser.add_argument("--trainEpochStart", type=int, default=0,
                        help="Starting train epoch (Only used for saving to log)")
    args = parser.parse_args()

    if args.saveEnabled and not args.saveDir:
        exit("Save was enabled but save directory was not provided.")
    if args.loadEnabled:
        if not args.loadDir or not args.loadEpoch:
            exit("load was enabled but load directory was not provided.")
    print(args)

    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainLoader = get_trainloader(trainBatch=args.trainBatchSize)
    testLoader = get_testloader(testBatch=args.testBatchSize, shuffle=False)
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    pgdAdv = PGD(model)

    if args.loadEnabled:
        args.loadDir = os.path.join("saved_models", args.loadDir)
        load_model(args.loadDir, args.loadEpoch, model, optimizer)
        # Update train epochs
        args.trainEpochStart = args.loadEpoch
    if args.saveEnabled:
        args.saveDir = os.path.join("saved_models", args.saveDir)
        if os.path.isdir(args.saveDir):
            # exit("saveDir already exists! Exiting...")
            print("Using existing save directory")
        else:
            os.mkdir(args.saveDir)
            print("Made model save directory at " + args.saveDir)
        logger = Logger(args.saveDir)
        logger.write_args(args)
    else:
        logger = Logger(None)

    # TRAIN STARTS HERE
    trainEpochEnd = args.trainEpochStart + args.numTrainEpoch
    print("Training Epoch {}-{}".format(args.trainEpochStart, trainEpochEnd))
    for e in range(args.trainEpochStart, trainEpochEnd):
        print("\nTrain Epoch: {}".format(e))
        beta_adv_train(
            trainLoader, args.eps, model, device,
            args.atkIter, criterion, optimizer, logger,
            args.useBETA
            )
        
        if args.testEnabled and e % 1 == 0:
            print("\nTest Epoch: {}".format(e))
            pgd_test(
                model, testLoader,
                criterion, pgdAdv, logger,
                )
 
        if args.saveEnabled and (e+1) % 10 == 0:
            print("\nSave Epoch: {}".format(e))
            save_model(model, e+1, optimizer, args.saveDir)

        # if e == 100:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.lr / 10
        #     log = f"Epoch {e}: Changed learning rate to {args.lr/10}\n"
        #     print(log)
        #     logger.write_args_single(log)

    # if save:
    #     print("\nSave Epoch: {}".format(trainEpochEnd-1))
    #     save_model(model, trainEpochEnd-1, optimizer, saveDir)

    # os.system("shutdown /s /t0")