import torch
from tqdm import tqdm

from attacker import best_targeted_attack
from pgd import PGD

def beta_adv_train(
        model, trainLoader, logger, 
        criterion, optimizer, device,
        args):
    """Gets and prints the spreadsheet's header columns

    Parameters
    ----------
    trainLoader : DataLoader
        Contains images and labels
    eps : float
        max perturbation value
    model : PyTorch Model
        Model to train
    train_iter : int
        Number of times to train model
    atkIter : int
        Number of times to perturb image for attack in BETA
    """
    model.train()
    if not args.useBETA:
        pgdAdv = PGD(model)
    batchSize = trainLoader.batch_size
    epochLoss = 0
    epochMargin = 0
    epochCorrectData = 0
    epochExcludedData = 0
    epochTotalData = 0
    for batchIdx, (data, target) in enumerate(tqdm(trainLoader)):
        # data = data.cuda(pin_memory=True, non_blocking=True)
        # target = target.cuda(pin_memory=True, non_blocking=True)
        data = data.to(device)
        target = target.to(device)

        if args.useBETA:
            perturbs, margins = best_targeted_attack(
                model, device, data, target,
                args.eps, args.atkIter, args.betaLr)

            validMargins = torch.gt(margins, 0.0)
            if True not in validMargins:
                continue
            perturbs = perturbs[validMargins]
            target = target[validMargins]
        else: # Uses PGD
            perturbs = pgdAdv.perturb(data, target, args.atkIter)

        optimizer.zero_grad()
        logits = model(perturbs)
        loss = criterion(logits, target)

        epochLoss += loss.detach().item()
        _, predIndices = logits.detach().max(dim=1)
        epochTotalData += batchSize
        epochCorrectData += predIndices.eq(target).sum().item()
        if args.useBETA:
            epochExcludedData += batchSize - target.size(0)
            epochMargin += torch.mean(margins)

        loss.backward()
        optimizer.step()

    if args.useBETA:
        epochAcc = 100 * (epochExcludedData + epochCorrectData) / epochTotalData
        epochUsedData = epochTotalData - epochExcludedData
        print("Excluded Data:  {}".format(epochExcludedData))
        print("Used Data:      {}".format(epochUsedData))
        print("Correct Data:   {}".format(epochCorrectData))
        print("Average Margin: {}".format(epochMargin/batchIdx))
        logger.save_train_margin(epochMargin/batchIdx)
    else:
        epochAcc = 100 * epochCorrectData / epochTotalData

    print("Accuracy:       {}".format(epochAcc))
    print("Total Loss:     {}".format(epochLoss))
    logger.save_train_loss(epochLoss)
    logger.save_train_acc(epochAcc)