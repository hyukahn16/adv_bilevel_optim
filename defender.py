from tqdm import tqdm

from attacker import best_targeted_attack
from pgd import PGD

def beta_adv_train(
        train_loader, eps, model, device, atk_iter, 
        criterion, optimizer, logger, trainPGD):
    """Gets and prints the spreadsheet's header columns

    Parameters
    ----------
    train_loader : DataLoader
        Contains images and labels
    eps : float
        max perturbation value
    model : PyTorch Model
        Model to train
    train_iter : int
        Number of times to train model
    atk_iter : int
        Number of times to perturb image for attack in BETA
    """
    model.train()

    if trainPGD:
        pgdAdv = PGD(model)

    epochLoss = 0
    epochMargin = 0
    epochCorrect = 0
    epochTotalData = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        if not trainPGD:
            perturbs, avgMargin = best_targeted_attack(
                data, target, eps, model, atk_iter, device)
        else: # Uses PGD
            perturbs = pgdAdv.perturb(data, target, atk_iter)
        
        # FIXME: check for margin>0?

        optimizer.zero_grad()
        logits = model(perturbs)
        loss = criterion(logits, target)

        epochLoss += loss.item()
        if not trainPGD:
            epochMargin += avgMargin.item()
        _, predIndices = logits.max(dim=1)
        epochCorrect += predIndices.eq(target).sum().item()
        epochTotalData += target.size(0)

        loss.backward()
        optimizer.step()

    epochAcc = 100 * epochCorrect / epochTotalData
    print("Epoch total loss:     {}".format(epochLoss))
    print("Epoch accuracy:       {}".format(epochAcc))
    if trainPGD:
        print("Epoch average margin: {}".format(epochMargin/batch_idx))

    logger.save_train_loss(epochLoss)
    logger.save_train_acc(epochAcc)
    if not trainPGD:
        logger.save_train_margin(epochMargin/batch_idx)