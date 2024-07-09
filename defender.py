import torch
from tqdm import tqdm

from attacker import best_targeted_attack
from pgd import PGD

def beta_adv_train(
        train_loader, eps, model, device, train_iter, atk_iter, 
        criterion, optimizer):
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

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        perturbs, margins = best_targeted_attack(
            data, target, eps, model, atk_iter, device)
        
        # valid = torch.gt(margins, 0)
        # temp = target.clone()[valid]
        # if torch.numel(temp) == 0:
        #     print("NO TARGET")
        #     continue
        
        optimizer.zero_grad()
        logits = model(perturbs)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()