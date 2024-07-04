import torch
from tqdm import tqdm

from attacker import best_targeted_attack

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
    iterator = iter(train_loader)
    for t in tqdm(range(train_iter)):
        # print("Training Iteration {}".format(t))
        optimizer.zero_grad()
        data, target = next(iterator)
        data, target = data.to(device), target.to(device)

        perturbs, margins = best_targeted_attack(
            data, target, eps, model, atk_iter, device)
        valid = torch.gt(margins, 0)
        data, perturbs, target = data[valid], perturbs[valid], target[valid]
        if torch.numel(target) == 0:
            continue
        pertInput = torch.add(data, perturbs)
        pertInput = torch.clamp(pertInput, min=0, max=1)

        logits = model(pertInput)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad)

