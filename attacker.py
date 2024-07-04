import torch
import torch.optim as optim
import numpy as np
from operator import itemgetter

def negative_margin(logits, j, y):
    """Forward feeds perturbed input and calculates margin value.
    Positive value corresponds to a misclassification.

    Parameters
    ----------
    perturbed_x : logits
        logits from forward feed model from perturbed input
    j : list(int)
        Attacking class
    y : list(int)
        Correct class

    Returns
    -------
    List of int
        margin value of each data point
        (positive value corresponds to misclassification)
    """
    atk_cls_pred = logits[:, j]
    corr_cls_pred = torch.gather(logits, 1, torch.unsqueeze(y, 1)).squeeze()
    return atk_cls_pred - corr_cls_pred

def best_targeted_attack(x, y, eps, model, atk_iter, device):
    """Calculates best-class perturbation for batch of images

    Parameters
    ----------
    x : list(PILImage)?
        Batch of original images for perturbation
        Image values are in [0,1]
    y : list(int)
        Batch of correct classes for x
    eps : float
        Max perturbation value
    model : PyTorch Model
    atk_iter : int
        Number of times to perturb image

    Returns
    -------
    List of perturbations (NOT the perturbed image)
    """
    numClasses = 10
    batchSize = x.shape[0]
    pertClasses = []
    for j in range(numClasses):
        # algorithm line 3:
        # Initialize perturbations from uniform distribution
        # FIXME: x + torch.zeros_like(...)??? Or stay without it
        # FIXME:: L2Norm(pert) <= eps
        pertBatch = torch.zeros_like(x).uniform_(-eps, eps)
        pertBatch = pertBatch.to(device)
        # pertBatch.requires_grad_(True)
        pertClasses.append(pertBatch)
    
        # algorithm line 6, 7
        optimizer = optim.RMSprop(
            [pertBatch],
            maximize=True,
            lr=1 # Default 0.01
            )
        for t in range(atk_iter):
            pertBatch.requires_grad_()
            optimizer.zero_grad()

            pertInput = torch.add(x, pertBatch)
            pertInput = torch.clamp(pertInput, 0, 1)
            logits = model(pertInput)

            margins = negative_margin(logits, j, y)
            avg_margin = torch.mean(margins)
            avg_margin.backward()
            print(torch.mean(pertBatch.grad))
            optimizer.step()

            pertBatch = pertBatch.detach()
            pertBatch = torch.clamp(pertBatch, -eps, eps)

            # if t == 0:
            #     temp = pertBatch[0].detach().clone()
            #     print(t, avg_margin)
            # if t == atk_iter-1:
            #     print(t, avg_margin)
            #     print(torch.equal(pertBatch[0], temp))
        
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # algorithm line 8: find negative-margin maximizing perturbations
    with torch.no_grad():
        maxMargins = torch.full((batchSize,), float('-inf')).to(device)
        maxMarginCls = torch.zeros(batchSize, dtype=torch.int64).to(device)
        for j in range(numClasses):
            pertInput = torch.add(x, pertClasses[j])
            pertInput = torch.clamp(pertInput, 0, 1)
            logits = model(pertInput)
            margins = negative_margin(logits, j, y)

            comp = torch.gt(margins, maxMargins)
            maxMargins[comp] = margins[comp]
            maxMarginCls[comp] = j

        maxPerts = [pertClasses[maxMarginCls[i]][i] for i in range(batchSize)]
        maxPerts = torch.stack(maxPerts)
        return (maxPerts, maxMargins)