import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def negative_margin(logits, j, y):
    """Forward feeds perturbed input and calculates margin value.
    Positive value corresponds to a misclassification.

    Parameters
    ----------
    perturbed_x : logits
        logits from forward feed model from perturbed input
    j : int
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

def negative_margin_multi(logits, j, y):
    """
    j: list(int)
        Attacking class
    """
    atk_cls_pred = torch.gather(
        logits, 1, torch.unsqueeze(j, 1)).squeeze()
    corr_cls_pred = torch.gather(
        logits, 1, torch.unsqueeze(y, 1)).squeeze()
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

    maxMargins = torch.full((batchSize,), float('-inf')).to(device)
    maxPerts = torch.zeros(x.shape).to(device)

    for j in range(numClasses):
        pertBatch = x + torch.zeros_like(x).uniform_(-eps, eps)
        pertBatch = torch.clamp(pertBatch, 0, 1)
        pertBatch = pertBatch.to(device)
    
        pertOptim = optim.RMSprop(
            [pertBatch],
            maximize=True,
            lr=0.01 # Default 0.01  
            )
        
        for t in range(atk_iter):
            pertBatch.requires_grad_()
            pertOptim.zero_grad()

            logits = model(pertBatch)
            margins = negative_margin(logits, j, y)
            avgMargin = torch.mean(margins)
            avgMargin.backward()
            pertOptim.step()

            with torch.no_grad():
                torch.clamp(pertBatch, x-eps, x+eps, out=pertBatch)
                torch.clamp(pertBatch, 0, 1, out=pertBatch)
            
                # if t == 0:
                #     tempM = avgMargin
                #     temp = pertBatch.data.clone().detach()
                #     tAvg = torch.mean(pertBatch)
                # if t == atk_iter-1:
                #     print("Class {}".format(j))
                #     print("Margin Diff: {}".format((avgMargin - tempM).data))
                #     print("Perturbation Equal: {}".format(torch.equal(pertBatch, temp)))
                #     print("Perturbation Avg Diff: {}".format(torch.mean(pertBatch) - tAvg))
                #     print()

        # algorithm line 8: find negative-margin maximizing perturbations
        with torch.no_grad():
            logits = model(pertBatch)
            margins = negative_margin(logits, j, y)

            comp = torch.gt(margins, maxMargins)
            maxMargins[comp] = margins[comp]
            maxPerts[comp] = pertBatch[comp]

    return (maxPerts, maxMargins)