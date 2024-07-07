import torch
import torch.optim as optim
import torch.nn.functional as F

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

    maxMargins = torch.full((batchSize,), float(0)).to(device)
    maxPerts = torch.zeros(x.shape).to(device)

    for j in range(numClasses):
        # algorithm line 3:
        pertBatch = torch.zeros_like(x).uniform_(-eps, eps) * eps
        pertBatch = pertBatch.to(device)
    
        # algorithm line 6, 7
        pertOptim = optim.RMSprop(
            [pertBatch],
            maximize=True,
            # lr=0.1 # Default 0.01
            lr = 2/255
            )
        
        for t in range(atk_iter):
            pertBatch.requires_grad_()
            pertOptim.zero_grad()

            pertInput = torch.add(x, pertBatch)
            pertInput = torch.clamp(pertInput, 0, 1)
            logits = model(pertInput)

            margins = negative_margin(logits, j, y)
            avgMargin = torch.mean(margins)
            avgMargin.backward() # This is the problem
            pertOptim.step()

            with torch.no_grad():
                F.normalize(pertBatch, p=2, dim=(1,2,3), out=pertBatch) * eps
            
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
            pertInput = torch.add(x, pertBatch)
            pertInput = torch.clamp(pertInput, 0, 1)
            logits = model(pertInput)
            margins = negative_margin(logits, j, y)

            comp = torch.gt(margins, maxMargins)
            maxMargins[comp] = margins[comp]
            maxPerts[comp] = pertBatch[comp]

    return (maxPerts, maxMargins)