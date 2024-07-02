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
        # Get max value from each of RGB 
        # mx = torch.amax(x, dim=(1,2,3))
        # mx = torch.clamp(mx-eps, min=0)

        # mn = torch.amin(x, dim=(1,2,3))
        # mn = torch.clamp(mn+eps, max=1)

        # algorithm line 3:
        # Initialize perturbations from uniform distribution
        # FIXME: x + torch.zeros_like(...)??? Or stay without it
        # FIXME:: L2Norm(pert) <= eps
        pertBatch = torch.zeros_like(x).uniform_(-eps, eps)
        pertBatch = pertBatch.to(device)
        pertBatch.requires_grad_(True)
        pertClasses.append(pertBatch)
    
        # algorithm line 6, 7
        optimizer = optim.RMSprop(
            [pertBatch],
            maximize=True,
            lr=0.01 # Default 0.01
            )
        # diffCls = torch.ne(y, j)
        for t in range(atk_iter):
            optimizer.zero_grad()
            pertInput = torch.add(x, pertBatch)
            pertInput = torch.clamp(pertInput, min=0.0, max=1.0)
            logits = model(pertInput)

            margins = negative_margin(logits, j, y)
            avg_margin = torch.mean(margins)
            avg_margin.backward()
            optimizer.step()

            pertBatch = pertBatch.detach()
            pertBatch = torch.clamp(pertBatch, min=-eps, max=eps)

            # if t == 0:
                # temp = pertBatch[0].detach().clone()
            #     print(t, avg_margin)
            # if t == atk_iter-1:
            #     print(t, avg_margin)
                # print(torch.equal(pertBatch[0], temp))

    # algorithm line 8: find negative-margin maximizing perturbations
    maxMargins = torch.full((batchSize,), float('-inf')).to(device)
    maxMarginCls = torch.zeros(batchSize, dtype=torch.int64).to(device)
    for j in range(numClasses):
        pertInput = torch.add(x, pertClasses[j])
        pertInput = torch.clamp(pertInput, min=0, max=1)
        logits = model(pertInput)
        margins = negative_margin(logits, j, y)

        comp = torch.gt(margins, maxMargins)
        maxMargins[comp] = margins[comp]
        maxMarginCls[comp] = j

    maxPerts = [pertClasses[maxMarginCls[i]][i] for i in range(batchSize)]
    maxPerts = torch.stack(maxPerts)
    return (maxPerts, maxMargins)