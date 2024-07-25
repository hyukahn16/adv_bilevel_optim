import torch
import torch.optim as optim

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
    # print("Max Logits: {}".format(atk_cls_pred))
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
    # print("\nY: {}".format(y))
    numClasses = 10
    batchSize = x.shape[0]

    maxMargins = torch.full((batchSize,), float('-inf')).to(device)
    maxPerts = torch.zeros(x.shape).to(device)
    maxClasses = torch.full((batchSize,), int('-1')).to(device)

    with torch.no_grad():
        for j in range(numClasses):
            perts = x + torch.zeros_like(x).uniform_(-eps, eps)
            perts = torch.clamp(perts, 0, 1)

            logits = model(perts)
            margins = negative_margin(logits, j, y)
            # print("Margin @ {}: {}".format(j, margins))

            comp = torch.logical_and(torch.gt(margins, maxMargins), torch.ne(y, j))
            maxMargins[comp] = margins[comp]
            maxPerts[comp] = perts[comp]
            maxClasses[comp] = j
    # print("---")
    # print("Max Margins: {}".format(maxMargins))
    # print("Max Classes: {}".format(maxClasses))
    # logits = model(maxPerts)
    # margins = negative_margin_multi(logits, maxClasses, y)
    # print("Logits:\n{}".format(logits))
    # print("Margins: {}".format(margins))
    pertOptim = optim.RMSprop(
        [maxPerts],
        maximize=True,
        lr=0.005 # Default 0.01
        )
    for t in range(atk_iter):
        maxPerts.requires_grad_()
        pertOptim.zero_grad()

        logits = model(maxPerts)
        margins = negative_margin_multi(logits, maxClasses, y)
        avgMargin = torch.mean(margins)
        avgMargin.backward()
        pertOptim.step()

        with torch.no_grad():
            torch.clamp(maxPerts, x-eps, x+eps, out=maxPerts)
            torch.clamp(maxPerts, 0, 1, out=maxPerts)
            
    return (maxPerts, margins)