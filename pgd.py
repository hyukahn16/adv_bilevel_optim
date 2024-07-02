import numpy as np
import os
import torch

from models import *

# pip install git+https://github.com/fra31/auto-attack
from autoattack import AutoAttack

class PGD(object):
    def __init__(
            self,
            device,
            model,
            rand_init=True,
            epsilon=8/255,
            alpha=2/255,
            testing=False):
        
        self.device = device
        self.testing = testing
        self.model = model

        # PGD hyperparameters
        self.rand_init = rand_init # atk noise starts random
        self.epsilon = epsilon # maximum distortion = 8/255
        self.alpha = alpha # attack step size = 2/255

    def perturb(self, x_natural, y, pgd_iter):
        x = x_natural.detach()
        # Random initialization
        if self.rand_init:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(pgd_iter):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(
                    torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon
                    )
            x = torch.clamp(x, 0, 1)
        return x