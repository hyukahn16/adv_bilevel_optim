import torch
import os
import matplotlib.pyplot as plt

def save_model(model, epoch, optimizer, save_dir):
    save_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': self.optimizer.state_dict(),
    }
    torch.save(save_state, save_dir + "/ckpt_{}.pt".format(epoch))
    print("Model saved at epoch {}".format(epoch))

def load_model(load_dir, model, epoch):
    load_dir = os.path.join(load_dir, "ckpt_{}.pt".format(epoch))
    checkpoint = torch.load(load_dir)
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded model from {}".format(load_dir))
    return epoch