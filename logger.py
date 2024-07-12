import os

class Logger:
    def __init__(self, saveDir):
        self.saveDir = saveDir

        # TRAIN logs
        self.train_loss = os.path.join(
            self.saveDir, "train_loss.txt")
        open(self.train_loss, 'w').close()

        self.train_margin = os.path.join(
            self.saveDir, "train_margin.txt")
        open(self.train_margin, 'w').close()

        self.train_acc = os.path.join(
            self.saveDir, "train_acc.txt")
        open(self.train_acc, 'w').close()

        # TEST logs
        self.test_benign_acc = os.path.join(
            self.saveDir, "test_benign_acc.txt")
        open(self.test_benign_acc, 'w').close()

        self.test_robust_acc = os.path.join(
            self.saveDir, "test_robust_acc.txt")
        open(self.test_robust_acc, 'w').close()

    def save_train_loss(self, loss):
        with open(self.train_loss, 'a') as f:
            f.write("{}\n".format(loss))

    def save_train_margin(self, margin):
        with open(self.train_margin, 'a') as f:
            f.write("{}\n".format(margin))

    def save_train_acc(self, acc):
        with open(self.train_acc, 'a') as f:
            f.write("{}\n".format(acc))

    def save_test_benign_acc(self, acc):
        with open(self.test_benign_acc, 'a') as f:
            f.write("{}\n".format(acc))
   
    def save_test_robust_acc(self, acc):
        with open(self.test_robust_acc, 'a') as f:
            f.write("{}\n".format(acc))