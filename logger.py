import os

class Logger:
    def __init__(self, saveDir):
        self.saveDir = saveDir

        # TRAIN logs
        self.trainLoss = []
        self.trainLoss = os.path.join(
            self.saveDir, "train_loss.txt")
        open(self.trainLoss, 'w').close()

        self.trainMargin = []
        self.trainMargin = os.path.join(
            self.saveDir, "train_margin.txt")
        open(self.trainMargin, 'w').close()

        self.trainAcc = []
        self.trainAcc = os.path.join(
            self.saveDir, "train_acc.txt")
        open(self.trainAcc, 'w').close()

        # TEST logs
        self.testBenignAcc = []
        self.testBenignAcc = os.path.join(
            self.saveDir, "test_benign_acc.txt")
        open(self.testBenignAcc, 'w').close()

        self.testRobustAcc = []
        self.testRobustAcc = os.path.join(
            self.saveDir, "test_robust_acc.txt")
        open(self.testRobustAcc, 'w').close()

    def save_train_loss(self, loss):
        self.trainLoss.append(loss)
        with open(self.trainLoss, 'a') as f:
            f.write("{}\n".format(loss))

    def save_train_margin(self, margin):
        self.trainMargin.append(margin)
        with open(self.trainMargin, 'a') as f:
            f.write("{}\n".format(margin))

    def save_train_acc(self, acc):
        self.trainAcc.append(acc)
        with open(self.trainAcc, 'a') as f:
            f.write("{}\n".format(acc))

    def save_test_benign_acc(self, acc):
        self.testBenignAcc.append(acc)
        with open(self.testBenignAcc, 'a') as f:
            f.write("{}\n".format(acc))
   
    def save_test_robust_acc(self, acc):
        self.testRobustAcc.append(acc)
        with open(self.testRobustAcc, 'a') as f:
            f.write("{}\n".format(acc))