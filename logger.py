import os

class Logger:
    def __init__(self, saveDir):
        self.saveDir = saveDir
        if not saveDir:
            return

        argFileName = "train_args.txt"
        self.trainArgsFile = os.path.join(self.saveDir, argFileName)
        if not os.path.exists(self.trainArgsFile):
            open(self.trainArgsFile, 'w').close()

        # TRAIN logs
        self.trainLoss = os.path.join(
            self.saveDir, "train_loss.txt")
        if not os.path.exists(self.trainLoss):
            open(self.trainLoss, 'w').close()

        self.trainMargin = os.path.join(
            self.saveDir, "train_margin.txt")
        if not os.path.exists(self.trainMargin):
            open(self.trainMargin, 'w').close()

        self.trainAcc = os.path.join(
            self.saveDir, "train_acc.txt")
        if not os.path.exists(self.trainAcc):
            open(self.trainAcc, 'w').close()

        # TEST logs
        self.testBenignAcc = os.path.join(
            self.saveDir, "test_benign_acc.txt")
        if not os.path.exists(self.testBenignAcc):
            open(self.testBenignAcc, 'w').close()

        self.testRobustAcc = os.path.join(
            self.saveDir, "test_robust_acc.txt")
        if not os.path.exists(self.testRobustAcc):
            open(self.testRobustAcc, 'w').close()


    def save_train_loss(self, loss):
        if not self.saveDir:
            return
        
        with open(self.trainLoss, 'a') as f:
            f.write("{}\n".format(loss))

    def save_train_margin(self, margin):
        if not self.saveDir:
            return
        
        with open(self.trainMargin, 'a') as f:
            f.write("{}\n".format(margin))

    def save_train_acc(self, acc):
        if not self.saveDir:
            return
        
        with open(self.trainAcc, 'a') as f:
            f.write("{}\n".format(acc))

    def save_test_benign_acc(self, acc):
        if not self.saveDir:
            return
        
        with open(self.testBenignAcc, 'a') as f:
            f.write("{}\n".format(acc))
   
    def save_test_robust_acc(self, acc):
        if not self.saveDir:
            return
        
        with open(self.testRobustAcc, 'a') as f:
            f.write("{}\n".format(acc))

    def write_args(self, args):
        if not self.saveDir:
            return

        # Convert the arguments to a dictionary
        args_dict = vars(args)

        # Save the arguments to a text file
        with open(self.trainArgsFile, "a") as file:
            file.write("\n\n")
            for key, value in args_dict.items():
                file.write(f"{key}: {value}\n")

        print("Arguments saved to args.txt")

    def write_args_single(self, arg):
        if not self.saveDir:
            return
        with open(self.trainArgsFile, "a") as file:
            file.write(f"{arg}\n")