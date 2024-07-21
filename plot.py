import matplotlib.pyplot as plt
import os

class Plot:
    def __init__(self, saveDir, logger=None):
        self.saveDir = saveDir
        self.logger = logger

        if not os.path.isdir(self.saveDir):
            os.mkdir(self.saveDir)
        
    def draw_figure_losses(self, file=None):
        if file:
            plt.plot(self._get_from_file(file), label="Train Loss")
        else:
            plt.plot(self.logger.trainLossList, label="Train Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plotName = 'losses.png'
        pltDir = os.path.join(self.saveDir, plotName)
        plt.savefig(pltDir)

    def draw_figure_margins(self, file=None):
        if file:
            plt.plot(self._get_from_file(file), label="Train Margin")
        else:
            plt.plot(self.logger.trainMarginList, label="Train Margin")
        plt.xlabel('Epoch')
        plt.ylabel('Perturbation Average Margins')

        plotName = 'perturbation_average_margins.png'
        pltDir = os.path.join(self.saveDir, plotName)
        plt.savefig(pltDir)

    def _get_from_file(self, file):
        dataList = []
        with open(file, "r") as f:
            for line in f:
                dataList.append(float(line))
        return dataList

# Run this script to generate plots
saveDir = "saved_models"
saveModel = "pgd_merge"
saveDir = os.path.join(saveDir, saveModel)

train_acc_file = os.path.join(saveDir, "train_acc.txt")
test_benign_acc_file = os.path.join(saveDir, "test_benign_acc.txt")
test_robust_acc_file = os.path.join(saveDir, "test_robust_acc.txt")
plotter = Plot(saveDir)
trainAccList = plotter._get_from_file(train_acc_file)
testBenAccList = plotter._get_from_file(test_benign_acc_file)
testRobAccList = plotter._get_from_file(test_robust_acc_file)

trainAccLine, = plt.plot(trainAccList, label="Train PGD Accuracy")
testBenAccLine, = plt.plot(testBenAccList, label="Test Benign Accuracy")
testRobAccLine, = plt.plot(testRobAccList, label="Test Robust Accuracy")
plt.legend(handles=[trainAccLine, testBenAccLine, testRobAccLine])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plotName = "pgd_accuracy.png"
plotDir = os.path.join(saveDir, plotName)
plt.savefig(plotDir, dpi=1200)

# plotter.draw_figure_losses(file)

