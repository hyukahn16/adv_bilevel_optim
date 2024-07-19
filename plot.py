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
# saveDir = "saved_models"
# saveModel = "pgd"
# saveDir = os.path.join(saveDir, saveModel)

# file = os.path.join(saveDir, "train_acc.txt")
# plotter = Plot(saveDir)
# plotter.draw_figure_losses(file)

