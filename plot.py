import matplotlib.pyplot as plt
import os

class Plot:
    def __init__(self, saveDir, logger):
        self.saveDir = saveDir
        self.logger = logger

        if not os.path.isdir(self.saveDir):
            os.mkdir(self.saveDir)
        
    def draw_figure_losses(self):
        plt.plot(self.logger.trainLoss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plotName = 'Losses.png'
        pltDir = os.path.join(self.saveDir, plotName)
        plt.savefig(pltDir)
        # plt.show()

    def draw_figure_margins(self):
        plt.plot(self.logger.trainMargin)
        plt.xlabel('Epoch')
        plt.ylabel('Perturbation Average Margins')

        plotName = 'Perturbation Average Margins.png'
        pltDir = os.path.join(self.saveDir, plotName)
        plt.savefig(pltDir)
        # plt.show()