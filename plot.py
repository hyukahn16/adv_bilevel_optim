import matplotlib.pyplot as plt
import os

class Plot:
    def __init__(self, saveDir):
        self.modelLosses = []
        self.pertMargins = []
        self.saveDir = saveDir

        if not os.path.isdir(self.saveDir):
            os.mkdir(self.saveDir)
        
    def draw_figure_losses(self):
        plt.plot(self.modelLosses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plotName = 'Losses.png'
        pltDir = os.path.join(self.saveDir, plotName)
        plt.savefig(pltDir)
        # plt.show()

    def draw_figure_margins(self):
        plt.plot(self.pertMargins)
        plt.xlabel('Epoch')
        plt.ylabel('Perturbation Average Margins')

        plotName = 'Perturbation Average Margins.png'
        pltDir = os.path.join(self.saveDir, plotName)
        plt.savefig(pltDir)
        # plt.show()