import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataModel:
    def __init__(self, dataSheet):
        self.fluxData = np.array(dataSheet["solidsFlux"]) / 100
        self.concentrationData = np.array(dataSheet["concentration"])
        self.max = np.max(self.fluxData)

    def plot(self):
        plt.plot(self.concentrationData,self.fluxData)
        plt.show()

    def interpolate(self, concentration):
        return np.interp(concentration,self.concentrationData,self.fluxData)
