import numpy as np
from numpy import exp
import matplotlib.pyplot as plt

class Terrain(object):

    def __init__(self):

        resample = False
        no_peaks = 20

        if resample:
            self.positions = np.array([
                [np.random.uniform(0.2,0.8), np.random.uniform(0.2,0.8)] for i in range(no_peaks)
            ])
            self.scales = [
                np.random.uniform(50, 100, (2,)) for i in range(no_peaks)
            ]
        else:
            self.positions = np.loadtxt('positions.csv')
            self.scales = np.loadtxt('scales.csv')
    def getElevationAtX(self, x):
        height = 1.0
        for i, position in enumerate(self.positions):
            dist = x[0:2] - position
            height += exp( - dist.dot(np.diag(self.scales[i]) ).dot(dist) )
        return height

    def plotElevation(self):
        X,Y = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
        grid = np.c_[X.ravel(), Y.ravel()]

        height = np.array(map(self.getElevationAtX, grid))

        plt.imshow(height.reshape(X.shape), extent=(0,1,0,1), origin='lower')
        return height

    def saveTerrain(self, filePath=''):
        np.savetxt('positions.csv', self.positions)
        np.savetxt('scales.csv', self.scales)


g_terrain = Terrain()
g_terrain.saveTerrain()


if __name__ == '__main__':
    terrain = Terrain()
    terrain.plotElevation()
