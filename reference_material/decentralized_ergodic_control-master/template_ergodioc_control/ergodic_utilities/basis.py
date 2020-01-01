import numpy as np
from numpy import pi, sin, cos
from math import sqrt
from scipy.integrate import nquad

class Basis(object):

    def __init__(self, xlim, coef):

        self.xlim = xlim
        self.dl = [i[1]-i[0] for i in self.xlim]

        self.hk = np.zeros(coef+1)
        for i in range(coef[0]+1):
            for j in range(coef[1]+1):
                self.hk[i,j] = sqrt(nquad(lambda y, x: (self._fk([i,j], [x,y]))**2,xlim)[0])

    def _fk(self, k, x):
        return cos(pi * k[0] * x[0] / self.dl[0]) * cos(pi * k[1] * x[1] / self.dl[1])

    def fk(self, k, x):
        return self._fk(k, x)/self.hk[k[0],k[1]]

    def dfk(self, k, x):
        dfk_temp = np.zeros(x.shape)
        dfk_temp[0] = -k[0]*pi*sin(pi*k[0]*x[0]/self.dl[0])*cos(pi*k[1]*x[1]/self.dl[1])/self.hk[k[0],k[1]]
        dfk_temp[1] = -k[1]*pi*sin(pi*k[1]*x[1]/self.dl[1])*cos(pi*k[0]*x[0]/self.dl[0])/self.hk[k[0],k[1]]
        return dfk_temp
