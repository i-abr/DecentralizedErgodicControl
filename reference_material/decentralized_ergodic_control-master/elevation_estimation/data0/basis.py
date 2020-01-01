import numpy as np
from numpy import pi, sin, cos
from math import sqrt
from scipy.integrate import nquad

class Basis(object):

    def __init__(self, xlim, coef, k_list):
        self.k_list = k_list # list of coefficients
        self.xlim = xlim # simulation limits
        self.dl = [i[1]-i[0] for i in self.xlim]
        self.hk = np.zeros(coef+1)
        print("Initializing the basis function parameters")
        for i in range(coef[0]+1):
            for j in range(coef[1]+1):
                self.hk[i,j] = sqrt(nquad(lambda y, x: (self._fk([i,j], [x,y]))**2, xlim)[0])
        print("Finished basis function initialization")
    def _fk(self, k, x):
        return cos(pi * k[0] * x[0] / self.dl[0]) * cos(pi * k[1] * x[1] / self.dl[1])# * cos(pi * k[2] * x[2] / self.dl[2])

    def fk(self, k, x):
        return self._fk(k, x)/self.hk[k[0],k[1]]

    def dfk(self, k, x):
        dfk_temp = np.zeros(x.shape)
        hk = self.hk[k[0], k[1]]
        dfk_temp[0] = -k[0]*pi*sin(pi*k[0]*x[0]/self.dl[0])*cos(pi*k[1]*x[1]/self.dl[1])/hk
        dfk_temp[1] = -k[1]*pi*sin(pi*k[1]*x[1]/self.dl[1])*cos(pi*k[0]*x[0]/self.dl[0])/hk
        # dfk_temp[2] = -k[2]*pi*sin(pi*k[2]*x[2]/self.dl[2])*cos(pi*k[0]*x[0]/self.dl[0])*cos(pi*k[1]*x[1]/self.dl[1])/hk
        return dfk_temp

    def calcCoefList(self):
        # TODO: make a function that calculates the list of coefficients
        pass
