import numpy as np
from numpy import exp, sqrt, pi, cos, sin
import matplotlib.pyplot as plt
import time
from integrators import monte_carlo
from scipy.integrate import nquad

class Phik(object):

    def __init__(self, settings):
        self.basis = settings.basis
        self.coef = settings.coef
        self.xlim = settings.xlim
        self.k_list = settings.k_list

        phi_temp = lambda x,y: 1.0
        # normfact = monte_carlo(lambda x,y: phi_temp(x,y), self.xlim, n=200)
        self.normfact = nquad(lambda x,y: phi_temp(x,y), self.xlim)[0]
        self.phi = lambda x,y: phi_temp(x,y)/self.normfact
        self.phik = self.get_phik(self.phi)

    def get_phik(self, phi):
        phik = np.zeros(self.coef+1).ravel()
        for i,k in enumerate(self.k_list):
            temp_fun = lambda x,y: phi(x,y) * self.basis.fk(k, [x,y])
            phik[i],_ = monte_carlo(temp_fun, self.xlim, n=200)
            # phik[i] = nquad(temp_fun, self.xlim)[0]
        phik /= phik[0]
        return phik

    def update_eid(self):
        self.phik = self.get_phik(self.phi)
