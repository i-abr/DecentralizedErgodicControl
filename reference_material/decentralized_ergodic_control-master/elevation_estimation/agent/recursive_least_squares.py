import numpy as np
from numpy import exp

class RBF2D(object):
    """ Radial Basis Function class """
    def __init__(self, x,c,gamma):
        self.__x = x[0]
        self.__y = x[1]
        self.c = c
        self.gamma = gamma
    def phi(self, x):
        return self.c * exp(-self.gamma*(x[0]-self.__x)**2) * exp(-self.gamma*(x[1]-self.__y)**2)
    def dphidc(self, x):
        return exp(-self.gamma*(x[0]-self.__x)**2) * exp(-self.gamma*(x[1]-self.__y)**2)

class RecursiveLeastSquares(object):

    def __init__(self):
        self.vhist = []
        self.phihist = []
        N = 20
        self.__N = N
        X,Y = np.meshgrid(np.linspace(0.1,0.9,N), np.linspace(0.1,0.9,N))
        x_list = np.c_[X.ravel(), Y.ravel()]
        self.c = np.random.rand(len(x_list))-1
        self.cvar = np.ones(self.c.shape)
        self.__sig = np.ones(self.c.shape)*0.01**2
        self.phi = []
        self.alpha = 0.8
        self.Pk = np.diag([2]*(N*N))
        self.Bk = np.ones(N*N)
        for i,xi in enumerate(x_list):
            self.phi.append(RBF2D(xi, self.c[i], 10.0))


    def getCost(self, v, x, c):
        self.update_c(c)
        err = v - self.__call__(x)
        return err**2

    def updateEstimate(self, vk, xk):

        Jinit = self.getCost(vk, xk, self.c)
        cprev = self.c.copy()
        dphidk = np.array([phi.dphidc(xk) for phi in self.phi])
        lam = 0.99#1.0
        Sk = 1.0/(lam + dphidk.dot(self.Pk).dot(dphidk))
        an = vk - self.c.dot(dphidk)
        gn = self.Pk.dot(dphidk)*Sk
        self.Pk = (1.0/lam)*self.Pk - np.outer(gn,dphidk) * (1/lam) * self.Pk
        self.c = self.c + an * gn

        self.update_c(self.c)
        # Jnew = self.getCost(vk, xk, self.c)


    def update_c(self,c):
        for i,phi in enumerate(self.phi):
            phi.c = c[i]

    def __call__(self, x):
        v = 0.0
        for phi in self.phi:
            v += phi.phi(x)
        return v
