import numpy as np
from numpy import exp, pi, sin, cos
from math import sqrt
from scipy.integrate import nquad
from .integrators import monte_carlo
import matplotlib.pyplot as plt

class Basis(object):

    def __init__(self, xlim, coef, k_list):
        self.k_list = k_list # list of coefficients
        self.xlim = xlim # simulation limits
        self.dl = [i[1]-i[0] for i in self.xlim]
        self.hk = np.zeros(coef+1)
        # print("Initializing the basis function parameters")
        for i in range(coef[0]+1):
            for j in range(coef[1]+1):
                self.hk[i,j] = sqrt(nquad(lambda x, y: (self._fk([i,j], [x,y]))**2, xlim)[0])
        # print("Finished basis function initialization")
    def _fk(self, k, x):
        return cos(pi * k[0] * x[0] / self.dl[0]) * cos(pi * k[1] * x[1] / self.dl[1])# * cos(pi * k[2] * x[2] / self.dl[2])

    def __call__(self, k, x):
        return self._fk(k, x)/self.hk[k[0],k[1]]

    def dfk(self, k, x):
        dfk_temp = np.zeros(x.shape)
        hk = self.hk[k[0], k[1]]
        dfk_temp[0] = -k[0]*pi*sin(pi*k[0]*x[0]/self.dl[0])*cos(pi*k[1]*x[1]/self.dl[1])/hk
        dfk_temp[1] = -k[1]*pi*sin(pi*k[1]*x[1]/self.dl[1])*cos(pi*k[0]*x[0]/self.dl[0])/hk
        # dfk_temp[2] = -k[2]*pi*sin(pi*k[2]*x[2]/self.dl[2])*cos(pi*k[0]*x[0]/self.dl[0])*cos(pi*k[1]*x[1]/self.dl[1])/hk
        return dfk_temp


class TrajectoryDistribution(object):
    '''
    Basically a class that calculate q(x(t)) for a trajectory
    '''
    def __init__(self, coefs, k_list, memory, x_lim):

        self.__coefs = coefs
        self.__coef_list = k_list
        self.__memory = memory
        self.__state_memory = [] # storage for the memory
        self.basis = Basis(x_lim, coefs, k_list)

    def get_ck_from_trajectory(self, trajectory):
        ''' Get the trajectory Fourier coefficients '''
        ck = []
        for i,k in enumerate(self.__coef_list):
            _fk = [self.basis(k, xi) for xi in self.__state_memory + trajectory]
            ck.append(np.sum(_fk))
        ck =  np.array(ck)/len(self.__state_memory + trajectory)
        return ck
    def remember(self, state):
        ''' Remember that state '''
        if len(self.__state_memory) >= self.__memory:
            del self.__state_memory[0]
            self.__state_memory.append(state.copy())
        else:
            self.__state_memory.append(state.copy())
    def get_remembered_time(self):
        return len(self.__state_memory)
    def __call__(self, sample, trajectory):
        ''' Get the actual distribution as a function of the sample '''
        print('Need to implement this properly as a reconstrution')
        # _q = 0.0
        # counter = 0
        # for state in (self.__state_memory + trajectory):
        #     _q += self.__g(sample, state) # add up the probability
        #
        # return _q


class TargetDistribution(object):
    '''
    Class to get the target distribution
    '''
    def __init__(self, coefs, x_lim, k_list):
        self.__coefs = coefs
        self.__coef_list = k_list
        self.__x_lim = x_lim
        self.basis = Basis(x_lim, coefs, k_list)
        self.center = lambda t: [# center of the gaussian
                    np.array([0.4,0.4]) ,
                    np.array([0.6,0.6]),
                    np.array([0.6,0.4])]
        self.t = 0.0
        vel = 0.4
        # self.center = lambda t: [
        #             0.25 * np.array([np.cos(vel*t+np.pi/2), np.sin(vel*t+np.pi/2)]) + 0.5,
        #             0.1 * np.array([np.cos(vel*t), np.sin(vel*t)]) + 0.5,
        #             0.25 * np.array([np.cos(vel*t+np.pi), np.sin(vel*t+np.pi)]) + 0.5,
        #             np.array([0.2*np.cos(vel*t-np.pi/2), 0.2*np.sin(vel*t-np.pi/2)]) + 0.5
        #
        #
        # ]
        self._center = self.center(self.t)
        self.weights = [ 100, 160, 200, 240]


        X,Y = np.meshgrid(np.linspace(0,1,20), np.linspace(0,1,20))
        self.__samples = np.c_[X.ravel(), Y.ravel()]

        sample_eval = [self.__phi(sample) for sample in self.__samples]
        self.sample_eval = np.array(sample_eval).reshape(X.shape)
        self.X_, self.Y_ = X, Y
        self.__normfact = np.sum(sample_eval)
        # self.__normfact = nquad(lambda x,y: self.__phi([x,y]), self.__x_lim)[0]
        self.phik = None
        self.phik = self.get_phik()

    def update_phik(self,time):
        self.phik = None
        self._center = self.center(time)
        self.phik = self.get_phik()
        return self.phik.copy()

    def get_phik(self):
        if self.phik is None:
            phik = []
            sample_eval = [self.__phi(sample) for sample in self.__samples]
            self.__normfact = np.sum(sample_eval)
            for k in self.__coef_list:
                temp = [sample_eval[i]*self.basis(k, sample)/self.__normfact for i,sample in enumerate(self.__samples)]
                phik.append(np.sum(temp))
                # temp_fun = lambda x,y: self.__call__([x,y]) * self.basis(k, [x,y])
                # phik.append(nquad(temp_fun, self.__x_lim)[0])
            phik = np.array(phik)
            # phik /= phik[0]
            return phik
            return phik
        else:
            return self.phik.copy()
    def __phi(self, sample):
        center = self._center
        # center = self.center
        # return 1.0
        res = 0.0
        for i,_center in enumerate(center):
            res = np.amax([res, exp(
                        -self.weights[i] * (sample - _center).dot(sample - _center)
                        )])
        return res
        # return 0*exp(
        #             -50 * (sample - center[0]).dot(sample - center[0])
        #         ) + exp(
        #             -100 * (sample - center[1]).dot(sample - center[1])
        #         ) + exp(
        #             -50 * (sample-center[2]).dot(sample - center[2]))


    def __call__(self, sample):
        return self.__phi(sample)/self.__normfact

    def plot(self):
        sample_eval = [self.__call__(sample) for sample in self.__samples]
        self.sample_eval = np.array(sample_eval).reshape(self.X_.shape)
        plt.contourf(self.X_, self.Y_, self.sample_eval)
