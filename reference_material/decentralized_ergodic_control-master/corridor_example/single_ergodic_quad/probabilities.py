import numpy as np
from numpy import exp, pi, sin, cos
from math import sqrt
from scipy.integrate import nquad
from .integrators import monte_carlo

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
        self.center = [# center of the gaussian
                    np.array([0.3,0.3]) ,
                    np.array([0.7,0.7]),
                    np.array([0.7,0.3])]
        X,Y = np.meshgrid(self.__x_lim[0], self.__x_lim[1])
        self.__samples = np.c_[X.ravel(), Y.ravel()]
        sample_eval = [self.__phi(sample) for sample in self.__samples]
        # self.__normfact = np.sum(sample_eval)
        self.__normfact = nquad(lambda x,y: self.__phi([x,y]), self.__x_lim)[0]
        self.phik = None
        self.phik = self.get_phik()
        self.t = 0.0
        # self.center = lambda t: [
        #             0.3 *np.cos(0*t) * np.array([np.cos(0.*t), np.sin(0.*t)]),
        #             -0.3 *np.cos(0*t)* np.array([np.cos(0.*t), np.sin(0.*t)])
        # ]
        # self._center = self.center(self.t)
    def update_time(self,time):
        # self.t = time
        # self._center = self.center(self.t)
        pass

    def get_phik(self):
        if self.phik is None:
            phik = []
            for k in self.__coef_list:
                # temp = [self.__call__(sample)*self.basis(k, sample) for sample in self.__samples]
                # phik.append(np.sum(temp))
                temp_fun = lambda x,y: self.__call__([x,y]) * self.basis(k, [x,y])
                phik.append(nquad(temp_fun, self.__x_lim)[0])
            phik = np.array(phik)
            # phik /= phik[0]
            return phik
        else:
            return self.phik.copy()
    def __phi(self, sample):
        # center = self.center(self.t)
        # center = self.center
        return 1.0
        # return exp(
        #             -50 * (sample - center[0]).dot(sample - center[0])
        #         ) + exp(
        #             -50 * (sample - center[1]).dot(sample - center[1])
        #         ) + exp( -50 * (sample-self.center[2]).dot(sample - self.center[2]))

    def __call__(self, sample):
        return self.__phi(sample)/self.__normfact
