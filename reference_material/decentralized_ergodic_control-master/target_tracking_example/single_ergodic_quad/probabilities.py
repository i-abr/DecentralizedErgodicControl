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

from .partical_filter import ParticleFilter
from .target import Target
class TargetDistribution(object):
    '''
    Class to get the target distribution
    '''
    def __init__(self, coefs, x_lim, k_list):
        self.__coefs = coefs
        self.__coef_list = k_list
        self.__x_lim = x_lim
        self.basis = Basis(x_lim, coefs, k_list)

        self.particle_filter = ParticleFilter(num_particles=100)

        self.targets = [Target()]

        self.phik = None
        self.phik = self.get_phik()
        self.particle_dump = []
        self.particle_dump.append(self.particle_filter.particles.copy())
    def update_phik(self, robot_state, time):
        self.phik = None
        yk = []
        for target in self.targets:
            target_state = target.update_position(time)
            yk.append(self.particle_filter.sensor_model.h(target_state, robot_state))
        self.particle_filter.update(robot_state, yk)
        self.phik = self.get_phik()
        self.particle_dump.append(self.particle_filter.particles.copy())
        return self.phik.copy()

    def get_phik(self):
        if self.phik is None:
            phik = []
            sample_eval = self.particle_filter.weights
            samples = self.particle_filter.particles
            for k in self.__coef_list:
                temp = [sample_eval[i]*self.basis(k, sample) for i,sample in enumerate(samples)]
                phik.append(np.sum(temp))
                # temp_fun = lambda x,y: self.__call__([x,y]) * self.basis(k, [x,y])
                # phik.append(nquad(temp_fun, self.__x_lim)[0])

            phik = np.array(phik)
            # print('ParticleFilter phik : {}'.format(phik))
            # phik /= phik[0]
            return phik
        else:
            return self.phik.copy()

    def save(self, filePath):
        np.save(filePath, self.particle_dump)

    def plot(self, ax=None):
        circle = plt.Circle(self.targets[0].state, 0.02, color='r')
        if ax is None:
            ax = plt.gca()
        ax.add_artist(circle)
        if ax is None:
            plt.scatter(self.particle_filter.particles[:,0],
                        self.particle_filter.particles[:,1],
                        c=self.particle_filter.weights)
        else:
            ax.scatter(self.particle_filter.particles[:,0],
                        self.particle_filter.particles[:,1],
                        c=self.particle_filter.weights)
