import numpy as np
from numpy.random import uniform, randint
import matplotlib.pyplot as plt
from filterpy.monte_carlo import systematic_resample, multinomial_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
from numpy import arctan, pi, arctan2
from numpy.random import random
from numpy.random import seed

def wrap2Pi(x):
    ''' helper function '''
    x = np.mod(x+pi, 2*pi)
    if x < 0:
        x += 2*pi
    return x-pi


def pdf(mean,sigma, _eval):
    k = len(mean)
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    Z = 1.0/np.sqrt(det_sigma * (2.0 * pi)**k)
    delta = mean - _eval
    return np.exp(-0.5 * (delta).dot(inv_sigma).dot(delta))

class BearingOnlySensor(object):

    def __init__(self):

        self.nM = 2
        self.R = np.diag([0.01] * self.nM)
        self.Rinv = np.linalg.inv(self.R)
        self.range = 2.0
    def h(self, target_state, agent_state, sim=False):
        xt, xs = target_state, agent_state # make it easier to write up
        d = np.sqrt( (xs[0] - xt[0])**2 + (xs[1] - xt[1])**2 )

        if d <= self.range or sim is True:
            if abs(xt[0]-xs[0]) < 0.01 or abs(xt[1] - xs[1]) < 0.01:
                return np.zeros(self.nM)
            else:
                return np.array([
                        wrap2Pi(arctan2(xs[0] - xt[0], (xs[1] - xt[1]) )),
                        wrap2Pi(arctan2(xs[2], d))
                    ])
        else:
            return None

class ParticleFilter(object):

    def __init__(self):
        self.N = 100
        self.particles = self._create_uniform_particles([0.,1.0], [0., 1.0], self.N) # generate the particles
        self.weights = np.ones(self.N)
        self.weights /= sum(self.weights)
        self.sensor_model = BearingOnlySensor()

    def update(self, agent_state, z):
        self.predict(self.particles)
        self.updateParticles(agent_state, self.particles, self.weights, z)
        N = len(self.particles)
        indexes = multinomial_resample(self.weights)
        # self.resample_from_index(self.particles, self.weights, indexes)
        # self.simple_resample(self.particles, self.weights)
        if self.neff(self.weights) < N/2:
            indexes = systematic_resample(self.weights)
            self.resample_from_index(self.particles, self.weights, indexes)

    def predict(self, particles):
        N = len(particles)
        particles[:,0] += randn(N)*0.01 # add some noise to the particles
        particles[:,1] += randn(N)*0.01

    def updateParticles(self, agent_state, particles, weights, z):
        # weights.fill(1.)
        # sub_idx = randint(0, self.N, size=self.N/4)
        #
        # for i in sub_idx:
        #     for zi in z:
        #         if zi is not None:
        #             y_bel = self.sensor_model.h(particles[i], agent_state, sim=True)
        #             weights[i] *= pdf(zi, self.sensor_model.R, y_bel)

        for i,particle in enumerate(particles):
            for zi in z:
                if zi is not None:
                    y_bel = self.sensor_model.h(particle, agent_state, sim=True)
                    # weights[i] *= scipy.stats.multivariate_normal(zi, self.sensor_model.R).pdf(y_bel)
                    weights[i] *= pdf(zi, self.sensor_model.R, y_bel)
                    # weights[i] *= pdf(y_bel, self.sensor_model.R, zi)

        weights += 1e-300
        weights /= sum(weights)

    def neff(self,weights):
        return 1.0/np.sum(np.square(weights))

    def simple_resample(self, particles, weights):
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, random(N))

        # resample according to indexes
        particles[:] = particles[indexes]
        weights.fill(1.0 / N)
        # weights /= sum(weights)

    def resample_from_index(self, particles, weights, indexes):
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        weights /= sum(weights)
        # weights.fill(1.0/len(weights))

    def estimate(self):
        """returns mean and variance of the weighted particles"""

        pos = self.particles[:, 0:2]
        mean = np.average(pos, weights=self.weights, axis=0)
        var  = np.average((pos - mean)**2, weights=self.weights, axis=0)
        return mean, var

    def _create_uniform_particles(self,x_range, y_range, N):
        particles = np.empty((N,3)) # create the list of particles
        particles[:,0] = uniform(x_range[0], x_range[1], size=N)
        particles[:,1] = uniform(y_range[0], y_range[1], size=N)
        return particles # return particles

if __name__ == '__main__':
    particle_filter = ParticleFilter()
    agent_state = np.array([0.5,0.5,1.0])
    target_state1 = np.array([0.45,0.45,0.0])
    target_state2 = np.array([0.55,0.55,0.0])
    targets = np.vstack((target_state1, target_state2))
    sensor = BearingOnlySensor()
    plt.ion()
    seed(6)
    for i in range(2000):
        plt.clf()
        plt.scatter(particle_filter.particles[:,0], particle_filter.particles[:,1], c=particle_filter.weights)
        plt.scatter(targets[:,0], targets[:,1],color='r')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.draw()
        plt.pause(0.01)

        agent_state[0] = uniform(0.,1.0)
        agent_state[1] = uniform(0.,1.0)
        yk = []
        if i % 2 == 0:
            yk.append(sensor.h(target_state1, agent_state))
        else:
            yk.append(sensor.h(target_state2, agent_state))

        particle_filter.update(agent_state, yk)

        

        mean, var = particle_filter.estimate()
        print('Estimate : {}, Variance : {}'.format(mean, var))
