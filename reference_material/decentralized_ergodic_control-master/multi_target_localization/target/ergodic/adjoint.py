import numpy as np
from integrators import rk4Step, eulerStep
import matplotlib.pyplot as plt

class Adjoint(object):

    def __init__(self, system_model, objective, time_step):
        self.time_step = time_step
        self.system_model = system_model
        self.objective = objective
    def rhodt(self, rho, x, u, *args):
        """ Adjoint differential equation """
        return -self.objective.ldx(x, *args) - self.system_model.fdx(x, u).T.dot(rho)

    def simulate(self, rhof, x, u, t0, tf, ck, phik, N=None):
        """ Simulate the adjoint differential equation """
        if N is None:
            N = int(np.rint((tf - t0) / self.time_step))
        rho = [None] * N
        rho[N-1] = rhof
        for i in reversed(range(1, N)):
            rho[i-1] = rk4Step(self.rhodt, rho[i], -self.time_step, \
                                                    *(x[i], u[i-1], ck, phik) )

        return rho
