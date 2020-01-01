import numpy as np
from .integrators import rk4Step, eulerStep
import matplotlib.pyplot as plt

class Adjoint(object):
    '''
    Adjoint class for computing the adjoint differential equation
    '''
    def __init__(self, system_model, objective, default_control=None):
        self.system_model = system_model
        self.objective = objective
        self.default_control = default_control

    def rhodt(self, rho, x, u, *args):
        ''' Adjoint differential equation '''
        return -self.objective.ldx(x, *args) \
                - (self.system_model.fdx(x, u) + self.system_model.fdu(x,u).dot(self.default_control.udx())).T.dot(rho)

    def simulate(self, rhof, x, u, ts, N, *args, uofx=None):
        """ Simulate the adjoint differential equation """
        rho = [None] * N
        rho[N-1] = rhof
        for i in reversed(range(1, N)):

            rho[i-1] = rk4Step(self.rhodt, rho[i], -ts, *(x[i], u[i-1], *args) )

        return rho
