import numpy as np
from .integrators import rk4Step, eulerStep
from numpy import cos, sin, pi, dot, outer

class Quadcopter(object):

    def __init__(self):
        self.nX = 12
        self.nU = 4
        self.m = 0.1
        self.damping = [0.]*3 + [0.]*3
        self.A = np.zeros((self.nX, self.nX)) +  np.diag([1.0]*6, 6)
        self.B = np.zeros((self.nX, self.nU))

    def f(self, x, u):

        psi = x[3]
        theta = x[4]
        phi = x[5]

        xddot = u[0] * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)) / self.m - self.damping[0]*x[6]
        yddot = u[0] * (cos(phi) * sin(theta) * sin(psi) - cos(psi) * sin(phi)) / self.m - self.damping[1]*x[7]
        zddot = u[0] * cos(theta) * cos(phi)/self.m  -  9.81 - self.damping[2] * x[8]

        psiddot = u[1] - self.damping[3] * x[9]
        thetaddot = u[2] - self.damping[4] * x[10]
        phiddot = u[3] - self.damping[5] * x[11]

        return np.array([
                x[6],
                x[7],
                x[8],
                x[9],
                x[10],
                x[11],
                xddot,
                yddot,
                zddot,
                psiddot,
                thetaddot,
                phiddot
            ])

    def fdx(self, x, u):
        psi = x[3]
        theta = x[4]
        phi = x[5]
        # A = np.zeros((self.nX, self.nX)) +  np.diag([1.0]*6, 6)
        self.A[6,3] = u[0] * (cos(psi) * sin(phi) - cos(phi) * sin(theta)*sin(psi) )/self.m
        self.A[6,4] = u[0] * cos(theta) * cos(phi) * cos(psi) / self.m
        self.A[6,5] = u[0] * (-cos(psi) * sin(theta) * sin(phi) + cos(phi) * sin(psi))/self.m
        self.A[7,3] = u[0] * (cos(phi) * cos(psi)*sin(theta) + sin(phi)*sin(psi) )/self.m
        self.A[7,4] = u[0] * cos(theta) * cos(phi) * sin(psi) / self.m
        self.A[7,5] = u[0] * (-cos(phi) * cos(psi) - sin(theta) * sin(phi) * sin(psi))/self.m
        self.A[8,4] = -u[0] * cos(phi) * sin(theta) / self.m
        self.A[8,5] = -u[0] * cos(theta) * sin(phi) / self.m
        # self.A[9,9] = -self.damping[3]
        # self.A[10,10] = -self.damping[4]
        # self.A[11,11] = -self.damping[5]
        return self.A
    def fdu(self, x, u):
        psi = x[3]
        theta = x[4]
        phi = x[5]
        self.B[6,0] = (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)) / self.m
        self.B[7,0] = (cos(phi) * sin(theta) * sin(psi) - cos(psi) * sin(phi)) / self.m
        self.B[8,0] = cos(theta) * cos(phi) / self.m
        self.B[9,1] = 1.0
        self.B[10,2] = 1.0
        self.B[11,3] = 1.0
        return self.B

    def get_linearization(self, x, u):
        return self.fdx(x, u), self.fdu(x, u)

    def simulate(self, x0, ts, N, u=None, uofx=None):
        """ Simulate the adjoint differential equation """

        x = [None] * N
        x[0] = x0.copy()
        _u = [None] * (N-1)
        for i in range(1,N):
            if uofx is not None:
                ueval = uofx(x[i-1])
                _u[i-1] = ueval
                x[i] = rk4Step(self.f, x[i-1], ts, *(ueval,) )
            else:
                x[i] = rk4Step(self.f, x[i-1], ts, *(u[i-1],) )
        return x, _u

    def step(self, x0, u0, ts):
        return rk4Step(self.f, x0, ts, *(u0,))
