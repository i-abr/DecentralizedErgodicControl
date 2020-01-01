import numpy as np
from integrators import rk4Step, eulerStep
from numpy import cos, sin, pi, dot, outer

class DoubleIntegrator(object):

    def __init__(self, time_step):

        self.nX = 4
        self.nU = 2

        self.time_step = time_step

        self.A = np.diag([1,1], 2) - np.diag([0,0,1,1]) * 0.25
        self.B = np.vstack((np.zeros((2,2)), np.diag([0.4,0.4])))
    def f(self, x, u):

        return self.A.dot(x) + self.B.dot(u)

    def fdx(self, x, u):
        return self.A

    def fdu(self, x, u):
        return self.B

    def simulate(self, x0, u, t0, tf):
        """ Simulate the adjoint differential equation """
        N = int((tf - t0) / self.time_step)
        x = [None] * N
        x[0] = x0
        for i in range(1,N):
            x[i] = rk4Step(self.f, x[i-1], self.time_step, \
                                                    *(u[i-1],) )
        return x
    def step(self, x0, u0):
        return rk4Step(self.f, x0, self.time_step, *(u0,))




class CartPendulum:
    '''
    default class for a cart pendulum
    This makes code more generalizable for different integration schemes
    like VI's and what not
    '''
    def __init__(self, time_step, l=1., m=1., g=9.81, forcing=True):
        self.nX = 4
        self.nU = 1
        # physical parameters of the cart pendulum
        self.g = g
        self.l = l
        self.m = m
        self.time_step = time_step

    def f(self, x, u):
        '''
        x[0] = theta
        x[1] = x
        x[2] = thetadot
        x[3] = xdot
        '''
        dxdt = np.array([x[2], x[3], 9.81*sin(x[0])/self.l + u[0]*cos(x[0])/self.l, u[0]])
        return dxdt
    def fdx(self, x, u):
        '''
        df/dx linearization
        '''
        dfdx = np.array([
                            [0.,0.,1.,0.],
                            [0.,0.,0.,1.],
                            [self.g*cos(x[0])/self.l - u[0]*sin(x[0])/self.l, 0., 0., 0.],
                            [0.,0.,0.,0.]])
        return dfdx
    def fdu(self, x, u):
        '''
        df/du linearization
        '''
        dfdu = np.array([[0.,0.,cos(x[0])/self.l,1.]]).T
        return dfdu

    def simulate(self, x0, u, t0, tf):
        """ Simulate the adjoint differential equation """
        N = int((tf - t0) / self.time_step)
        x = [None] * N
        x[0] = x0
        for i in range(1,N):
            x[i] = rk4Step(self.f, x[i-1], self.time_step, \
                                                    *(u[i-1],) )
        return x
    def step(self, x0, u0):
        return rk4Step(self.f, x0, self.time_step, *(u0,))


class Quadcopter(object):

    def __init__(self, time_step):
        self.time_step = time_step
        self.nX = 12
        self.nU = 4
        self.m = 0.1
        self.damping = 0
        self.A = np.zeros((self.nX, self.nX)) +  np.diag([1.0]*6, 6)
        self.B = np.zeros((self.nX, self.nU))

    def f(self, x, u):

        psi = x[3]
        theta = x[4]
        phi = x[5]

        xddot = u[0] * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)) / self.m
        yddot = u[0] * (cos(phi) * sin(theta) * sin(psi) - cos(psi) * sin(phi)) / self.m
        zddot = u[0] * cos(theta) * cos(phi)/self.m  -  9.81

        psiddot = u[1] - self.damping * x[9]
        thetaddot = u[2] - self.damping * x[10]
        phiddot = u[3] - self.damping * x[11]

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
        self.A[9,9] = -self.damping
        self.A[10,10] = -self.damping
        self.A[11,11] = -self.damping
        return self.A
    def fdu(self, x, u):
        psi = x[3]
        theta = x[4]
        phi = x[5]
        self.B[6,0] = (cos(phi) * cos(psi) * sin(theta) + sin(phi) * sin(psi) )/ self.m
        self.B[7,0] = (-cos(psi) * sin(phi) + cos(phi) * sin(theta) * sin(psi)) / self.m
        self.B[8,0] = cos(theta) * cos(phi) / self.m
        self.B[9,1] = 1.0
        self.B[10,2] = 1.0
        self.B[11,3] = 1.0
        return self.B
    def simulate(self, x0, u, t0, tf, N=None):
        """ Simulate the adjoint differential equation """
        if N is None:
            N = int(np.rint(np.rint((tf - t0)/self.time_step)))
        x = [None] * N
        x[0] = x0
        for i in range(1,N):
            x[i] = rk4Step(self.f, x[i-1], self.time_step, \
                                                    *(u[i-1],) )
        return x
    def step(self, x0, u0):
        return rk4Step(self.f, x0, self.time_step, *(u0,))
