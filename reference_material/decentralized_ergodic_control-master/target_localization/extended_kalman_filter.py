import numpy as np
from numpy import arctan, pi, arctan2
from integrators import rk4Step

def wrap2Pi(x):
    ''' helper function '''
    x = np.mod(x+pi, 2*pi)
    if x < 0:
        x += 2*pi
    return x-pi

class BearingOnlySensor(object):

    def __init__(self):

        self.nM = 2
        self.R = np.diag([0.01] * self.nM)
        self.Rinv = np.linalg.inv(self.R)
        self.range = 0.25
    def h(self, xt, xs, ekf=False):
        d = np.sqrt( (xs[0] - xt[0])**2 + (xs[1] - xt[1])**2 )

        if d <= self.range or ekf is True:
            if abs(xt[0]-xs[0]) < 0.01 or abs(xt[1] - xs[1]) < 0.01:
                return np.zeros(self.nM)
            else:
                return np.array([
                        wrap2Pi(arctan2(xs[0] - xt[0], (xs[1] - xt[1]) )),
                        wrap2Pi(arctan2(xs[2], d))
                        ])
        else:
            return None
    def hdx(self, xt, xs):
        if abs(xt[0]-xs[0]) < 0.01 or abs(xt[1] - xs[1]) < 0.01:
            return np.zeros((self.nM, self.nM))
        else:
            temp1 = (xs[0] - xt[0])**2 + (xs[1] - xt[1])**2
            temp2 = 2.0 * np.sqrt( (xs[0]-xt[0])**2 + (xs[1]-xt[1])**2  ) * (xs[2]**2 + (xt[0] - xs[0])**2 + (xt[1] - xs[1])**2)
            return np.array([
                [(xt[1]-xs[1])/temp1, -(xt[0] - xs[0])/temp1],
                [-(xs[2]*(2.0*xt[0]-2.0*xs[0]))/temp2, -(xs[2]*(2.0*xt[1]-2.0*xs[1]))/temp2]
            ])

    def fisher_information_matrix(self, xt, xs):
        _hdx = self.hdx(xt, xs)
        return _hdx.T.dot(self.Rinv).dot(_hdx)

class DiffusionProcess(object):

    def __init__(self, time_step):
        self.time_step = time_step
        self.nX=2
        self.A = np.eye(self.nX)
    def f(self, x, u):
        return u
    def step(self,x, u):
        return rk4Step(self.f, x, self.time_step, *(u,))
    def Fdx(self, x, u):
        return self.A


class EKF(object):

    def __init__(self, sys_dyn, sensor_model):
        self.sys_dyn = sys_dyn
        self.sensor_model = sensor_model

        self.mu = np.array([0.41] * self.sys_dyn.nX)
        self.Q = np.diag([0.01] * self.sys_dyn.nX)
        self.sigma = np.diag([1.0] * self.sys_dyn.nX)
        self.I = np.eye(self.sys_dyn.nX)

    def update(self, measurement, sensor_state):

        uk = np.array([0.,0.]) + np.random.normal([0.01]*2, [0.01,0.01])# model as noisey process
        # prediction
        xk = self.sys_dyn.step(self.mu, uk)
        A = self.sys_dyn.Fdx(self.mu, uk)
        H = self.sensor_model.hdx(xk, sensor_state)
        if np.linalg.norm(H) < 1e-3:
            self.mu = xk
            self.sigma = A.dot(self.sigma).dot(A.T) + self.Q
        else:
            P = A.dot(self.sigma).dot(A.T) + self.Q
            if measurement is not None:
                yk = measurement - self.sensor_model.h(xk, sensor_state, ekf=True)
                yk[0] = wrap2Pi(yk[0])
                yk[1] = wrap2Pi(yk[1])
                S = H.dot(P).dot(H.T) + self.sensor_model.R
                Sinv = np.linalg.inv(S)
                K = P.dot(H.T).dot(Sinv)
                mu = xk + K.dot(yk)
                if np.linalg.norm(self.mu - mu) < 0.2:
                    self.mu = mu
                    self.sigma = (self.I - K.dot(H)).dot(P)
            else:
                self.mu = xk
                self.sigma = P


if __name__ == '__main__':
    process = DiffusionProcess(0.01)
    sensor = BearingOnlySensor()
    ekf = EKF(process, sensor)

    xs = np.array([0.5,0.5,1])
    xt = np.array([0.9,0.2,0])
    yk = sensor.h(xt, xs)
    for i in range(60):
        ekf.update(yk, xs)
        print ekf.mu
        print ekf.sigma, np.linalg.det(ekf.sigma)
