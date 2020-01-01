import numpy as np
from numpy import cos, sin
from lqr import LQR

target_state = np.array([0. for _ in range(12)])
target_state[0] = 0.
target_state[1] = 0.
target_state[2] = 1.0

# Sort of just copied this from else where
Q, R = np.diag([0.0,0.0,100.] + [0.0001]*9), np.diag([0.1,0.1,0.1,0.1])

class Model(object):

    def __init__(self):

        self.observation_space = np.array([[-np.inf]*12,[np.inf]*12], dtype=np.float32).T
        self.action_space       = np.array([[-1., -1., -1, -1],[+1., +1., +1, +1]], dtype=np.float32).T
        self.explr_space        = np.array([[0., 0.],[1., 1.]], dtype=np.float32)
        self.explr_idx          = [0, 1]

        self.dt = 0.05
        self.m = 0.1
        self.damping = [0.]*3 + [0.]*3

        self._A = np.zeros((12,12)) +  np.diag([1.0]*6, 6)
        self._B = np.zeros((12,4))


        self.reset()


    def reset(self, state=None):
        '''
        Resets the property self.state
        '''
        if state is None:
            self._state = np.zeros(self.observation_space.shape[0])
            self._state[:2] = np.random.uniform(0.1, 0.2, size=(2,))
            self._state[3:6] = np.random.normal(0., 0.1, size=(3,))
        else:
            self._state = state.copy()
        return self._state.copy()

    @property
    def state(self):
        return self._state.copy()
    @state.setter
    def state(self, x):
        self._state = x.copy()
    @property
    def A(self, x=None, u=None):
        return self._A.copy()
    @property
    def B(self, x=None, u=None):
        return self._B.copy()

    def f(self, x, u):

        psi = x[3]
        theta = x[4]
        phi = x[5]

        xddot = u[0] * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)) / self.m - self.damping[0]*x[6]
        yddot = u[0] * (cos(phi) * sin(theta) * sin(psi) - cos(psi) * sin(phi)) / self.m - self.damping[1]*x[7]
        zddot = u[0] * cos(theta) * cos(phi)/self.m - self.damping[2] * x[8]

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
        self._A[6,3] = u[0] * (cos(psi) * sin(phi) - cos(phi) * sin(theta)*sin(psi) )/self.m
        self._A[6,4] = u[0] * cos(theta) * cos(phi) * cos(psi) / self.m
        self._A[6,5] = u[0] * (-cos(psi) * sin(theta) * sin(phi) + cos(phi) * sin(psi))/self.m
        self._A[7,3] = u[0] * (cos(phi) * cos(psi)*sin(theta) + sin(phi)*sin(psi) )/self.m
        self._A[7,4] = u[0] * cos(theta) * cos(phi) * sin(psi) / self.m
        self._A[7,5] = u[0] * (-cos(phi) * cos(psi) - sin(theta) * sin(phi) * sin(psi))/self.m
        self._A[8,4] = -u[0] * cos(phi) * sin(theta) / self.m
        self._A[8,5] = -u[0] * cos(theta) * sin(phi) / self.m
        # self.A[9,9] = -self.damping[3]
        # self.A[10,10] = -self.damping[4]
        # self.A[11,11] = -self.damping[5]



    def fdu(self, x, u):
        psi = x[3]
        theta = x[4]
        phi = x[5]
        self._B[6,0] = (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)) / self.m
        self._B[7,0] = (cos(phi) * sin(theta) * sin(psi) - cos(psi) * sin(phi)) / self.m
        self._B[8,0] = cos(theta) * cos(phi) / self.m
        self._B[9,1] = 1.0
        self._B[10,2] = 1.0
        self._B[11,3] = 1.0

    def step(self, a, ldx=False):
        self.fdx(self._state.copy(), a)
        self.fdu(self._state.copy(), a)
        _ldx = np.dot(Q, self.state-target_state)

        state = self._state + self.f(self._state, a) * self.dt
        self._state = state
        if ldx:
            return self.state, _ldx
        return state.copy()
