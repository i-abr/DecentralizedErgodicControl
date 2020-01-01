import numpy as np

class Model(object):

    def __init__(self):

        self.observation_space  = np.array([[0., 0., -np.inf, -np.inf],[1., 1., np.inf, np.inf]], dtype=np.float32).T
        self.action_space       = np.array([[-1., -1.],[+1., +1.]], dtype=np.float32)
        self.explr_space        = np.array([[0., 0.],[1., 1.]], dtype=np.float32)
        self.explr_idx          = [0, 1]

        self.dt = 0.1

        self._A = np.array([
                [0., 0., 0.8, 0.],
                [0., 0., 0., 0.8],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]
        ])

        self._B = np.array([
                [0., 0.],
                [0., 0.],
                [1.0, 0.],
                [0., 1.0]
        ])

        self.reset()

    def reset(self, state=None):
        '''
        Resets the property self.state
        '''
        if state is None:
            self.state = np.random.uniform(0.1, 0.2, size=(self.observation_space.shape[0],))
        else:
            self.state = state.copy()

        return self.state.copy()

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
        '''
        Continuous time dynamics
        '''
        return np.dot(self._A, x) + np.dot(self._B, u)

    def step(self, a):
        state = self._state + self.f(self._state, a) * self.dt
        self._state = state
        return state.copy()
