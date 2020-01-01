import numpy as np

class Model(object):

    def __init__(self):

        self.observation_space = np.array([[0., 0.],
                                            [1., 1.]], dtype=np.float32)
        self.action_space = np.array([[-1., -1.],
                                        [+1., +1.]], dtype=np.float32)
        self.dt = 0.4

        self.A = np.array([
                [0., 0.],
                [0., 0.]
        ])# - np.diag([0,0,1,1]) * 0.25

        self.B = np.array([
                [1.0, 0.],
                [0., 1.0]
        ])

        self.reset()

    def reset(self, state=None):
        '''
        Resets the property self.state
        '''
        if state is None:
            self.state = np.random.uniform(0., 0.9, size=(self.observation_space.shape[0],))
        else:
            self.state = state

        return self.state.copy()

    @property
    def state(self):
        return self._state.copy()

    @state.setter
    def state(self, x):
        self._state = x.copy()

    @property
    def A(self, x=None, u=None):
        return self.A.copy()

    @property
    def B(self, x=None, u=None):
        return self.B.copy()


    def f(self, x, u):
        '''
        Continuous time dynamics
        '''
        return np.dot(self.A, x) + np.dot(self.B, u)

    def step(self, a):
        state = self._state + self.f(self._state, a) * self.dt
        self._state = state
        return state.copy()
