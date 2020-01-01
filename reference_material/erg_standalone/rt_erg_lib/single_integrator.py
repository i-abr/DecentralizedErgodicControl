import numpy as np

class SingleIntegrator(object):


    def __init__(self):

        self.observation_space = np.array([[0., 0.],
                                            [1., 1.]], dtype=np.float32)

        self.action_space = np.array([[-1., -1.],
                                        [+1., +1.]], dtype=np.float32)

        self.explr_space = np.array([[0., 0.],
                                    [1., 1.]], dtype=np.float32)

        self.explr_idx  = [0, 1]

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


    def fdx(self, x, u):
        '''
        State linearization
        '''
        return self.A.copy()

    def fdu(self, x):
        '''
        Control linearization
        '''
        return self.B.copy()

    def reset(self, state=None):
        '''
        Resets the property self.state
        '''
        if state is None:
            self.state = np.random.uniform(0., 0.9, size=(self.observation_space.shape[0],))
        else:
            self.state = state

        return self.state.copy()

    def f(self, x, u):
        '''
        Continuous time dynamics
        '''
        return np.dot(self.A, x) + np.dot(self.B, u)

    @property
    def state(self):
        return self._state.copy()

    @state.setter
    def state(self, x):
        self._state = x.copy()

    def step(self, a, save=False):
        # TODO: include ctrl clip
        state = self._state + self.f(self._state, a) * self.dt
        if save:
            self._state = state
        return state.copy()
