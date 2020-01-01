import numpy as np
import scipy.linalg


class LQR(object):
    '''
        Infinite time horizon lqr controller class
    '''

    def __init__(self, A, B, Q, R, target_state=None):
        self.A, self.B = A, B # linearized dynamics
        self.Q, self.R = Q, R # weights


        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.Klqr = np.linalg.inv(R).dot(B.T.dot(P))


        if target_state is not None:
            self.xd = target_state
        else:
            self.xd = np.zeros(self.A.shape[0])
    @property
    def dx(self):
        return -self.Klqr.copy()
        
    def __call__(self, x, xd=None):
        if xd is None:
            return -self.Klqr.dot(x - self.xd)
        else:
            return -self.Klqr.dot(x - xd)
