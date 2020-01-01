import numpy as np
from system_dynamics import DoubleIntegratorR3, CartPendulum, Quadcopter
from objective import Objective
from adjoint import Adjoint
from basis import Basis
from barrier import BarrierFunction

class Settings(object):

    def __init__(self, time_step=0.05):
        self.time_step = time_step
        self.horizon = 2.
        self.q = 50
        self.Q = np.diag([0.,0.,50.0] +  [1.0]*3 + [6.0]*3 + [1.0]*3)
        self.Qf = self.Q*0.1
        self.R = np.diag([0.1]*4)
        self.ergodic_dim = 2
        self.umax = [-2.0,2.0]
        self.xlim = [[0.0,1.0]]*self.ergodic_dim
        self.u0 = np.array([0.2,0.,0.,0.])
        self.max_iter = 20
        self.ergodic_memory = 5
        self.coef = np.array([2]*self.ergodic_dim)
        self.lamk = np.zeros(self.coef+1)
        self.k_list = []
        for i in range(self.coef[0] + 1):
            for j in range(self.coef[1] + 1):
                # self.lamk[i,j] = np.exp(-0.8 * np.linalg.norm([i,j]))
                self.lamk[i,j] = 1.0/(np.linalg.norm([i,j])+1)**(3.0/2.0)
                self.k_list.append([i,j])

        self.lamk = self.lamk.ravel()
        self.basis = Basis(self.xlim, self.coef, self.k_list)

        self.barrier = BarrierFunction(self.xlim, self.ergodic_dim, barr_weight=10000.0)

        self.system_model = Quadcopter(self.time_step)

        # self.system_model = CartPendulum(self.time_step)
        self.objective = Objective(self.Q, self.Qf, self.q, self.k_list,\
                        self.lamk, self.barrier, self.basis, self.coef, self.time_step)

        self.adjoint = Adjoint(self.system_model, self.objective, self.time_step)
