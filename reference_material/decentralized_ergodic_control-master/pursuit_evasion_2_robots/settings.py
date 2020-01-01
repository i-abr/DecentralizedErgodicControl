import numpy as np
from system_dynamics import DoubleIntegrator, CartPendulum, Quadcopter
from objective import Objective
from adjoint import Adjoint
from ergodic_utilities.basis import Basis
from ergodic_utilities.barrier import BarrierFunction
from information_density import PursuerExpectedInformation, EvaderExpectedInformation

class Settings(object):

    def __init__(self, time_step = 0.1, robot_type = 'pursuer'):
        self.time_step = time_step
        self.horizon = 2.
        self.q = 50
        self.Q = np.diag([0.,0.,50] +  [1.0]*3 + [1.0]*6)
        self.Qf = self.Q*0.1
        self.R = np.diag([0.1]*4)
        self.umax = [-2.0,2.0]
        self.xlim = [[0.0,1.0],[0.0,1.0]]
        self.u0 = np.array([0.2,0.,0.,0.])
        self.max_iter = 20

        self.coef = np.array([2]*2)
        self.lamk = np.zeros(self.coef+1)
        for i in range(self.coef[0] + 1):
            for j in range(self.coef[1] + 1):
                # self.lamk[i,j] = np.exp(-0.8 * np.linalg.norm([i,j]))
                self.lamk[i,j] = 1.0/(np.linalg.norm([i,j])+1)**(3.0/2.0)

        self.basis = Basis(self.xlim, self.coef)
        if robot_type is 'pursuer':
            self.eid = PursuerExpectedInformation(self.xlim, self.coef, self.basis)
        else:
            self.eid = EvaderExpectedInformation(self.xlim, self.coef, self.basis)
        self.barrier = BarrierFunction(self.xlim, barr_weight=10000.0)
        self.system_model = Quadcopter(self.time_step)
        # self.system_model = CartPendulum(self.time_step)
        self.objective = Objective(self.Q, self.Qf, self.q,\
                        self.lamk, self.barrier, self.basis, self.coef, self.time_step)

        self.adjoint = Adjoint(self.system_model, self.objective, self.time_step)
