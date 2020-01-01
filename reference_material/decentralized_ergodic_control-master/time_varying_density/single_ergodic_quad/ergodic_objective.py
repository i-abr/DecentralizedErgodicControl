import numpy as np
from .probabilities import TrajectoryDistribution, TargetDistribution, Basis
from .barrier import BarrierFunction
from .floor_plan import FloorPlan

class ErgodicObjective(object):

    def __init__(self, horizon, time_step):
        self.horizon = horizon
        self.q_ergodic = 100.0
        self.time_step = time_step
        self.x_lim = [[0.0, 1.0],[0.0, 1.0]]
        self.t_erg = 5.0
        self.ergodic_memory = int(self.t_erg/time_step)
        self.coefs = np.array([6]*2)

        # weights for the ergodic measure
        self.lamk = np.zeros(self.coefs+1)
        self.k_list = []
        for i in range(self.coefs[0] + 1):
            for j in range(self.coefs[1] + 1):
                # self.lamk[i,j] = np.exp(-0.8 * np.linalg.norm([i,j]))
                self.lamk[i,j] = 1.0/(np.linalg.norm([i,j])+1)**(3.0/2.0)
                self.k_list.append([i,j])
        self.lamk = self.lamk.ravel()

        self.basis = Basis(self.x_lim, self.coefs, self.k_list)
        self.trajectory_distribution = TrajectoryDistribution(self.coefs, self.k_list, self.ergodic_memory, self.x_lim)
        self.target_distribution = TargetDistribution(self.coefs, self.x_lim, self.k_list)
        self.barrier = BarrierFunction(self.x_lim, 2)

        self.floor_plan = FloorPlan()

    def l(self, x):
        ''' Running Cost '''
        return self.barrier.barr(x) + 2000.0* self.floor_plan.get_penality(x)

    def ldx(self, x, ck, phik):
        dJerg = np.zeros(x.shape)
        dErg = (ck - phik).ravel()
        for i,k in enumerate(self.k_list):
            dJerg += self.lamk[i] * dErg[i] * self.basis.dfk(k, x) / 3.0
        t_erg = self.trajectory_distribution.get_remembered_time() * self.time_step
        return 2.0*self.q_ergodic*dJerg/(self.horizon+t_erg) + self.barrier.dbarr(x) + 2000.0* self.floor_plan.first_order_derivative(x)

    def m(self, x):
        """ Terminal Cost """
        return 0.0

    def mdx(self, x):
        return np.zeros(x.shape)

    def trajectory_cost(self, x, cki=None):
        ck = self.trajectory_distribution.get_ck_from_trajectory(x)
        phik = self.target_distribution.get_phik()
        dErg = (ck-phik).ravel()
        J = 0.0
        for i in range(len(x)):
            J += self.l(x[i]) * self.time_step


        return J + self.m(x[-1]) + self.q_ergodic * np.sum(self.lamk * (dErg)**2)
