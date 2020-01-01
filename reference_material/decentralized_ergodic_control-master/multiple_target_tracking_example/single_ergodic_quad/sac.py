import numpy as np
from .adjoint import Adjoint
import matplotlib.pyplot as plt
import copy
class SAC(object):


    def __init__(self, dynamics, objective, time_step, horizon_in_sec, default_control=None):
        ''' SAC Controller '''
        if default_control is None:
            self.uofx = lambda x: np.array([.0,.0,.0,.0])
        else:
            self.uofx = default_control
        self.unom = np.array([.0,.0,.0,.0])
        self.udef = None
        self.umax = [-1.0,1.0]

        self.time_step = time_step
        self.horizon = int(horizon_in_sec/time_step)
        self.R = np.diag([.1]*4)
        self.Rinv = np.linalg.inv(self.R)
        self.dynamics = dynamics
        self.objective = objective
        self.adjoint = Adjoint(dynamics, objective, default_control)

    def saturate(self, action):
        unorm = np.linalg.norm(action)
        saturated_action = None
        for u in action:
            if u < self.umax[0] or u > self.umax[1]:
                saturated_action = action / unorm  * self.umax[1]
        if saturated_action is None:
            return action
        else:
            return saturated_action

    def get_ck(self, x):
        return self.objective.trajectory_distribution.get_ck_from_trajectory([x])

    def __call__(self, init_state, cki=None):

        line_search = False

        if self.udef is None:
            x, self.udef = self.dynamics.simulate(init_state, self.time_step,
                                                    self.horizon, uofx=self.uofx)
        else:
            x, self.udef = self.dynamics.simulate(init_state, self.time_step,
                                            self.horizon, uofx=self.uofx)
            # x, _= self.dynamics.simulate(init_state, self.time_step,
            #                                 self.horizon, u=self.udef)
        Jinit = self.objective.trajectory_cost(x, cki=cki)
        if cki is not None:
            ck = cki
        else:
            ck = self.objective.trajectory_distribution.get_ck_from_trajectory(x)

        phik = self.objective.target_distribution.get_phik()
        rhof = self.objective.mdx(init_state)
        rho = self.adjoint.simulate(rhof, x, self.udef, self.time_step, self.horizon, ck, phik)
        dJdlam = [None] * (self.horizon - 1)
        ustar = [None] * (self.horizon - 1)
        alpha_d = -55.0
        for i in range(len(rho)-1):
            B = self.dynamics.fdu(x[i], self.udef[i])
            if line_search:
                ustar[i] = self.Rinv.dot(-B.T.dot(rho[i])) + self.udef[i]
                # omega = B.T.dot(np.outer(rho[i], rho[i])).dot(B)
                # ustar[i] = np.linalg.inv(omega + self.R.T).dot(omega.dot(self.udef[i]) + alpha_d * B.T.dot(rho[i]))
            else:
                ustar[i] = self.Rinv.dot(-B.T.dot(rho[i]))
            f1 = self.dynamics.f(x[i], self.udef[i])
            f2 = self.dynamics.f(x[i], ustar[i])
            dJdlam[i] = rho[i].dot(f2 - f1)

        tau = np.argmin(dJdlam)
        udef = copy.copy(self.udef)

        Jnew = np.inf
        i = 0
        break_flag = False

        if line_search:
            # regular Line search
            while Jnew - Jinit > 0.0:
                udef[tau+i] = np.clip(ustar[tau], self.umax[0], self.umax[1])
                # udef[tau+i] = self.saturate(ustar[tau])
                x, _ = self.dynamics.simulate(init_state, self.time_step,
                                                self.horizon, u=udef)
                Jnew = self.objective.trajectory_cost(x)
                i += 1
                if i > 4 or tau+i > len(udef)-1:
                    break_flag = True
                    break

        else:
            # Magnitude Search
            beta = 0.1
            while Jnew - Jinit > 0.0:
                for j, ui in enumerate(ustar):
                    udef[j] = np.clip( (beta ** i) * ui + self.udef[j], self.umax[0], self.umax[1])
                    # udef[j] = ui + self.udef[j]
                x, _ = self.dynamics.simulate(init_state, self.time_step,
                                                self.horizon, u=udef)
                Jnew = self.objective.trajectory_cost(x)
                i += 1
                if i > 4:
                    break_flag = True
                    break

        print('Iterations taken {}, Improvement : {}'.format(i, Jnew-Jinit))
        if break_flag is False:
            # self.udef[tau] = np.clip(ustar[tau], self.umax[0], self.umax[1])
            self.udef = udef

        # else:
        #     self.udef[0] = self.uofx(init_state)
        #     # self.udef[0] = np.clip(self.uofx(init_state), self.umax[0], self.umax[1])


        # Get the predicted trajectory ck
        x, _ = self.dynamics.simulate(init_state, self.time_step,
                                                    self.horizon, u=self.udef)

        unow = self.udef[0].copy()
        del self.udef[0]
        self.udef.append(self.unom.copy())
        # self.udef.append(self.uofx(x[-1]))
        return unow, self.objective.trajectory_distribution.get_ck_from_trajectory(x)
