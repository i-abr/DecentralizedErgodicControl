import numpy as np
import matplotlib.pyplot as plt
from integrators import monte_carlo
import copy
import time
class CKState(object):

    def __init__(self, coef, basis, time_step, memory, k_list):
        self.time_step = time_step
        self.coef = coef
        self.basis = basis
        self.cki = np.zeros(self.coef+1).ravel()
        self.k_list = k_list
        self.cktemp = np.zeros(self.coef+1).ravel()

        self.remember_state_counter= memory
        self.current_state_counter = 0
        self.state_history = []

    def get_ck_from_x(self, x):
        N = len(x)
        for i,k in enumerate(self.k_list):
            _fk = [self.basis.fk(k, x[ii]) for ii in range(N)]
            self.cktemp[i] = np.sum(_fk)
        return self.cktemp.copy()/ float(N)

    def update_ck(self, x):

        if self.current_state_counter < self.remember_state_counter:
            self.state_history.append(x)
            self.current_state_counter += 1
        else:
            del self.state_history[0]
            self.state_history.append(x)
        self.cki = self.get_ck_from_x(self.state_history)
        return self.cki.copy()

class SAC(object):

    def __init__(self, param):
        self.time_step = param.time_step
        self.objective = param.objective
        self.system_model = param.system_model
        self.umax = param.umax
        self.horizon = param.horizon
        self.adjoint = param.adjoint
        self.max_iter = param.max_iter
        self.R = param.R
        self.Rinv = np.linalg.inv(self.R)
        # self.udef = DefaultControl(self.horizon, param.time_step, param.u0, self.umax)
        self.dJdlam = np.array([0.0] * int(self.horizon/param.time_step))
        self.ustar = np.array([param.u0] * int(self.horizon/param.time_step))
        self.ck = CKState(param.coef, param.basis, param.time_step, param.ergodic_memory, param.k_list)
        self.u0 = param.u0
        self.udef = [param.u0] * (int(self.horizon/param.time_step)-1)

    def getCKFromDefaultControl(self, x0):
        N = int(self.horizon/self.time_step)
        x = self.system_model.simulate(x0, self.udef, 0.0, 0.0+self.horizon, N=N)
        return self.ck.get_ck_from_x(self.ck.state_history + x[1:])

    def __call__(self, x0, phik, ck = None, t0=0.0):
        N = int(self.horizon/self.time_step)
        x = self.system_model.simulate(x0, self.udef, t0, t0+self.horizon, N=N)
        if ck is None:
            cki = self.ck.get_ck_from_x(self.ck.state_history + x[1:])
        else:
            cki = ck.copy()
        Jinit = self.objective.getCost(x, cki, phik)
        rhof = self.objective.mdx(x[-1])
        rho = self.adjoint.simulate(rhof, x, self.udef, t0, t0+self.horizon, cki, phik, N=N)
        # plt.plot(rho)
        ustar = []
        dJdlam = []
        for k in range(len(x)-1):
            B = self.system_model.fdu(x[k], self.udef[k])
            ustar.append(self.Rinv.dot(-B.T.dot(rho[k])))
            # f1 = self.system_model.f(x[k], self.udef[k])
            # f2 = self.system_model.f(x[k], ustar[k])
            # dJdlam.append(rho[k].dot(f2 - f1))

        # tau = np.argmin(dJdlam)
        # self.udef[tau] = np.clip(ustar[tau], self.umax[0], self.umax[1])
        # self.udef.add_to_temp(uOpt)
        # usat = np.clip(self.ustar, self.umax[0], self.umax[1])
        # x = self.system_model.simulate(x0, self.udef, t0, t0+self.horizon)
        # Jnew = self.objective.getCost(x, cki, phik)
        # if Jnew - Jinit < 0:
        #     # self.udef.add_to_default()
        #     return usat[0]
        # else:
        #     return self.udef[k]

        # tau = np.argmin(dJdlam)
        # uOpt = ustar[tau]
        i = 1
        Jnew = np.inf
        # utemp = self.udef.utemp
        while Jnew - Jinit > 0.0:
            beta = 0.2**i
            # utemp = self.udef.add_to_temp(beta * self.ustar)
            utemp = []
            for ii in range(len(ustar)):
                utemp.append(np.clip(self.udef[ii] + beta*ustar[ii], self.umax[0], self.umax[1]) )
            x = self.system_model.simulate(x0, utemp, t0, t0+self.horizon)
            cki = self.ck.get_ck_from_x(self.ck.state_history + x[1:]) # totally forgot to do this
            Jnew = self.objective.getCost(x, cki, phik)
            i += 1
            if i > self.max_iter:
                print('Did not improve cost..')
                break
        # print('Default Cost : {}, Updated Cost : {}'.format(Jinit, Jnew))
        # print self.udef.u
        # print self.udef.utemp
        if i < self.max_iter:
            # self.udef.add_to_default(beta * self.ustar)
            self.udef = utemp
            # print "Improvement in Objective : ", Jnew - Jinit, " \t Steps :", i
        # print self.udef[0:5]
        unow = self.udef[0]
        # print unow
        del self.udef[0]
        # print self.udef[0:5]
        # raw_input()
        self.udef.append(self.u0.copy())
        return unow
