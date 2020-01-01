import numpy as np

class Objective(object):

    def __init__(self, Q, Qf, q, lamk, barrier, basis, coef, time_step):
        self.Q = Q
        self.Qf = Qf
        self.q = q
        self.lamk = lamk.ravel()
        self.time_step = time_step
        self.barrier = barrier
        self.basis = basis
        self.coef = coef
        self.k_list = []
        for i in range(self.coef[0] +1):
            for j in range(self.coef[1]+1):
                self.k_list.append([i,j])

    def l(self, x):
        """ Running Cost """
        return x.dot(self.Q.dot(x)) + self.barrier.barr(x)

    def ldx(self, x, ck, phik):
        dJerg = np.zeros(x.shape)
        dErg = (ck - phik).ravel()
        for i,k in enumerate(self.k_list):
            dJerg += self.lamk[i] * dErg[i] * self.basis.dfk(k, x)
        # for i in range(self.coef[0]+1):
        #     for j in range(self.coef[1]+1):
        #         dJerg += self.lamk[i,j] * (ck[i,j] - phik[i,j]) * self.basis.dfk([i,j], x)
        return 2.0*self.q*dJerg + self.Q.dot(x) + self.barrier.dbarr(x)

    def m(self, x):
        """ Terminal Cost """
        return x.dot(self.Qf.dot(x))

    def mdx(self, x):
        return self.Qf.dot(x)

    def getCost(self, x, ck, phik):
        dErg = (ck-phik).ravel()
        J = self.q * np.sum(self.lamk * (dErg)**2)
        for i in range(len(x)):
            J += self.l(x[i]) * self.time_step

        return J + self.m(x[-1])
