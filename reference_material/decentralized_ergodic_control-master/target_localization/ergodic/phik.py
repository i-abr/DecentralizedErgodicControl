import numpy as np
from numpy import exp, sqrt, pi, cos, sin
import matplotlib.pyplot as plt
import time
from integrators import monte_carlo
from scipy.integrate import nquad

class Phik(object):

    def __init__(self, settings):
        self.basis = settings.basis
        self.coef = settings.coef
        self.xlim = settings.xlim
        self.k_list = settings.k_list


        # phi_temp = lambda x,y: np.ones(len(x))
        phi_temp = lambda x,y: exp( -50*(x-.2)**2)*exp(-50*(y-.7)**2) + exp( -50*(x-.7)**2)*exp(-50*(y-.3)**2)
        # phi_temp = lambda x,y: 1.0
        # normfact = monte_carlo(lambda x,y: phi_temp(x,y), self.xlim, n=200)
        self.normfact = nquad(lambda x,y: phi_temp(x,y), self.xlim)[0]
        self.phi = lambda x,y: phi_temp(x,y)/self.normfact
        self.phik = self.get_phik(self.phi)
        self.samples = None

        target_space = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))
        search_space = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))

        self.target_space = np.c_[target_space[0].ravel(), target_space[1].ravel()]
        self.search_space = np.c_[search_space[0].ravel(), search_space[1].ravel()]

    def get_phik(self, phi):
        phik = np.zeros(self.coef+1).ravel()
        for i,k in enumerate(self.k_list):
            temp_fun = lambda x,y: phi(x,y) * self.basis.fk(k, [x,y])
            phik[i],_ = monte_carlo(temp_fun, self.xlim, n=200)
            # phik[i] = nquad(temp_fun, self.xlim)[0
        phik /= phik[0]
        return phik

    # def update_eid(self, time):
    #     x_pos = 0.25*cos(0.25 * time) + 0.5
    #     y_pos = 0.25*sin(0.25 * time) + 0.5
    #     # einv = np.linalg.inv(np.diag([0.01,0.01]))
    #     # det1 = np.linalg.det(np.diag([0.01,0.01]))
    #     self.phi = lambda x,y: exp( -20*(x-x_pos)**2)*exp(-20*(y-y_pos)**2)/self.normfact
    #     # self.phi = lambda x,y: 1.0-exp(-0.5 * np.array(([x-x_pos,y-y_pos])).dot(einv).dot([x-x_pos,y-y_pos]))/( np.sqrt( (2.0*np.pi)**2 * det1) )
    #     # self.phik = self.get_phik(np.vectorize(self.phi))
    #     self.phik = self.get_phik(self.phi)

    # def update_eid(self, mu, sig):
    #     siginv = np.linalg.inv(sig)
    #     detsig = np.linalg.det(sig)
    #     eta = 1.0/( np.sqrt( (2.0*np.pi)**2 * detsig) )
    #     _phi = lambda x, y: exp(-0.5 * np.array(([x-mu[0],y-mu[1]])).dot(siginv).dot([x-mu[0],y-mu[1]]))
    #     normfact, self.samples = monte_carlo(np.vectorize(_phi), self.xlim, n=200, xsamp=self.samples)
    #
    #     phi = lambda x,y: _phi(x,y)/normfact
    #
    #     self.phik = self.get_phik(np.vectorize(phi))
    #     self.phik /= self.phik[0]

    # def update_eid(self, fim, mu, sig, sensor_state):
    #     sigInv = np.linalg.inv(sig)
    #     detSig = np.linalg.det(sig)
    #     eta = 1.0/( np.sqrt( (2.0*np.pi)**2 * detSig) )
    #     pdf = lambda x: exp( - 0.5 * (x - mu).dot(sigInv).dot(x - mu)) * eta
    #     phik = np.zeros(self.coef+1).ravel()
    #     # run through once to get the evaluations out of the way
    #     fim_evaled = [[None]* len(self.search_space)] * len(self.target_space)
    #     for i in range(len(self.search_space)):
    #         robot_state = np.hstack((self.search_space[i], sensor_state[2]))
    #         for j in range(len(self.target_space)):
    #             fim_evaled[i][j] = fim(self.target_space[j], robot_state) *(pdf(self.target_space[j]))
    #
    #     for ii, k in enumerate(self.k_list):
    #         integrand = 0.0
    #         for i in range(len(self.search_space)):
    #             robot_state = np.hstack((self.search_space[i], sensor_state[2]))
    #             eidofx = np.zeros((2,2))
    #             for j in range(len(self.target_space)):
    #                 # eid = fim(self.target_space[j], robot_state) *(pdf(self.target_space[j]))
    #                 # eidofx += eid
    #                 eidofx += fim_evaled[i][j]
    #             eidofx /= float(len(self.target_space))
    #             integrand += (1.0 - np.linalg.det(eidofx/fim_evaled)  ) * self.basis.fk(k, robot_state)
    #
    #         phik[ii] = integrand/float(len(self.search_space))
    #     phik /= phik[0]
    #     self.phik = phik.reshape(self.coef+1)


    def update_eid(self, fim, mu, sig, sensor_state):
        siginv = np.linalg.inv(sig)
        detsig = np.linalg.det(sig)

        eta = 1.0/( np.sqrt( (2.0*np.pi)**2 * detsig) )
        _pdf = lambda x, y: exp(-0.5 * np.array(([x-mu[0],y-mu[1]])).dot(siginv).dot([x-mu[0],y-mu[1]]))
        pdfnorm,_ = monte_carlo(np.vectorize(_pdf), self.xlim, n=200, xsamp=self.samples)
        pdf = lambda x,y: _pdf(x,y)/pdfnorm

        _phi = lambda x,y: np.linalg.det(fim(mu, [x,y,sensor_state[2]]))
        normfact, self.samples = monte_carlo(np.vectorize(_phi), self.xlim, n=200, xsamp=self.samples)
        # phi = lambda x,y: np.amax([_phi(x,y)/normfact, pdf(x,y)])
        # phi = lambda x,y: _phi(x,y)/normfact
        phi = lambda x,y: pdf(x,y)
        self.phik = self.get_phik(np.vectorize(phi))
        self.phik /= self.phik[0]

class EvaderExpectedInformation(object):

    def __init__(self, xlim, coef, basis):
        self.basis = basis
        self.coef = coef
        self.xlim = xlim
        self.k_list = []
        for i in range(self.coef[0]+1):
            for j in range(self.coef[1]+1):
                self.k_list.append([i,j])
        # phi_temp = lambda x,y: np.ones(len(x))
        # phi_temp = lambda x,y: exp( -20*(x-.7)**2)*exp(-20*(y-.7)**2)
        phi_temp = lambda x,y: 1.0
        # normfact = monte_carlo(lambda x,y: phi_temp(x,y), self.xlim, n=200)
        self.normfact = nquad(lambda x,y: phi_temp(x,y), self.xlim)[0]
        self.phi = lambda x,y: phi_temp(x,y)/self.normfact
        self.phik = self.get_phik(self.phi)
        self.samples = None

        target_space = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))
        search_space = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))

        self.target_space = np.c_[target_space[0].ravel(), target_space[1].ravel()]
        self.search_space = np.c_[search_space[0].ravel(), search_space[1].ravel()]

    def get_phik(self, phi):
        phik = np.zeros(self.coef+1).ravel()
        for i,k in enumerate(self.k_list):
            temp_fun = lambda x,y: phi(x,y) * self.basis.fk(k, [x,y])
            phik[i],_ = monte_carlo(temp_fun, self.xlim, n=200)
            # phik[i] = nquad(temp_fun, self.xlim)[0
        return phik.reshape(self.coef+1)

    # def update_eid(self, time):
    #     x_pos = 0.25*cos(0.25 * time) + 0.5
    #     y_pos = 0.25*sin(0.25 * time) + 0.5
    #     # einv = np.linalg.inv(np.diag([0.01,0.01]))
    #     # det1 = np.linalg.det(np.diag([0.01,0.01]))
    #     self.phi = lambda x,y: exp( -20*(x-x_pos)**2)*exp(-20*(y-y_pos)**2)/self.normfact
    #     # self.phi = lambda x,y: 1.0-exp(-0.5 * np.array(([x-x_pos,y-y_pos])).dot(einv).dot([x-x_pos,y-y_pos]))/( np.sqrt( (2.0*np.pi)**2 * det1) )
    #     # self.phik = self.get_phik(np.vectorize(self.phi))
    #     self.phik = self.get_phik(self.phi)

    # def update_eid(self, mu, sig):
    #     siginv = np.linalg.inv(sig)
    #     detsig = np.linalg.det(sig)
    #     eta = 1.0/( np.sqrt( (2.0*np.pi)**2 * detsig) )
    #     _phi = lambda x, y: exp(-0.5 * np.array(([x-mu[0],y-mu[1]])).dot(siginv).dot([x-mu[0],y-mu[1]]))
    #     normfact, self.samples = monte_carlo(np.vectorize(_phi), self.xlim, n=200, xsamp=self.samples)

    #     phi = lambda x,y: 1.0 - _phi(x,y)/normfact

    #     self.phik = self.get_phik(np.vectorize(phi))
        # self.phik /= self.phik[0,0]

    # def update_eid(self, fim, mu, sig, sensor_state):
    #     sigInv = np.linalg.inv(sig)
    #     detSig = np.linalg.det(sig)
    #     eta = 1.0/( np.sqrt( (2.0*np.pi)**2 * detSig) )
    #     pdf = lambda x: exp( - 0.5 * (x - mu).dot(sigInv).dot(x - mu)) * eta
    #     phik = np.zeros(self.coef+1).ravel()
    #     # run through once to get the evaluations out of the way
    #     fim_evaled = [[None]* len(self.search_space)] * len(self.target_space)
    #     for i in range(len(self.search_space)):
    #         robot_state = np.hstack((self.search_space[i], sensor_state[2]))
    #         for j in range(len(self.target_space)):
    #             fim_evaled[i][j] = fim(self.target_space[j], robot_state) *(pdf(self.target_space[j]))
    #
    #     for ii, k in enumerate(self.k_list):
    #         integrand = 0.0
    #         for i in range(len(self.search_space)):
    #             robot_state = np.hstack((self.search_space[i], sensor_state[2]))
    #             eidofx = np.zeros((2,2))
    #             for j in range(len(self.target_space)):
    #                 # eid = fim(self.target_space[j], robot_state) *(pdf(self.target_space[j]))
    #                 # eidofx += eid
    #                 eidofx += fim_evaled[i][j]
    #             eidofx /= float(len(self.target_space))
    #             integrand += (1.0 - np.linalg.det(eidofx)) * self.basis.fk(k, robot_state)
    #
    #         phik[ii] = integrand/float(len(self.search_space))
    #     phik /= phik[0]
    #     self.phik = phik.reshape(self.coef+1)


    def update_eid(self, fim, mu, sig, sensor_state):
        siginv = np.linalg.inv(sig)
        detsig = np.linalg.det(sig)

        eta = 1.0/( np.sqrt( (2.0*np.pi)**2 * detsig) )
        _pdf = lambda x, y: exp(-0.5 * np.array(([x-mu[0],y-mu[1]])).dot(siginv).dot([x-mu[0],y-mu[1]]))
        pdfnorm,_ = monte_carlo(np.vectorize(_pdf), self.xlim, n=200, xsamp=self.samples)
        pdf = lambda x,y: _pdf(x,y)/pdfnorm

        _phi = lambda x,y: np.linalg.det(fim(mu, [x,y,sensor_state[2]]) )
        normfact, self.samples = monte_carlo(np.vectorize(_phi), self.xlim, n=200, xsamp=self.samples)
        phi = lambda x,y: 1.0-_phi(x,y)/normfact
        self.phik = self.get_phik(np.vectorize(phi))
        self.phik /= self.phik[0,0]
