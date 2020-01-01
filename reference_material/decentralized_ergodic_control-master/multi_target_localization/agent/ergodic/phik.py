import numpy as np
from numpy import exp, sqrt, pi, cos, sin
import matplotlib.pyplot as plt
import time
from integrators import monte_carlo
from scipy.integrate import nquad
from extended_kalman_filter import EKF, DiffusionProcess, BearingOnlySensor
from terrain.city_block import g_city_terrain
from sklearn.preprocessing import normalize

class Phik(object):

    def __init__(self, settings, no_targets):
        self.basis = settings.basis
        self.coef = settings.coef
        self.xlim = settings.xlim
        self.k_list = settings.k_list

        # stock up on those EKFSSSS
        self.ekfs = [
            EKF(DiffusionProcess(settings.time_step), BearingOnlySensor())
                for targets in range(no_targets)
        ]

        phi_temp = lambda x,y: 1.0
        # normfact = monte_carlo(lambda x,y: phi_temp(x,y), self.xlim, n=200)
        self.normfact = nquad(lambda x,y: phi_temp(x,y), self.xlim)[0]
        self.phi = lambda x,y: phi_temp(x,y)/self.normfact
        self.phik = self.get_phik(self.phi)
        self.samples = None
        self.__X, self.__Y = np.meshgrid(np.linspace(0,1,20), np.linspace(0,1,20))
        self.spatial_samples = np.c_[self.__X.ravel(), self.__Y.ravel()]
        self.__N_samples = self.spatial_samples.shape[0]
        self.basis_func_eval = []
        for sample in self.spatial_samples:
            basis_sample_eval = []
            for i,k in enumerate(self.k_list):
                basis_sample_eval.append(self.basis.fk(k, sample))
            self.basis_func_eval.append(basis_sample_eval)

        self.basis_func_eval = np.array(self.basis_func_eval) # make a numpy array

    def get_phik(self, phi):
        phik = np.zeros(self.coef+1).ravel()
        for i,k in enumerate(self.k_list):
            temp_fun = lambda x,y: phi(x,y) * self.basis.fk(k, [x,y])
            phik[i],_ = monte_carlo(temp_fun, self.xlim, n=200)
            # phik[i] = nquad(temp_fun, self.xlim)[0
        phik /= phik[0]
        return phik

    def update_eid(self, agent_state, targets):

        ekf_pdf_eval = None
        for ekf, target in zip(self.ekfs, targets):
            if not g_city_terrain.isInBlock(target.state):
                yk = ekf.sensor_model.h(target.state, agent_state[0:3])
                if yk is not None:
                    yk += np.random.normal([0.]*2, [0.001,0.001])
            else:
                yk = None
            ekf.update(yk, agent_state[0:3])
            print('ekf mean : {}, true state : {}'.format(ekf.mu, target.state[0:2]))

            evaluations = np.array(map(ekf.pdf, self.spatial_samples))
            if ekf_pdf_eval is None:
                ekf_pdf_eval = evaluations
            else:
                ekf_pdf_eval += evaluations # np.amax( np.c_[ekf_pdf_eval,evaluations] , axis=1 )

        # ekf_pdf_eval = np.amax(ekf_pdf_eval, axis=0)
        ekf_pdf_eval /= np.sum(ekf_pdf_eval)


        self.phik = np.dot(ekf_pdf_eval, self.basis_func_eval)
        # siginv = np.linalg.inv(sig)
        # detsig = np.linalg.det(sig)
        #
        # eta = 1.0/( np.sqrt( (2.0*np.pi)**2 * detsig) )
        # _pdf = lambda x, y: exp(-0.5 * np.array(([x-mu[0],y-mu[1]])).dot(siginv).dot([x-mu[0],y-mu[1]]))
        # pdfnorm,_ = monte_carlo(np.vectorize(_pdf), self.xlim, n=200, xsamp=self.samples)
        # pdf = lambda x,y: _pdf(x,y)/pdfnorm
        #
        # _phi = lambda x,y: np.linalg.det(fim(mu, [x,y,sensor_state[2]]))
        # normfact, self.samples = monte_carlo(np.vectorize(_phi), self.xlim, n=200, xsamp=self.samples)
        # # phi = lambda x,y: np.amax([_phi(x,y)/normfact, pdf(x,y)])
        # # phi = lambda x,y: _phi(x,y)/normfact
        # phi = lambda x,y: pdf(x,y)
        # self.phik = self.get_phik(np.vectorize(phi))
        # self.phik /= self.phik[0]





    ##### Older reference functions #####
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
