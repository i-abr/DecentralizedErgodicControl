import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

def pdf(mu, sigma, _eval):
    k = len(mean)
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    Z = 1.0/np.sqrt(det_sigma * (2.0 * pi)**k)
    delta = mean - _eval
    return np.exp(-0.5 * (delta).dot(inv_sigma).dot(delta))

X, Y = np.meshgrid(np.linspace(0,1,60), np.linspace(0,1,60))

target_data = [np.loadtxt('agent0/target_mean_data.csv'), np.loadtxt('agent0/target_covar_data.csv')]


for i in range(len(target_data[0])):
    mean = target_data[0][i]
    covar = target_data[1][i].reshape((2,2))
    data = np.array(map(lambda x: pdf(mean, covar, x), np.c_[X.ravel(),Y.ravel()]))
    data = data.reshape(X.shape)
    np.save('agent0_pdf_data/pdf_data_at_{}.npy'.format(i), data)
