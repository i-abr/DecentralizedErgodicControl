import numpy as np
from settings import (coef, xlim)
from odeint2 import monte_carlo
import basis
from scipy.integrate import nquad, trapz

def get_phik(phi):
    phik = np.zeros(coef+1)
    for i in range(coef[0]+1):
        for j in range(coef[1]+1):
            temp_fun = lambda x,y: phi(x,y) * basis.fk([i,j], x,y)
            phik[i,j] = monte_carlo(temp_fun, xlim, n=200)
            # phik[i,j] = nquad(temp_fun, xlim)[0]
    return phik

def get_ck(x):
    N = len(x)
    ck = np.zeros(coef+1)
    for i in range(coef[0] + 1):
        for j in range(coef[1] + 1):
            _fk = [basis.fk([i,j], x[k][0],x[k][1]) for k in range(N)]
            ck[i,j] = np.sum(_fk)/float(N)#trapz(_fk, x=t)
            # ck[i,j] = trapz(_fk, x=t)/t[-1]
    return ck
