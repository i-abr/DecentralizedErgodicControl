import numpy as np

def rk4Step(f, x0, dt, *args):
    k1 = f(x0, *args) * dt
    k2 = f(x0 + k1/2.0, *args) * dt
    k3 = f(x0 + k2/2.0, *args) * dt
    k4 = f(x0 + k3, *args) * dt
    return x0 + (k1 + 2.0 * (k2 + k3) + k4)/6.0

def eulerStep(f, x0, dt, *args):
    return x0 + f(x0, *args) * dt


def rk4(f, x0, t0, tf, dt, *args):
    '''
    rk4 integrator scheme
    '''
    for i in t:
        if i == t[0]:
            x = np.array(x0)
        else:
            x = np.vstack((x, x0))
        k1 = f(x0, i, *args)*dt
        k2 = f(x0 + k1/2, i+dt/2, *args)*dt
        k3 = f(x0 + k2/2, i+dt/2, *args)*dt
        k4 = f(x0+k3, i+dt, *args)*dt
        x0 = x0 + (k1 + 2*(k2 + k3) + k4)/6
    return x

def euler(f, x0, t, dt, *args):
    '''
    Euler integrator scheme
    '''
    for i in t:
        if i == t[0]:
            x = np.array(x0)
        else:
            x = np.vstack((x, x0))
        x0 = x0 + f(x0, i, *args)*dt # tadah??
    return x

def monte_carlo(f, xlim, n=200, xsamp=None,*args):
    '''
    monte carlo integration scheme to speed up dumb things
    '''
    if xsamp is None:
        xsamp = [np.random.uniform(low=i[0], high=i[1], size=n) for i in xlim]
    # integrand = map(f, *xsamp)
    return np.sum(f(*xsamp))/(n), xsamp
    # return np.sum(integrand)/(n)
