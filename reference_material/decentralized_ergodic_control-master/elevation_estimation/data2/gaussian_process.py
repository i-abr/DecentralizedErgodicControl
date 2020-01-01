import numpy as np

class RBF(object):

    def __init__(self, variance=1, sigma=0.1):
        self.sigma = sigma
        self.variance = variance

    def __call__(self, x1, x2):

        return self.variance * np.exp( -0.5 * self.sigma * np.linalg.norm(x1-x2) )

class GaussianProcess(object):

    def __init__(self, kernel=RBF(), X=None, y=None, variance=0.1):

        self.kernel = kernel
        self.is_calculated = False
        self.variance = variance
        if X is not None:
            self.compute(X, y)

    def compute(self, X, y):
        self.X = X
        self.y = np.array(y)
        self.G = np.zeros((len(self.X), len(self.X)))
        self.n = len(self.X)
        for i in range( len(self.X) ):
            for j in range( len(self.X) ):
                self.G[i,j] = self.kernel(self.X[i], self.X[j])
        self.Ginv = np.linalg.inv(self.G + self.variance**2 * np.identity(self.n) )
        self.w = self.y.T.dot(self.Ginv)
        self.is_calculated = True

    def __call__(self, x):
        """ function call returns the mean and variance of the GP """
        if self.is_calculated:
            k = np.array(map( lambda xi: self.kernel(xi, x), self.X))
            # var = self.kernel(x, x) - k.dot(self.Ginv).dot(k)
            ystar = self.w.dot(k)
            return ystar

        else:
            print "Has not initialized!!!"
            return None
