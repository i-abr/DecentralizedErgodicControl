import numpy as np

class BarrierFunction(object):

    def __init__(self, b_lim, barr_weight=10000.0, b_buff=0.01):

        self.b_lim = np.copy(b_lim)
        self.barr_weight = barr_weight
        self.b_lim[0][0] = self.b_lim[0][0] + b_buff
        self.b_lim[0][1] = self.b_lim[0][1] - b_buff
        self.b_lim[1][0] = self.b_lim[1][0] + b_buff
        self.b_lim[1][1] = self.b_lim[1][1] - b_buff


    def barr(self,x):
        barr_temp = 0.
        if x[0] >= self.b_lim[0][1]:
            barr_temp += self.barr_weight*(x[0] - (self.b_lim[0][1]))**2
        elif x[0] <=  self.b_lim[0][0]+0.01:
            barr_temp += self.barr_weight*(x[0] - (self.b_lim[0][0]))**2

        if x[1] >= self.b_lim[1][1]:
            barr_temp += self.barr_weight*(x[1] - (self.b_lim[1][1]))**2
        elif x[1] <= self.b_lim[1][0]:
            barr_temp += self.barr_weight*(x[1] - (self.b_lim[1][0]))**2
        return barr_temp
    def dbarr(self, x):
        dbarr_temp = np.zeros(len(x))
        if x[0] >= self.b_lim[0][1]:
            dbarr_temp[0] += 2*self.barr_weight*(x[0] - (self.b_lim[0][1]))
        elif x[0] <=  self.b_lim[0][0]:
            dbarr_temp[0] += 2*self.barr_weight*(x[0] - (self.b_lim[0][0]))

        if x[1] >= self.b_lim[1][1]:
            dbarr_temp[1] += 2*self.barr_weight*(x[1] - (self.b_lim[1][1]))
        elif x[1] <= self.b_lim[1][0]:
            dbarr_temp[1] += 2*self.barr_weight*(x[1] - (self.b_lim[1][0]))
        return dbarr_temp
