import numpy as np
import matplotlib.pyplot as plt

class BoundingBox(object):


    def __init__(self, x1, x2):

        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.mean = (self.x1 + self.x2 )/2.0


    def __call__(self, x):
        rand_ = 0*np.random.uniform(-0.01,0.01,size=(2,))
        if x[0] >= self.x1[0]+rand_[0] and x[0] <= self.x2[0]+rand_[0]:
            if x[1] >= self.x1[1]+rand_[1] and x[1] <= self.x2[1]+rand_[1]:
                return np.linalg.norm(x[0:2] - self.mean) + 1.0
        return 0.0

    def plot_box(self, ax=None):
        dx = self.x2 - self.x1
        p1 = self.x1.copy()
        p2 = self.x1.copy()
        p2[1] += dx[1]
        p3 = self.x2.copy()
        p4 = self.x1.copy()
        p4[0] += dx[0]
        if ax is None:
            plt.plot([p1[0], p2[0], p3[0], p4[0], p1[0]]
                        ,[p1[1], p2[1], p3[1], p4[1], p1[1]], linewidth=2, color='k')
        else:
            ax.plot([p1[0], p2[0], p3[0], p4[0], p1[0]]
                        ,[p1[1], p2[1], p3[1], p4[1], p1[1]], linewidth=1, color='k', zorder=5)
class FloorPlan(object):

    def __init__(self):

        bounding_regions = [
                ([0.2, 0.2], [0.4, 0.4]),
                ([0.6, 0.7], [0.8, 1.0]),
                ([0.7, 0.2], [1.0, 0.4])
        ]

        self.bounding_boxes = []
        for bounding_region in bounding_regions:

            self.bounding_boxes.append(
                    BoundingBox(*bounding_region)
            )
        self.X_,self.Y_ = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
        samples = np.c_[self.X_.ravel(), self.Y_.ravel()]
        eval_samples = np.array([self.get_penality(sample) for sample in samples])
        self.eval_samples = eval_samples.reshape(self.X_.shape)



    def plot_ground(self, ax=None):
        for bounding_box in self.bounding_boxes:
            bounding_box.plot_box(ax)
        # plt.contourf(self.X_, self.Y_, self.eval_samples)


    def get_penality(self, x):
        penalty = 0.0
        for bounding_box in self.bounding_boxes:
            penalty += bounding_box(x)
        return penalty
    def first_order_derivative(self, x):
        dx = 0.01
        dpdx = np.zeros(x.shape)
        for i in range(2):
            delta = np.zeros(x.shape)
            delta[i] = dx
            dxp = self.get_penality(x + delta)
            dxn = self.get_penality(x - delta)
            dpdx[i] = (dxp - dxn) / (2 * dx)
        return dpdx
if __name__ == '__main__':
    fp = FloorPlan()
    X,Y = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
    samples = np.c_[X.ravel(), Y.ravel()]
    eval_samples = np.array([fp.get_penality(sample) for sample in samples])
    deriv_eval_samples = np.array([
                        np.linalg.norm(fp.first_order_derivative(sample))
                        for sample in samples])
    eval_samples = eval_samples.reshape(X.shape)
    deriv_eval_samples = deriv_eval_samples.reshape(X.shape)

    plt.imshow(eval_samples, origin='lower', extent=(0,1,0,1))
    plt.colorbar()
    plt.show()
