import numpy as np
from scipy.interpolate import interp1d


class Target(object):


    def __init__(self, random_traj=False):
        points = 14
        if random_traj:
            path =  np.random.uniform(0.2,0.8, size=(2,points))
        # if tnum == 1:
        #     path = np.array([
        #             [0.24862725, 0.37101595, 0.49304188, 0.6683328,  0.49970905, 0.23357482, 0.32086709, 0.32590007, 0.46791224, 0.47702624],
        #             [0.65146308, 0.51689872, 0.37392194, 0.23548558, 0.31835174, 0.56316766, 0.78273506, 0.6096532,  0.77999845, 0.52396451]
        #             ])
        # else:
        #     path = np.array([
        #             [0.34862725, 0.47101595, 0.59304188, 0.7683328,  0.59970905, 0.33357482, 0.42086709, 0.42590007, 0.56791224, 0.57702624],
        #             [0.75146308, 0.61689872, 0.47392194, 0.33548558, 0.41835174, 0.66316766, 0.68273506, 0.5096532,  0.67999845, 0.72396451]
        #             ])
        else:
            path = np.array([
            [0.25, 0.2, 0.3, 0.27, 0.5, 0.6, 0.75, 0.65, 0.7, 0.75, 0.6, 0.5, 0.3, 0.2],
            [0.8, 0.75, 0.75, 0.6, 0.6, 0.5, 0.4, 0.4, 0.3, 0.25, 0.2, 0.3, 0.2, 0.3]
            ])
        random_point = np.random.uniform(0.2,0.8,size=(2,))
        path = np.ones((points,2))
        for i in range(len(path)):
            path[i] = random_point
        path = path.T
        print(path)

        time = np.linspace(0,36,points)
        new_time = np.linspace(0,36,200)
        parametrized_path = interp1d(time, path, kind='cubic')
        self.trajectory = lambda t: parametrized_path(t)
        self.state = self.trajectory(0.0)
        self.past_path = np.array([
            self.trajectory(ti)
            for ti in new_time
        ])

    def update_position(self, time):
        self.state = self.trajectory(time)
        # self.past_path = np.vstack((self.past_path, self.state))
        return self.state.copy()

    def get_position(self):
        return self.state.copy()

if __name__ == '__main__':
    target = Target()
    import matplotlib.pyplot as plt
    time = np.linspace(0,15,200)
    path = np.array([target.trajectory(ti) for ti in time])
    plt.plot(path[:,0], path[:,1])
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
