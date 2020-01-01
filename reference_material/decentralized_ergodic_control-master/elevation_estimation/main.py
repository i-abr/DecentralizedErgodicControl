from agent.agent import Agent
from terrain.terrain import g_terrain
import matplotlib.pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from graph import Graph

class World(object):

    def __init__(self):
        self.time_step = 0.05
        self.no_agents = 3
        self.agents = [Agent(self.time_step) for i in range(self.no_agents)]
        self.ck_comm_link = self.makeCommunicationLink()

    def saveData(self, dirPath = ''):
        import os
        import errno
        for i,agent in enumerate(self.agents):
            filePath = dirPath + 'agent{}/'.format(i)
            try:
                os.makedirs(filePath)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            agent.save_data(dirPath + 'agent{}/'.format(i))

    def makeCommunicationLink(self):
        degree = [2]*self.no_agents
        edges = []
        for i in range(self.no_agents-1):
            edges.append([i+1, i+2])
            edges.append([i+2, i+1])
            if i == 0:
                edges.append([1, self.no_agents])
                edges.append([self.no_agents, 1])
        no_param = np.product(self.agents[0].settings.coef+1)
        return Graph(self.no_agents, edges, degree, no_param)

    def world_step(self):
        ck_stack = None
        ck_prior = None
        for agent_no, agent in enumerate(self.agents):
            yk = g_terrain.getElevationAtX(agent.state)
            agent.updateBelief(yk)
            if ck_stack is None:
                ck_stack = agent.controller.getCKFromDefaultControl(agent.state)
                ck_prior = agent.controller.ck.cki.copy()
            else:
                ck_stack = np.hstack(( ck_stack, agent.controller.getCKFromDefaultControl(agent.state)))
                ck_prior = np.hstack(( ck_prior, agent.controller.ck.cki.copy()))

        #### comm loop #####
        ck_update = self.ck_comm_link.update_consensus(ck_stack)
        ck_prior_update = self.ck_comm_link.update_consensus(ck_prior)
        for agent_no, agent in enumerate(self.agents):
            agent.controller.ck.cki = ck_prior_update[agent_no].copy()
            agent.step(ck_update[agent_no].copy())

    def draw(self):
        X = []
        y = []
        for agent in self.agents:
            X = X + agent.X
            y = y + agent.y
        # print('Calculating GP...')
        self.agents[0].estimator.compute(X, y)
        # print('Done calculating!')
        print('Plotting!!!!')

        plt.figure(1)
        X,Y = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
        # height = gp.predict(np.c_[X.ravel(), Y.ravel()])
        height = np.array(map(self.agents[0].estimator, np.c_[X.ravel(), Y.ravel()]))
        plt.imshow(height.reshape(X.shape), extent=(0,1,0,1), origin='lower')

        for agent in self.agents:
            agent.plotTrajectory()

        plt.figure(2)
        true_height = g_terrain.plotElevation()

        error = 0.0
        for elem1, elem2 in zip(height, true_height):
            error += (elem1 - elem2)**2
        print error
        plt.show()

def main():

    world = World()
    tf = int(40/0.05)
    tic = 0
    toc = 5

    for i in range(tf):
        world.world_step()
        print('Iteration {} out of {}'.format(i, tf))

    world.saveData('data2/') # save the data
    world.draw()


if __name__ == '__main__':
    main()
