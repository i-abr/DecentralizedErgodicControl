import numpy as np
from terrain.city_block import g_city_terrain
from agent.agent import Agent
from target.target import Target
import matplotlib.pyplot as plt
from graph import Graph
# from ergodic.city_block import g_city_terrain
import sys, os

class World(object):

    def __init__(self, time_step):
        self.time_step = time_step
        self.no_agents = 3
        self.no_targets = 1
        self.agents = [Agent(self.time_step) for i in range(self.no_agents)]
        self.targets = [Target(self.time_step) for i in range(self.no_targets)]
        self.ck_comm_link, self.target_comm_link = self._makeCommunicationLink()

    def _makeCommunicationLink(self):
        degree = [2]*self.no_agents
        edges = []
        for i in range(self.no_agents-1):
            edges.append([i+1, i+2])
            edges.append([i+2, i+1])
            if i == 0:
                edges.append([1, self.no_agents])
                edges.append([self.no_agents, 1])
        no_param = np.product(self.agents[0].settings.coef+1)
        no_target_param = 2
        return Graph(self.no_agents, edges, degree, no_param), Graph(self.no_agents, edges, degree, no_target_param)

    def world_step(self): # Step the world forward

        ck_stack = None
        mean_stack = None

        for target in self.targets: # step the targets forward in time
            target.step() # step the target
            print('Target state : {}'.format(target.state))

        for i,agent in enumerate(self.agents): # For each agent...

            for target in self.targets: # sense each target
                if not g_city_terrain.isInBlock(target.state):
                    yk = agent.sensor.h(target.state, agent.state[0:3])
                else:
                    yk = None
                agent.updateBelief(yk) # update the PHIK value TODO: make this into a class for phik
            agent.step() # step the agent forward in time

            if ck_stack is None: # collect the ck values
                ck_stack = agent.controller.ck.cki.copy()
            else:
                ck_stack = np.hstack(( ck_stack, agent.controller.ck.cki.copy() ))

            if mean_stack is None: # also collect the mean values from the bayes filter
                mean_stack = agent.kalman_filter.mu
            else:
                mean_stack = np.hstack((mean_stack, agent.kalman_filter.mu))

            print('Agent {} state : {}, Target Belief : {}'.format(i, agent.state[0:3], agent.kalman_filter.mu)) # print information

        ###### communication loop #####
        ck_update = self.ck_comm_link.update_consensus(ck_stack)
        mean_update = self.target_comm_link.update_consensus(mean_stack)

        for i,agent in enumerate(self.agents):
            agent.controller.ck.cki = ck_update[i].copy()
            agent.kalman_filter.mu = mean_update[i].copy()

    def draw(self, ax):
        g_city_terrain.plotCity(ax)
        for agent in self.agents:
            agent.plotTrajectory(ax)
        for target in self.targets:
            target.plotTrajectory(ax)

    def save_agent_data(self, dirPath):
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

        for i, target in enumerate(self.targets):
            filePath = dirPath + 'target{}/'.format(i)
            try:
                os.makedirs(filePath)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            target.save_data(dirPath + 'target{}/'.format(i))
def main():
    world = World(0.05) # instantiate a world with time step 0.05
    tf = int(20/0.05)
    tic = 0
    toc = 2
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    plt.ion()
    for i in range(tf):
        world.world_step()
        if tic > toc:
            # plt.clf()
            ax.clear()
            world.draw(ax)
            plt.draw()
            plt.pause(0.01)
            tic = 0

        tic += 1
        print('Step : {} out of {}'.format(i, tf))
        # os.system('cls' if os.name == 'nt' else 'clear')
    filePath = 'target_test_data_for_ros/'
    world.save_agent_data(filePath)
    g_city_terrain.save_data(filePath)
if __name__ == '__main__':
    main()
