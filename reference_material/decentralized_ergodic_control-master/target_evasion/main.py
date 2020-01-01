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
        self.no_agents = 1
        self.no_targets = 3
        self.agents = [Agent(self.time_step) for i in range(self.no_agents)]
        self.targets = [Target(self.time_step) for i in range(self.no_targets)]
        self.ck_comm_link = self._makeCommunicationLink()

    def _makeCommunicationLink(self):
        degree = [2]*self.no_targets
        edges = []
        for i in range(self.no_targets-1):
            edges.append([i+1, i+2])
            edges.append([i+2, i+1])
            if i == 0:
                edges.append([1, self.no_targets])
                edges.append([self.no_targets, 1])
        no_param = np.product(self.targets[0].settings.coef+1)
        no_target_param = 2
        return Graph(self.no_targets, edges, degree, no_param)

    def world_step(self): # Step the world forward

        ck_stack = None
        mean_stack = None

        for target in self.targets: # step the targets forward in time
            for agent in self.agents: # TODO: add capabilities to differentiate other agents for kalman filter
                target.updateBelief(agent.state) # update agent
            print('Target state : {}'.format(target.state))

            if ck_stack is None: # collect the ck values
                ck_stack = target.controller.getCKFromDefaultControl(target.state).copy()
            else:
                ck_stack = np.hstack(( ck_stack, target.controller.getCKFromDefaultControl(target.state).copy() ))
        for i,agent in enumerate(self.agents): # For each agent...

            for target_no, target in enumerate(self.targets):
                if not g_city_terrain.isInBlock(target.state):
                    yk = agent.sensor.h(target.state, agent.state[0:3])
                    if target_no == 0 and yk is not None:
                        yk += np.random.normal(0.0, 0.2, size=yk.shape)
                else:
                    yk = None
                agent.updateBelief(yk) # update the PHIK value TODO: make this into a class for phik
            agent.step() # step the agent forward in time

            print('Agent {} state : {}, Target Belief : {}'.format(i, agent.state[0:3], agent.kalman_filter.mu)) # print information

        ###### communication loop #####
        ck_update = self.ck_comm_link.update_consensus(ck_stack)
        print ck_update
        for i,target in enumerate(self.targets):
            # target.controller.ck.cki = ck_update[i].copy()
            target.step(ck_update[i].copy()) # step the target


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
    filePath = 'target_evasion_data/'
    world.save_agent_data(filePath)
    g_city_terrain.save_data(filePath)
if __name__ == '__main__':
    main()
