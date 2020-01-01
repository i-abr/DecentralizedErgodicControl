import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class Block(object):

    def __init__(self):
        position = [np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]
        self.position = np.array(position) # position of the city block
        self.length = np.random.uniform(0.05,0.09)
        self.order = 4 # this needs to be an even number
        # self.scale = [np.random.uniform(0.5,2.), np.random.uniform(0.5,2.)]
        self.scale = [1.0,1.0]
        self.slope = 4000.0

    def isInBlock(self, x): # check to see if a point is in the block
        dx = x[0] - self.position[0]
        dx /= self.scale[0]
        dy = x[1] - self.position[1]
        dy /= self.scale[1]
        dist = dx**self.order + dy**self.order - self.length**self.order
        if dist < 0:
            return True
        else:
            return False

    def cost(self, x):
        dx = x[0] - self.position[0]
        dx /= self.scale[0]
        dy = x[1] - self.position[1]
        dy /= self.scale[1]
        dist = dx**self.order + dy**self.order - self.length**self.order
        # if dist < 0:
        #     return 1.0
        # else:
        #     return 0.0
        return np.exp(-self.slope * dist)

    def dcost(self, x):
        _ddist = np.zeros(x.shape)
        dx = x[0] - self.position[0]
        dx /= self.scale[0]
        dy = x[1] - self.position[1]
        dy /= self.scale[1]
        _ddist[0] = self.order * (dx ** (self.order-1)) / self.scale[0]
        _ddist[1] = self.order * (dy ** (self.order-1)) / self.scale[1]
        dist = dx**self.order + dy**self.order - self.length**self.order
        val =  -self.slope * np.exp(-self.slope * dist) * _ddist
        return val

class CityBlocks(object):

    def __init__(self, num_blocks=20, visualize_city_block=True):
        self.city_blocks = []
        print('Generating City Blocks...')
        while len(self.city_blocks) < num_blocks:
            temp_block = Block()
            if len(self.city_blocks) == 0:
                self.city_blocks.append(temp_block)
            else:
                flag = True
                for block in self.city_blocks:
                    # print np.linalg.norm(temp_block.position - block.position) - (temp_block.length + block.length)
                    if np.linalg.norm(temp_block.position - block.position) - (temp_block.length + block.length) < 0.15:
                        flag = False
                        break
                if flag is True:
                    self.city_blocks.append(temp_block)
                    print('Added city block no.{} out of {}'.format(len(self.city_blocks), num_blocks))
        # if visualize_city_block is True:
        #     self.plotCity()

    def cost(self, x):
        _cost = 0.0
        for block in self.city_blocks:
            _cost = np.amax([_cost, block.cost(x)])
        return _cost

    def dcost(self, x):
        _dcost = np.zeros(x.shape)
        temp_cost = 0.0
        block_no = 0
        for i,block in enumerate(self.city_blocks):
            if temp_cost < block.cost(x):
                temp_cost = block.cost(x)
                block_no = i
        return self.city_blocks[block_no].dcost(x)
        # temp_cost = block.dcost(x)
        # _dcost = np.amax(np.c_[_dcost, temp_cost], axis=1)
        # for i in range(len(_dcost)):
        #     _dcost[i] = np.amax([_dcost[i], temp_cost[i]])

        return _dcost

    def isInBlock(self, x):

        for block in self.city_blocks:
            if block.isInBlock(x):
                return True # if the point is in the block then return true and call it a day
                break
        return False # if nothing is in the block then return false

    def plotCity(self, ax):
        X,Y = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
        grid = np.c_[X.ravel(), Y.ravel()]
        cost_map = -np.inf * np.ones(grid.shape[0])
        for block in self.city_blocks:
            ax.add_patch(
                patches.Rectangle(
                    block.position - block.length,
                    2*block.length,
                    2*block.length
                )
            )

            block_cost = np.array(map(block.cost, grid))
            temp_array = np.c_[cost_map, block_cost]
            cost_map = np.amax(temp_array, axis=1)
            for i in range(len(temp_array)):
                cost_map[i] = np.amax(temp_array[i])

        # print cost_map.shape.reshape(X.shape)
        cmap = plt.get_cmap('Greys')
        ax.imshow(cost_map.reshape(X.shape), extent=(0,1,0,1), origin='lower', cmap=cmap)
        # plt.show()

    def save_data(self, filePath=''):
        data = None
        for block in self.city_blocks:
            if data is None:
                data = np.hstack((block.position.copy(), block.length))
            else:
                input_data = np.hstack((block.position.copy(), block.length))
                data = np.vstack((data, input_data))
        np.savetxt(filePath + 'city_blocks.csv', data)

g_city_terrain = CityBlocks(4) # add this to the file

if __name__ == '__main__':
    city = CityBlocks(20, visualize_city_block=True )
    # city.plotCity()
    # plt.show()

    # num_blocks = 20
    # city_blocks = []
    # while len(city_blocks) < num_blocks:
    #     temp_block = Block()
    #     if len(city_blocks) == 0:
    #         city_blocks.append(temp_block)
    #     else:
    #         flag = True
    #         for block in city_blocks:
    #             # print np.linalg.norm(temp_block.position - block.position) - (temp_block.length + block.length)
    #             if np.linalg.norm(temp_block.position - block.position) - (temp_block.length + block.length) < 0.2:
    #                 flag = False
    #                 break
    #         if flag is True:
    #             print "Added"
    #             city_blocks.append(temp_block)
    # X,Y = np.meshgrid(np.linspace(0,1,60), np.linspace(0,1,60))
    # grid = np.c_[X.ravel(), Y.ravel()]
    # cost_mapx = -np.inf * np.ones(grid.shape[0])
    # cost_mapy = -np.inf * np.ones(grid.shape[0])
    # import matplotlib.pyplot as plt
    # for block in city_blocks:
    #     block_cost = np.array(map(block.dcost, grid))
    #     temp_arrayx = np.c_[cost_mapx, block_cost[:,0]]
    #     temp_arrayy = np.c_[cost_mapy, block_cost[:,1]]
    #     for i in range(len(temp_arrayx)):
    #         cost_mapx[i] = np.amax(temp_arrayx[i])
    #         cost_mapy[i] = np.amax(temp_arrayy[i])
    #
    # # print cost_map.shape.reshape(X.shape)
    # print np.c_[cost_mapx, cost_mapy].shape
    # cost_map = np.amax(np.c_[cost_mapx, cost_mapy])
    # print cost_map.shape
    # plt.figure(1)
    # plt.imshow(cost_map.reshape(X.shape), extent=(0,1,0,1), origin='lower')
    # # plt.figure(2)
    # # plt.imshow(cost_mapy.reshape(X.shape), extent=(0,1,0,1), origin='lower')
    #
    # plt.figure(3)
    # X,Y = np.meshgrid(np.linspace(0,1), np.linspace(0,1))
    # grid = np.c_[X.ravel(), Y.ravel()]
    # cost_map = -np.inf * np.ones(grid.shape[0])
    # import matplotlib.pyplot as plt
    # for block in city_blocks:
    #     block_cost = np.array(map(block.cost, grid))
    #     temp_array = np.c_[cost_map, block_cost]
    #     for i in range(len(temp_array)):
    #         cost_map[i] = np.amax(temp_array[i])
    #
    # # print cost_map.shape.reshape(X.shape)
    # plt.imshow(cost_map.reshape(X.shape), extent=(0,1,0,1), origin='lower')
    #
    # plt.show()
