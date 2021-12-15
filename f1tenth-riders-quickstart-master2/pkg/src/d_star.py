"""

D* grid planning

author: Nirnay Roy

See Wikipedia article (https://en.wikipedia.org/wiki/D*)

"""
import math
import cv2
import numpy as np

from sys import maxsize

import matplotlib.pyplot as plt

show_animation = True


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."
        self.t = "new"  # tag for state
        self.h = 0
        self.k = 0

    def cost(self, state):
        if self.state == "#" or state.state == "#":
            return maxsize

        return math.sqrt(math.pow((self.x - state.x), 2) +
                         math.pow((self.y - state.y), 2))

    def set_state(self, state):
        """
        .: new
        #: obstacle
        e: oparent of current state
        *: closed state
        s: current state
        """
        if state not in ["s", ".", "#", "e", "*"]:
            return
        self.state = state


class Map:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    def get_neighbors(self, state):
        state_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if state.x + i < 0 or state.x + i >= self.row:
                    continue
                if state.y + j < 0 or state.y + j >= self.col:
                    continue
                state_list.append(self.map[state.x + i][state.y + j])
        return state_list

    def set_obstacle(self, point_list):
        for x, y in point_list:
            if x < 0 or x >= self.row or y < 0 or y >= self.col:
                continue

            self.map[x][y].set_state("#")


class Dstar:
    def __init__(self, maps):
        self.map = maps
        self.open_list = set()

    def process_state(self):
        x = self.min_state()

        if x is None:
            return -1

        k_old = self.get_kmin()
        self.remove(x)

        if k_old < x.h:
            for y in self.map.get_neighbors(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
        elif k_old == x.h:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y) \
                        or y.parent != x and y.h > x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                else:
                    if y.parent != x and y.h > x.h + x.cost(y):
                        self.insert(y, x.h)
                    else:
                        if y.parent != x and x.h > y.h + x.cost(y) \
                                and y.t == "close" and y.h > k_old:
                            self.insert(y, y.h)
        return self.get_kmin()

    def min_state(self):
        if not self.open_list:
            return None
        min_state = min(self.open_list, key=lambda x: x.k)
        return min_state

    def get_kmin(self):
        if not self.open_list:
            return -1
        k_min = min([x.k for x in self.open_list])
        return k_min

    def insert(self, state, h_new):
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)

    def remove(self, state):
        if state.t == "open":
            state.t = "close"
        self.open_list.remove(state)

    def modify_cost(self, x):
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))

    def run(self, start, end):

        rx = []
        ry = []

        self.open_list.add(end)

        while True:
            self.process_state()
            if start.t == "close":
                break

        start.set_state("s")
        s = start
        s = s.parent
        s.set_state("e")
        tmp = start

        while tmp != end:
            tmp.set_state("*")
            rx.append(tmp.x)
            ry.append(tmp.y)
            if show_animation:
                plt.plot(rx, ry, "-k")
                plt.pause(0.01)
            if tmp.parent.state == "#":
                self.modify(tmp)
                continue
            tmp = tmp.parent
        tmp.set_state("e")

        return rx, ry

    def modify(self, state):
        self.modify_cost(state)
        while True:
            k_min = self.process_state()
            if k_min >= state.h:
                break

def Img2PltCoordinate(img_coordinate):
    x_plt = img_coordinate[1]
    y_plt = 2000 - img_coordinate[0]
    return [x_plt, y_plt]

def main():
    kernel = np.ones((3, 3), np.uint8)
    
    map = cv2.imread('pkg/maps/ROBOT_NAVIGATION.png',  cv2.IMREAD_GRAYSCALE)
    map = cv2.erode(map, kernel, iterations=12)
    # cv2.imshow('eroded map', map)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    obs = np.where(map < 200)
    
    m = Map(2000, 2000)
    ox, oy = [], []
    ox = obs[1]
    oy = 2000 - obs[0]
    m.set_obstacle([(i, j) for i, j in zip(ox, oy)])
    
    # start =  [1845, 2000-582]
    #start = [1808, 2000-582]
    start = [1758, 2000-656]
    goal1 = [1122, 2000-846]
    goal2 = [944, 2000-1144]
    goal3 = [482, 2000-732]
    
    print(map[2000-start[1]][start[0]])
    
    if show_animation:
        plt.plot(ox, oy, ".k", markersize=0.05)
        plt.plot(start[0], start[1], "ob")
        plt.plot(goal1[0], goal1[1], "xr")
        plt.plot(goal2[0], goal2[1], "xg")
        plt.plot(goal3[0], goal3[1], "xb")
        plt.axis("equal")
        
    start = m.map[start[0]][start[1]]
    goal1 = m.map[goal1[0]][goal1[1]]
    goal2 = m.map[goal2[0]][goal2[1]]
    goal3 = m.map[goal3[0]][goal3[1]]
    
    # dstar1 = Dstar(m)
    # rx1, ry1 = dstar1.run(start, goal1)
    # np.save('route1', [rx1, ry1])
    # dstar2 = Dstar(m)
    # rx2, ry2 = dstar2.run(goal1, goal2)
    # np.save('route2', [rx2, ry2])
    # dstar3 = Dstar(m)
    # rx3, ry3 = dstar3.run(goal2, goal3)
    # np.save('route3', [rx3, ry3])
    dstar4 = Dstar(m)
    rx4, ry4 = dstar4.run(goal3, start)
    np.save('route4', [rx4, ry4])

    if show_animation:
        # plt.plot(rx1, ry1, "-r")
        # print('goal1 check')
        # plt.plot(rx2, ry2, "-g")
        # print('goal2 check')
        # plt.plot(rx3, ry3, "-b")
        # print('goal3 check')
        plt.plot(rx4, ry4, "-r")
        print('goal4 check')
        plt.show()


if __name__ == '__main__':
    main()
    