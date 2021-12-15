import numpy as np
import cv2
import matplotlib.pyplot as plt

show_animation = True

map = cv2.imread('pkg/maps/ROBOT_NAVIGATION.png',  cv2.IMREAD_GRAYSCALE)
obs = np.where(map < 200)

ox, oy = [], []
ox = obs[1]
oy = 2000 - obs[0]

route1 = np.load('route1.npy')
route2 = np.load('route2.npy')
route3 = np.load('route3.npy')

# print(route1)
# print(route2)
# print(route3)

start = [1808, 2000-582]
goal1 = [1122, 2000-846]
goal2 = [944, 2000-1144]
goal3 = [482, 2000-732]

if show_animation:
    plt.plot(ox, oy, ".k", markersize=0.05)
    plt.plot(start[0], start[1], "ob")
    plt.plot(goal1[0], goal1[1], "xr")
    plt.plot(goal2[0], goal2[1], "xg")
    plt.plot(goal3[0], goal3[1], "xb")
    plt.plot(route1[0], route1[1], "-r")
    plt.plot(route2[0], route2[1], "-g")
    plt.plot(route3[0], route3[1], "-b")
    plt.axis("equal")
    plt.show()
    