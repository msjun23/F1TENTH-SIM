import time
import gym
import numpy as np
import concurrent.futures
import os
import sys
import math

# Get ./src/ folder & add it to path
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# import your drivers here
from pkg.drivers import  CustomDriver as race_driver

# choose your drivers here (1-4)
drivers = [race_driver()]

# choose your racetrack here (SOCHI, SOCHI_OBS)
RACETRACK = 'ROBOT_NAVIGATION'


def _pack_odom(obs, i):
    keys = {
        'poses_x': 'pose_x',
        'poses_y': 'pose_y',
        'poses_theta': 'pose_theta',
        'linear_vels_x': 'linear_vel_x',
        'linear_vels_y': 'linear_vel_y',
        'ang_vels_z': 'angular_vel_z',
    }
    return {single: obs[multi][i] for multi, single in keys.items()}


class GymRunner(object):

    def __init__(self, racetrack, drivers):
        self.racetrack = racetrack
        self.drivers = drivers

    def run(self):
        # load map
        env = gym.make('f110_gym:f110-v0',
                       map="{}/maps/{}".format(current_dir, RACETRACK),
                       map_ext=".png", num_agents=len(drivers))

        # specify starting positions of each agent
        driver_count = len(drivers)
        if driver_count == 1:
            poses = np.array([[0.8007017, -0.2753365, 4.1421595]])
        elif driver_count == 2:
            poses = np.array([
                [0.8007017, -0.2753365, 4.1421595],
                [0.8162458, 1.1614572, 4.1446321],
            ])
        else:
            raise ValueError("Max 2 drivers are allowed")

        obs, step_reward, done, info = env.reset(poses=poses)
        env.render()

        laptime = 0.0
        start = time.time()

        coordinate =[0.8007017,-0.2753365]
        labcount =0
        while not done:
            actions = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                odom_0, odom_1 = _pack_odom(obs, 0), None
                if len(drivers) > 1:
                    odom_1 = _pack_odom(obs, 1)

                for i, driver in enumerate(drivers):
                    if i == 0:
                        ego_odom, opp_odom = odom_0, odom_1
                    else:
                        ego_odom, opp_odom = odom_1, odom_0
                    scan = obs['scans'][i]
                    if hasattr(driver, 'process_observation'):
                        futures.append(executor.submit(driver.process_observation, ranges=scan, ego_odom=ego_odom))
                    elif hasattr(driver, 'process_lidar'):
                        futures.append(executor.submit(driver.process_lidar, scan, coordinate))

            for future in futures:
                speed, steer = future.result()
                actions.append([steer, speed])
            actions = np.array(actions)
            obs, step_reward, done, info = env.step(actions)

            coordinate =[obs['poses_x'][0], obs['poses_y'][0]]
            laptime += step_reward
            # if obs['lap_counts'] ==1:
            #     labcount =1
            #     break
            x = (coordinate[1] - 2.403370489 - coordinate[0] * (1 / 0.59547882)) / (-0.59547882 - 1 / 0.59547882)
            if x > - 2.89891 and x < 2.136149:
                numer = (0.59547882 * coordinate[0] + coordinate[1] - 2.403370489) if (0.59547882 * coordinate[0] +
                                                                                       coordinate[1] - 2.403370489) > 0 \
                    else -(0.59547882 * coordinate[0] + coordinate[1] - 2.403370489)
                denom = math.sqrt(0.59547882 * 0.59547882 + 2.403370489 * 2.403370489)
                distance = numer / denom
                if distance < 0.4:
                    labcount = 1
                    break
            env.render(mode='human')

        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start,'lab count : ', labcount)


if __name__ == '__main__':
    runner = GymRunner(RACETRACK, drivers)
    runner.run()
