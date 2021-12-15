import numpy as np
import math
import cv2

from sys import maxsize

class GapFollower:
    BUBBLE_RADIUS = 160
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000
    STRAIGHTS_SPEED = 8.0
    CORNERS_SPEED = 5.0
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees

    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle

    def process_lidar(self, ranges, coordinate):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        proc_ranges = self.preprocess_lidar(ranges)
        # Find closest point to LiDAR
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED
        print('Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        return speed, steering_angle


# drives straight ahead at a speed of 5
class CustomDriver:

    def process_lidar(self, ranges, coordinate):
        x_sim = coordinate[0]
        y_sim = coordinate[1]
        speed = 5.0
        steering_angle = 0.0
        return speed, steering_angle


# drives toward the furthest point it sees
class AnotherDriver:

    def process_lidar(self, ranges, coordinate):
        # the number of LiDAR points
        NUM_RANGES = len(ranges)
        # angle between each LiDAR point
        ANGLE_BETWEEN = 2 * np.pi / NUM_RANGES
        # number of points in each quadrant
        NUM_PER_QUADRANT = NUM_RANGES // 4

        # the index of the furthest LiDAR point (ignoring the points behind the car)
        max_idx = np.argmax(ranges[NUM_PER_QUADRANT:-NUM_PER_QUADRANT]) + NUM_PER_QUADRANT
        # some math to get the steering angle to correspond to the chosen LiDAR point
        steering_angle = max_idx * ANGLE_BETWEEN - (NUM_RANGES // 2) * ANGLE_BETWEEN
        speed = 5.0

        return speed, steering_angle


class DisparityExtender:
    CAR_WIDTH = 0.31
    # the min difference between adjacent LiDAR points for us to call them disparate
    DIFFERENCE_THRESHOLD = 2.
    SPEED = 5.
    # the extra safety room we plan for along walls (as a percentage of car_width/2)
    SAFETY_PERCENTAGE = 300.

    def preprocess_lidar(self, ranges):
        """ Any preprocessing of the LiDAR data can be done in this function.
            Possible Improvements: smoothing of outliers in the data and placing
            a cap on the maximum distance a point can be.
        """
        # remove quadrant of LiDAR directly behind us
        eighth = int(len(ranges) / 8)
        return np.array(ranges[eighth:-eighth])

    def get_differences(self, ranges):
        """ Gets the absolute difference between adjacent elements in
            in the LiDAR data and returns them in an array.
            Possible Improvements: replace for loop with numpy array arithmetic
        """
        differences = [0.]  # set first element to 0
        for i in range(1, len(ranges)):
            differences.append(abs(ranges[i] - ranges[i - 1]))
        return differences

    def get_disparities(self, differences, threshold):
        """ Gets the indexes of the LiDAR points that were greatly
            different to their adjacent point.
            Possible Improvements: replace for loop with numpy array arithmetic
        """
        disparities = []
        for index, difference in enumerate(differences):
            if difference > threshold:
                disparities.append(index)
        return disparities

    def get_num_points_to_cover(self, dist, width):
        """ Returns the number of LiDAR points that correspond to a width at
            a given distance.
            We calculate the angle that would span the width at this distance,
            then convert this angle to the number of LiDAR points that
            span this angle.
            Current math for angle:
                sin(angle/2) = (w/2)/d) = w/2d
                angle/2 = sininv(w/2d)
                angle = 2sininv(w/2d)
                where w is the width to cover, and d is the distance to the close
                point.
            Possible Improvements: use a different method to calculate the angle
        """
        angle = 2 * np.arcsin(width / (2 * dist))
        num_points = int(np.ceil(angle / self.radians_per_point))
        return num_points

    def cover_points(self, num_points, start_idx, cover_right, ranges):
        """ 'covers' a number of LiDAR points with the distance of a closer
            LiDAR point, to avoid us crashing with the corner of the car.
            num_points: the number of points to cover
            start_idx: the LiDAR point we are using as our distance
            cover_right: True/False, decides whether we cover the points to
                         right or to the left of start_idx
            ranges: the LiDAR points

            Possible improvements: reduce this function to fewer lines
        """
        new_dist = ranges[start_idx]
        if cover_right:
            for i in range(num_points):
                next_idx = start_idx + 1 + i
                if next_idx >= len(ranges): break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        else:
            for i in range(num_points):
                next_idx = start_idx - 1 - i
                if next_idx < 0: break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        return ranges

    def extend_disparities(self, disparities, ranges, car_width, extra_pct):
        """ For each pair of points we have decided have a large difference
            between them, we choose which side to cover (the opposite to
            the closer point), call the cover function, and return the
            resultant covered array.
            Possible Improvements: reduce to fewer lines
        """
        width_to_cover = (car_width / 2) * (1 + extra_pct / 100)
        for index in disparities:
            first_idx = index - 1
            points = ranges[first_idx:first_idx + 2]
            close_idx = first_idx + np.argmin(points)
            far_idx = first_idx + np.argmax(points)
            close_dist = ranges[close_idx]
            num_points_to_cover = self.get_num_points_to_cover(close_dist,
                                                               width_to_cover)
            cover_right = close_idx < far_idx
            ranges = self.cover_points(num_points_to_cover, close_idx,
                                       cover_right, ranges)
        return ranges

    def get_steering_angle(self, range_index, range_len):
        """ Calculate the angle that corresponds to a given LiDAR point and
            process it into a steering angle.
            Possible improvements: smoothing of aggressive steering angles
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_point
        steering_angle = np.clip(lidar_angle, np.radians(-90), np.radians(90))
        return steering_angle

    def _process_lidar(self, ranges):
        """ Run the disparity extender algorithm!
            Possible improvements: varying the speed based on the
            steering angle or the distance to the farthest point.
        """
        self.radians_per_point = (2 * np.pi) / len(ranges)
        proc_ranges = self.preprocess_lidar(ranges)
        differences = self.get_differences(proc_ranges)
        disparities = self.get_disparities(differences, self.DIFFERENCE_THRESHOLD)
        proc_ranges = self.extend_disparities(disparities, proc_ranges,
                                              self.CAR_WIDTH, self.SAFETY_PERCENTAGE)
        steering_angle = self.get_steering_angle(proc_ranges.argmax(),
                                                 len(proc_ranges))
        speed = self.SPEED
        return speed, steering_angle

    def process_observation(self, ranges, ego_odom):
        return self._process_lidar(ranges)


##################################################
############## My customized driver ##############
##################################################
## Robot Navigation Term Project2 ################
## ID: 2016741012 ################################
## NAME: Moon Seokjun ############################
## CONTACT: msjun23@gmail.com ####################
##################################################

# class CustomDriver_:
#     BUBBLE_RADIUS = 160             # 160
#     PREPROCESS_CONV_SIZE = 3
#     BEST_POINT_CONV_SIZE = 80
#     MAX_LIDAR_DIST = 3000000
#     STRAIGHTS_SPEED = 5.0           # 8.0
#     CORNERS_SPEED = 5.0             # 5.0
#     STRAIGHTS_STEERING_ANGLE = np.pi / 18       # 10 degrees = np.pi / 18

#     def __init__(self):
#         # used when calculating the angles of the LiDAR data
#         self.radians_per_elem = None

#     def preprocess_lidar(self, ranges):
#         """ Preprocess the LiDAR scan array. Expert implementation includes:
#             1.Setting each value to the mean over some window
#             2.Rejecting high values (eg. > 3m)
#         """
#         self.radians_per_elem = (2 * np.pi) / len(ranges)
#         # we won't use the LiDAR data from directly behind us
#         proc_ranges = np.array(ranges[135:-135])
#         # sets each value to the mean over a given window
#         proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
#         proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
#         return proc_ranges

#     def find_max_gap(self, free_space_ranges):
#         """ Return the start index & end index of the max gap in free_space_ranges
#             free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
#         """
#         # mask the bubble
#         masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
#         # get a slice for each contigous sequence of non-bubble data
#         slices = np.ma.notmasked_contiguous(masked)
#         max_len = slices[0].stop - slices[0].start
#         chosen_slice = slices[0]
#         # I think we will only ever have a maximum of 2 slices but will handle an
#         # indefinitely sized list for portablility
#         for sl in slices[1:]:
#             sl_len = sl.stop - sl.start
#             if sl_len > max_len:
#                 max_len = sl_len
#                 chosen_slice = sl
#         return chosen_slice.start, chosen_slice.stop

#     def find_best_point(self, start_i, end_i, ranges):
#         """Start_i & end_i are start and end indices of max-gap range, respectively
#         Return index of best point in ranges
#         Naive: Choose the furthest point within ranges and go there
#         """
#         # do a sliding window average over the data in the max gap, this will
#         # help the car to avoid hitting corners
#         averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
#                                        'same') / self.BEST_POINT_CONV_SIZE
#         return averaged_max_gap.argmax() + start_i

#     def get_angle(self, range_index, range_len):
#         """ Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
#         """
#         lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
#         steering_angle = lidar_angle / 2
#         return steering_angle
    
#     ##########

#     prev_point = [0.8007017, -0.2753365]
    
#     # check point 0~17
#     dest = [[1664.0, 773.0], [1551.0, 842.0], [1207.0, 900.0], [1122, 846], 
#             [1207.0, 900.0], [1551.0, 842.0], [1594.0, 924.0], [979.0, 1231.0], [902.0, 1004.0], 
#             [943.0, 917.0], [847, 709], [666.0, 775.0], [610, 776], [594.0, 727.0], [544.0, 723.0], [508, 742], [480, 711], [591.0, 568.0]]
    
#     check_point_cnt = 0
#     check_point_cnt_end = 18
#     check_point_mission_flag = True
    
#     def RAD2DEG(self, rad):
#         return rad * 180 / math.pi
    
#     def Img2SimCoordinate(self, img_coordinate):
#         x_sim = img_coordinate[0] * 0.08534 - 156.68159080844768
#         y_sim = (2000 - img_coordinate[1]) * 0.08534 - 121.23484964729177
#         return [x_sim, y_sim]
        
#     def Sim2ImgCoordinate(self, sim_coordinate):
#         x_img = (sim_coordinate[0] + 156.68159080844768) / 0.08534
#         y_img = 2000 - ((sim_coordinate[1] + 121.23484964729177) / 0.08534)
#         return [x_img, y_img]
    
#     def GetVect(self, sx, sy, gx, gy):
#         # Calculate current direction vector
#         # sx, sy: starting coordinate
#         # gx, gy: goal coordinate
#         return [gx-sx, gy-sy]
    
#     def CalcDist(self, cx, cy, gx, gy):
#         # Calculate distance, current point to goal point
#         # cx, cy: current coordinate
#         # gx, gy: goal coordinate
#         return math.sqrt(pow((gx-cx), 2) + pow((gy-cy), 2))
    
#     def process_lidar(self, ranges, coordinate):
#         if (self.check_point_mission_flag):
#             tar_point = self.Img2SimCoordinate(self.dest[self.check_point_cnt])
            
#             dist = self.CalcDist(coordinate[0], coordinate[1], tar_point[0], tar_point[1])
#             if (dist <= 1.0):
#                 # if robot is arrived at check point
#                 self.check_point_cnt += 1
#                 if (self.check_point_cnt >= self.check_point_cnt_end):
#                     # if check point missions are completed
#                     self.check_point_mission_flag = False
#                     return 10.0, 0.0
#                 else:
#                     tar_point = self.Img2SimCoordinate(self.dest[self.check_point_cnt])
#             else:
#                 pass
            
#             curr_dir_vect = self.GetVect(self.prev_point[0], self.prev_point[1], coordinate[0], coordinate[1])
#             tar_dir_vect = self.GetVect(self.prev_point[0], self.prev_point[1], tar_point[0], tar_point[1])
#             vect_angle = math.atan2(np.cross(tar_dir_vect, curr_dir_vect), np.dot(tar_dir_vect, curr_dir_vect))
            
#             self.prev_point = coordinate
            
#             # Angle Controller
#             Kp = 0.6
#             steering_angle = Kp * -vect_angle
            
#             # Speed Controller
#             # if abs(steering_angle) > 0.1:    speed = 5.0
#             # else:   speed = 8.0
#             # normal state
#             speed = 10 - (10 * abs(steering_angle))
            
#             # straight line
#             front_max_dist = max(ranges[530:550])
#             if self.check_point_cnt>=7 and abs(steering_angle) < 0.05 and front_max_dist > 20.0:
#                     speed = 0.6 * front_max_dist
#                     steering_angle = 0.0
            
#             # slowly near target point
#             if (dist < 3.0):
#                 speed = 5.0;
#                 steering_angle = Kp * -vect_angle
#             # minimum speed limit
#             if (speed < 3.0):
#                 speed = 3.0
#                 steering_angle = Kp * -vect_angle
#         else:
#             proc_ranges = self.preprocess_lidar(ranges)
#             # Find closest point to LiDAR
#             closest = proc_ranges.argmin()

#             # Eliminate all points inside 'bubble' (set them to zero)
#             min_index = closest - self.BUBBLE_RADIUS
#             max_index = closest + self.BUBBLE_RADIUS
#             if min_index < 0: min_index = 0
#             if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
#             proc_ranges[min_index:max_index] = 0

#             # Find max length gap
#             gap_start, gap_end = self.find_max_gap(proc_ranges)

#             # Find the best point in the gap
#             best = self.find_best_point(gap_start, gap_end, proc_ranges)

#             # Publish Drive message
#             steering_angle = self.get_angle(best, len(proc_ranges))
#             if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
#                 speed = self.CORNERS_SPEED
#             else:
#                 speed = self.STRAIGHTS_SPEED
            
#         #print('steering_angle: ', steering_angle, '/ speed: ', speed)
#         return speed, steering_angle



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
            # if show_animation:
            #     plt.plot(rx, ry, "-k")
            #     plt.pause(0.01)
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

class CustomDriver:
    load_path = True
    
    prev_point = [0.8007017, -0.2753365]
    path = []
    path_idx = 0
    
    def Img2SimCoordinate(self, img_coordinate):
        x_sim = img_coordinate[0] * 0.08534 - 156.68159080844768
        y_sim = (2000 - img_coordinate[1]) * 0.08534 - 121.23484964729177
        return [x_sim, y_sim]
        
    def Sim2ImgCoordinate(self, sim_coordinate):
        x_img = (sim_coordinate[0] + 156.68159080844768) / 0.08534
        y_img = 2000 - ((sim_coordinate[1] + 121.23484964729177) / 0.08534)
        return [x_img, y_img]
    
    def GetVect(self, sx, sy, gx, gy):
        # Calculate current direction vector
        # sx, sy: starting coordinate
        # gx, gy: goal coordinate
        return [gx-sx, gy-sy]
    
    def CalcDist(self, cx, cy, gx, gy):
        # Calculate distance, current point to goal point
        # cx, cy: current coordinate
        # gx, gy: goal coordinate
        return math.sqrt(pow((gx-cx), 2) + pow((gy-cy), 2))
    
    def process_lidar(self, ranges, coordinate):
        if self.load_path:
            # Load saved path at beginning
            route1 = np.load('route1.npy')
            route2 = np.load('route2.npy')
            route3 = np.load('route3.npy')
            route1[1] = 2000-route1[1]
            route2[1] = 2000-route2[1]
            route3[1] = 2000-route3[1]
            #print(np.shape(route1), np.shape(route2), np.shape(route3))
            
            # Loaded routes -> check point mission path
            self.path = np.append(route1, route2, axis=1)
            self.path = np.append(self.path, route3, axis=1)
            
            # Make route for unknown map
            kernel = np.ones((3, 3), np.uint8)
            map = cv2.imread('pkg/maps/ROBOT_NAVIGATION.png',  cv2.IMREAD_GRAYSCALE)
            map = cv2.erode(map, kernel, iterations=12)
            
            obs = np.where(map < 200)
    
            m = Map(2000, 2000)
            ox, oy = [], []
            ox = obs[1]
            oy = 2000 - obs[0]
            m.set_obstacle([(i, j) for i, j in zip(ox, oy)])
            
            start = [482, 2000-732]
            goal = [1758, 2000-656]
            start = m.map[start[0]][start[1]]
            goal = m.map[goal[0]][goal[1]]
            
            dstar = Dstar(m)
            rx, ry = dstar.run(start, goal)
            route4 = np.array([rx, ry])
            route4[1] = 2000-route4[1]
            self.path = np.append(self.path, route4, axis=1)
            
            self.load_path = False
            return 0, 0
        else:
            # Path tracking
            tar_point = self.Img2SimCoordinate([self.path[0][self.path_idx], self.path[1][self.path_idx]])
            
            dist = self.CalcDist(coordinate[0], coordinate[1], tar_point[0], tar_point[1])
            if (dist <= 1.0):
                # if robot is arrived at check point
                self.path_idx += 20
                if (self.path_idx >= len(self.path[0])):
                    # if check point missions are completed
                    print('Check point mission complete!')
                    return 0.0, 0.0
                else:
                    tar_point = self.Img2SimCoordinate([self.path[0][self.path_idx], self.path[1][self.path_idx]])
            else:
                pass
            
            curr_dir_vect = self.GetVect(self.prev_point[0], self.prev_point[1], coordinate[0], coordinate[1])
            tar_dir_vect = self.GetVect(self.prev_point[0], self.prev_point[1], tar_point[0], tar_point[1])
            vect_angle = math.atan2(np.cross(tar_dir_vect, curr_dir_vect), np.dot(tar_dir_vect, curr_dir_vect))
            
            self.prev_point = coordinate
            
            # Angle Controller
            Kp = 0.6
            steering_angle = Kp * -vect_angle
            speed = 5.0
        
            print('steering_angle: ', steering_angle, '/ speed: ', speed)
            return speed, steering_angle
        