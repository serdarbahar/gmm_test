#!/usr/bin/env python3

import numpy as np
import os
import time
from movement_primitives.promp import ProMP
from movement_primitives.dmp import DMP
from gmr import GMM


from geometry_msgs.msg import Pose
from ur10_mover.msg import ListOfPoses

isTraining = False
sample_length = 100
n_dims = 3

priors = 20 # number of GMM components  ?

g = GMM(n_components=priors, random_state=1234)

def read_data_files():
    data_folder_path = os.path.join(os.path.dirname(__file__), 'data')
    trajectory_list= []
    for filename in os.listdir(data_folder_path):
        if filename.startswith("data_"):
            file_path = os.path.join(data_folder_path, filename)
            with open(file_path, 'r') as input_file:
                traj = convert_data_file_to_list(input_file)
                trajectory_list.append(traj)
    return trajectory_list

def convert_data_file_to_list(input_file):
    traj = []
    saved_trajectory = input_file.readlines()
    input_file.close()
    for point in saved_trajectory:
        point= [float(i) for i in point[1:-2].split(',')]
        traj.append(point)
    return traj

def interpolate_points(points, total_points):
    num_segments = len(points) - 1
    points_per_segment = total_points // num_segments
    remaining_points = total_points % num_segments

    interpolated_points = []
    points = np.array(points)
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        num_points = points_per_segment + (1 if i < remaining_points else 0)

        for j in range(num_points):
            ratio = (j + 1) / (num_points + 1)  # Calculate ratio between start and end points
            interpolated_point = (1 - ratio) * start_point + ratio * end_point  # Linear interpolation formula
            interpolated_points.append(interpolated_point)
    return np.array(interpolated_points)

def start_training():

    trajectories = read_data_files()
    number_of_demonstrations = len(trajectories)
    all_demonstrations = []
    for trajectory in trajectories:
        interpolated_points = interpolate_points(trajectory,sample_length) 
        all_demonstrations.append(interpolated_points) 
    #rospy.loginfo(all_demonstrations)
    demo_data = np.array(all_demonstrations)
    demo_data = demo_data.reshape((number_of_demonstrations, sample_length, n_dims)) 
    
    Y = demo_data
        
    g.from_samples(Y)

def sample_trajectory(waypoints):

    trajectory = ""

    start = waypoints[0]
    end = waypoints[-1]

    _g = g
    _g = _g.condition([0,-1], [start, end])
    trajectory = _g.sample(1)
    trajectory = np.insert(trajectory, 0, start)
    trajectory = np.append(trajectory, end)

    
    sample = []
    sample = trajectory

    response_trajectory = []
    for point in sample: 
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = point[0], point[1], point[2]
        response_trajectory.append(pose)

    file_path = os.path.join(os.path.dirname(__file__), 'sample.txt')
    with open(file_path, 'w+') as file:
        for point in sample:
            file.write(str(point) + '\n')
    

if __name__ == "__main__":
    start_training()
    
    ### store waypoints with np arrays (num_of_waypoints, 3)
    
    ### then, sample with
    # sample_trajectory(waypoints)





    
