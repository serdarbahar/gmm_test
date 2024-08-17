#!/usr/bin/env python3
import time
import argparse
import rospy
import numpy as np
import csv
import keyboard 
from movement_primitives.promp import ProMP
from all_models.py import T
from UR10_robot_interface import ur10_interface_moveit
from Gripper3F_interface import gripper3f_interface
RECORDRATE=100 #ms

LAST_TIME=int(time.time_ns() / 1_000_000)
LIMIT = 1000

print('Initialize robot')
rospy.init_node("Recorder")
rospy.sleep(2.0)
robot = ur10_interface_moveit.UR10()

# p = robot.get_cartesian_position()

data = []
num_data = input("Enter the number of trajectories to record: ")

for i in range(int(num_data)):
    print("Press Enter to start recording trajectory")
    print("Press X to stop recording")
    input()
    run = True
    print(f"Recording trajectory {i+1}")
    data.apppend([])
    while run and keyboard.is_pressed('x')==False:
        NOW=int(time.time_ns() / 1_000_000)
        if (NOW-LAST_TIME)>=RECORDRATE:
            LAST_TIME=NOW
            #add position and gripper data
            p = robot.get_cartesian_position()
            data[-1].append(p)
        #if limit passed
        if len(data)>=LIMIT :
            print("STOP RECORDING")
            run=False
        rospy.sleep(1)
    print(f"Trajectory {i+1} recorded")

print("Data recorded")

T = T()
INPUT = "promp"
T.start_training(INPUT, data)

print("Training done")

print("Adjust to waypoint")
print("Press X when ready")

waypoint = []
while True:
    if keyboard.is_pressed('x'):
        waypoint = robot.get_cartesian_position()
        break
    rospy.sleep(1)

print("Waypoint recorded")

T.condition_model(waypoint)

# calculate the probability of the waypoint
prob_pos = T.p_pos.get_probability(waypoint, 0.5)
prob_ori = T.p_or.get_probability(waypoint, 0.5)

print(f("Probability of position: {prob_pos}"))
print(f("Probability of orientation: {prob_ori}"))









