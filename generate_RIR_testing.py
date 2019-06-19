# -*- coding: utf-8 -*-
"""
@author: Jacopo Carrani
"""

import numpy as np
from tqdm import tqdm
import gpuRIR
import pickle as pkl
import os

angles = np.radians(np.arange(0,185,5))
nb_src = 1  # Number of sources
nb_rcv = 4 # Number of receivers
fs=16000.0 # Sampling frequency [Hz]
att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
att_max = 60.0 # Attenuation at the end of the simulation [dB]
Tmax = 0.3
MIN_ARR2WALL = np.array([2.01, 0.01, 1.5])
MAX_ARR2WALL = np.array([2.01, 2.01, 1])

# Room settings
rooms   = [[5,8,3],     # Room sizes [x,y,z] [m]
           [11,10,3],
           [6,7,3],
           [9,6,3]]
T60     = [0.9,             # Time for the RIR to reach 60dB of attenuation [s]
           0.1,
           0.95,
           1]

# Generate RIRs
for r in range(0,len(rooms)):
    room_sz = rooms[r]
    for p in [0,1,2]:
        arr_centre = np.random.uniform(MIN_ARR2WALL,room_sz-MAX_ARR2WALL,size=[1,3])
        arr = arr_centre[0,-1]*np.ones((3,4))
        arr[0,0] = arr_centre[0,0]-0.045
        arr[0,1] = arr_centre[0,0]-0.015
        arr[0,2] = arr_centre[0,0]+0.015
        arr[0,3] = arr_centre[0,0]+0.045
        arr[1,:] = arr_centre[0,1]
        pos_rcv = arr.T
        for d in [0,1]:
            dist = d+1
            RIRs = np.zeros((0,4))
            for ang in tqdm(range(0,37)):
                x = np.cos(angles[ang]) * dist + arr_centre[0,0]
                y = np.sin(angles[ang]) * dist + arr_centre[0,1]
                pos_src = np.array([[x, y, 1.5]])
            
                beta = gpuRIR.beta_SabineEstimation(room_sz, T60[r]) # Reflection coefficients
                Tdiff= 0.075 # Time to start the diffuse reverberation model [s]
                nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
                _rirs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff).reshape(4,4800).T
                RIRs = np.concatenate((RIRs,_rirs))
            with open("Test_ardiff\\room{0}_position{1}_distance{2}.pkl".format(r,p,d), 'wb') as f:
                pkl.dump(RIRs, f,protocol=2)
            print("Saved room{0}_position{1}_distance{2}.pkl".format(r,p,d))
