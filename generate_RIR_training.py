# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:47:23 2019

@author: Jacopo
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
rooms   = [[6,6,2.5],     # Room sizes [x,y,z] [m]
           [5,4,2.5],
           [10,6,2.5],
           [8,3,2.5],
           [8,5,2.5]]
T60     =  [0.3,             # Time for the RIR to reach 60dB of attenuation [s]
            0.2,
            0.8,
            0.4,
            0.6]

# Generate RIRs
pos_srcc = []
for r in range(0,1):#len(rooms)):
    room_sz = rooms[r]
    for p in range(0,7):
        if os.path.exists("RIRs_new\\Array_positions\\room{0}_position{1}.pkl".format(r,p)):
            with open("RIRs_new\\Array_positions\\room{0}_position{1}.pkl".format(r,p), 'rb') as f:
                arr_centre = pkl.load(f)
        else:
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
            for ang in tqdm(range(0,2)):
                x = np.cos(angles[ang]) * dist + arr_centre[0,0]
                y = np.sin(angles[ang]) * dist + arr_centre[0,1]
                pos_src = np.array([[x, y, 1.5]])
            
                beta = gpuRIR.beta_SabineEstimation(room_sz, T60[r]) # Reflection coefficients
                Tdiff= 0.075 # Time to start the diffuse reverberation model [s]
                #Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60[r])	 # Time to stop the simulation [s]
                nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
                _rirs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff).reshape(4,4800).T
                RIRs = np.concatenate((RIRs,_rirs))
                pos_srcc.append(pos_src)
            with open("RIRs_boh\\room{0}_position{1}_distance{2}.pkl".format(r,p,d), 'wb') as f:
                pkl.dump(RIRs, f,protocol=2)
            print("Saved room{0}_position{1}_distance{2}.pkl".format(r,p,d))
