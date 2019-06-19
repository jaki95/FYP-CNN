import numpy as np
from tqdm import tqdm
import gpuRIR
import pickle as pkl
import os

angles = np.radians(np.arange(0,185,5))
nb_src = 1  # Number of sources
nb_rcv = 5 # Number of receivers
fs=16000.0 # Sampling frequency [Hz]
att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
att_max = 60.0 # Attenuation at the end of the simulation [dB]
Tmax = 0.6
Tdiff = Tmax
MIN_ARR2WALL = np.array([2.01,0.01,1])
MAX_ARR2WALL = np.array([2.01,2.01,1])


# Room settings
rooms   = [[4,3.5,2.5],
          [5.5,4,3.5],
          [7,7.5,5]]
centres = [[2.25,2,1.5],
          [2.5,1.75,1.6],
          [3.3,3,1.7]]
distances = [0.5,1,2]
rev_times = [[0,0.16,0.16*2,0.16*3,0.16*4,0.16*5],
            [0,0.24,0.24*2,0.24*3,0.24*4,0.24*5],
            [0,0.3,0.3*2,0.3*3,0.3*4,0.3*5]]

# Generate RIRs
RIRs = np.zeros((0,5))
for r in range(0,len(rooms)):
    room_sz = rooms[r]
    arr_centre = np.random.uniform(MIN_ARR2WALL,room_sz-MAX_ARR2WALL,size=[1,3])
    arr = arr_centre[-1]*np.ones((3,5))
    arr[0,0] = arr_centre[0]-0.08
    arr[0,1] = arr_centre[0]-0.04
    arr[0,2] = arr_centre[0]
    arr[0,3] = arr_centre[0]+0.04
    arr[0,4] = arr_centre[0]+0.08
    arr[1,:] = arr_centre[1]
    pos_rcv = arr.T
    for rt in range(0,6):
      T60 = rev_times[r]
      dist = distances[r]
      for ang in tqdm(range(0,37)):
          x = np.cos(angles[ang]) * dist + arr_centre[0,0]
          y = np.sin(angles[ang]) * dist + arr_centre[0,1]
          pos_src = np.array([[x, y, 1.5]])
          beta = gpuRIR.beta_SabineEstimation(room_sz, T60[rt]) # Reflection coefficients
          nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
          _rirs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs).reshape(5,9600).T
          RIRs = np.concatenate((RIRs,_rirs))