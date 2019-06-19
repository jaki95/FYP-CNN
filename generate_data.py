# -*- coding: utf-8 -*-
"""
@author: Jacopo
"""

#%% Imports

import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.signal import stft
import pickle as pkl
from tqdm import tqdm
from build_dataset import build_dataset

#%% Merge RIRs - training
    
RIRs = np.zeros((0,4),float)
for r in range(0,1):
    for p in range(0,7):
        for d in [0,1]:
            with open("RIRs_new\\room{0}_position{1}_distance{2}.pkl".format(r,p,d), 'rb') as f:
                _rir = pkl.load(f)
            RIRs = np.concatenate((RIRs,_rir))
            
#%% Merge RIRs - testing
    
RIRs = np.zeros((0,4),float)
for r in range(0,2):
    for p in range(0,7):
        for d in [0,1]:
            with open("Test\\room{0}_position{1}_distance{2}.pkl".format(r,p,d), 'rb') as f:
                _rir = pkl.load(f)
            RIRs = np.concatenate((RIRs,_rir))

x_valid,y_valid=build_dataset(RIRs,test=1)

open('x_valid_sr.dat', 'wb').write(x_valid)
open('y_valid_sr.dat', 'wb').write(y_valid)

#%% Merge RIRs - LOCATA
    
RIRs = np.zeros((0,12),float)
for r in range(0,1):
    for p in range(0,7):
        for d in [0]:
            with open("RIRs_LOCATA\\room{0}_position{1}_distance{2}.pkl".format(r,p,d), 'rb') as f:
                _rir = pkl.load(f)
            RIRs = np.concatenate((RIRs,_rir))

x_valid,y_valid=build_dataset(RIRs,rir_len=9600,test=1)

#%% Generate and save training data

x_data,y_data=build_dataset(RIRs)

with open("Training Data\\x_data.pkl", 'wb') as f:
    pkl.dump(x_data, f)
with open("Training Data\\y_data.pkl", 'wb') as f:
    pkl.dump(y_data, f)
    
            