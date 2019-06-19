# -*- coding: utf-8 -*-
"""
@author: Jacopo Carrani
"""

import numpy as np
from tqdm import tqdm
from scipy.signal import stft
from scipy.io import wavfile

def build_dataset(RIR,rir_len=4800,test=0,locata=0):
    '''
    Create train/test set and respective labels.
    
    Inputs:
        RIR         np array containing RIRs of size [rir_len*exp_nr,M]
        rir_len     length of each RIR
        test        flag to build testing/training dataset
    Outputs:
        x_data      np array containing test/train data of size [exp_nr*timeframes,1,256,M]
        y_data      np array containing labels of size [exp_nr*timeframes,37]
    '''
    exp_nr = int(len(RIR)/rir_len)
    
    if locata == 0:
      mic_nr = 4
    else:
      mic_nr = 5
    
    rir_unpacked = RIR.reshape((exp_nr,rir_len,mic_nr))
    
    print('Number of experiments:',exp_nr)
    
    tframes=176
    nangles=37
    
    # Initialise outputs
    x_data = np.zeros((exp_nr*tframes,1,256,mic_nr))
    y_data= np.zeros((exp_nr*tframes,nangles))
    

    sig_spatial = np.zeros((44799,mic_nr))
    
    if test != 0:
      sig_input, fs = librosa.load('/content/gdrive/My Drive/CNN/audio_source_loudspeaker1.wav', sr=16000)
    # for each experiment convolve wgn and RIR, add noise
    ang = 0
    for exp in tqdm(range(0,exp_nr)):
        if test == 0:
          sig_input = np.random.normal(0,np.random.uniform(0.1,1),40000)
        else:
          sig_input = np.random.normal(0,np.random.uniform(0.1,1),40000) 
        # do for each microphone
        for m in range(0,mic_nr):
            # wgn * RIR
            sig_spatial[:,m] = np.convolve(sig_input,rir_unpacked[exp,:,m])
        # Set a target SNR
        if test == 0:
          SNR = np.random.uniform(0,20)
        else:
          SNR = test-1
        sig_pow = 10 * np.log10(np.sum((sig_spatial[:,1]**2)/len(sig_spatial[:,1])))
        noise_pow = 10 ** ((sig_pow - SNR)/10)
        # Generate a sample of white noise
        noise = np.random.normal(0, np.sqrt(noise_pow), sig_spatial.shape)
        # add noise
        sig_noisy = sig_spatial + noise
        # stft
        sig_stft = stft(sig_noisy.T,fs=16000,nperseg=512,window=np.hanning(512),noverlap=256)[2][:,0:-1,:].T
        # Output from experiment
        x_data[exp*tframes:exp*tframes+tframes,0,:,:] = np.angle(sig_stft)
        if ang >(nangles-1):
            ang = 0
        y_data[exp*tframes:exp*tframes+tframes,ang] = 1
        ang +=1
    
    return x_data, y_data