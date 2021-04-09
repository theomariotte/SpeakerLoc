"""
This code estimates de DOA of multiple sound sources based on the method described in [1]
[1] MAXIMUM LIKELIHOOD MULTI-SPEAKER DIRECTION OF ARRIVAL ESTIMATION UTILIZING A WEIGHTED HISTOGRAM, Hadad et Gannot (2020)
"""

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from microphone_array import MicArray
from grid import CircularGrid2D
from wave_reader import WaveProcessorSlidingWindow
from doa_estimator import DoaMLE
from doa_estimator import DoaDelayAndSumBeamforming

# Synthetic data
wav_dir = "./data/Synth_data/output/CIRC/"
audio_names = ["IS1000a_T23_nch8_snrinf_ola1_noise0"]

# open pickle file where segments are stored with doa per speaker
with open(wav_dir+audio_names[0]+".pkl",'rb') as fh:
    segments = pickle.load(fh)

doa = list()
for seg in segments:
    doa.append(seg["doa"])


# True DOAs and microphone array parameters
doa_ref = np.array(doa)
nb_mic = 8
ref_mic_idx=0

#circular array parameters
theta_start = 0
theta_stop = 360
radius = 0.1

# circular grid parameters
start = 0
step = 5
stop = 360
r = 1

# confidence measure threshold
thres = 1.0

# wave reader instance
sig = WaveProcessorSlidingWindow(wav_dir=wav_dir,
                                 audio_names=audio_names)

# sliding window parameters
winlen = 2048
winshift = winlen//2

# number of snapshots for doa estimation (i.e. number of frame)
duration = 4.0
num_snapshot = int(duration*16000//winlen)
#num_snapshot = int(0.25*duration*16000//winlen)
num_snapshot = 2

sig.load(winlen=winlen, shift=winshift, avoid_null=True)
fs = sig.getFs()
num_frame = len(sig)

# frequency vector
freq = np.linspace(0.0,fs/2,int(winlen/2))

# Frequency band to be studied
fmin = 100
fmax = fs/2
tmp_inf = np.abs(freq-fmin)
tmp_sup = np.abs(freq-fmax)
f_start_idx = np.where(tmp_inf == min(tmp_inf))[0]
f_stop_idx = np.where(tmp_sup == min(tmp_sup))[0]
freq_idx_vec = np.arange(f_start_idx, f_stop_idx+1).astype(np.int32)
nb_freq = len(freq_idx_vec)

# create circular microphone array
theta_step = (theta_stop - theta_start)/nb_mic
theta = np.arange(theta_start,theta_stop,theta_step)
theta*=np.pi/180.0
x_mic = radius * np.cos(theta)
y_mic = radius * np.sin(theta)
z_mic = np.zeros(x_mic.shape)

micropnts = np.array([x_mic,y_mic,z_mic]).T

# microphone array instance
mic_array = MicArray(micropnts=micropnts)

# Grid instance
grid = CircularGrid2D(theta_start=start,
                      theta_stop=stop,
                      theta_step=step,
                      radius=r)

# add microphone array to the grid to allow RTF computations
grid.addMicArray(mic_array)

rtf = grid.getRDTF(freq=freq,
                  fs=fs,
                  freq_idx_vec=freq_idx_vec,
                  reference_idx=ref_mic_idx)

coord = grid.components()
theta = coord["theta"]
theta *= 180. / np.pi
num_src = len(grid)

"""
# baseline beamforming
doaBaseline = DoaDelayAndSumBeamforming(microphone_array=mic_array,
                                        grid=grid,
                                        wave_reader=sig)
tt = sig.timeSupport()

for idx in range(len(sig)):

    doaMap = doaBaseline.energyMap(idx_frame=idx)
    doaMap = 10*np.log10(doaMap/np.max(doaMap))

    fig = plt.figure()
    theta_plt = theta*np.pi/180.0 -np.pi
    #theta_plt = theta

    plt.polar(theta_plt,doaMap)
    if tt[idx] >= segments[0]["start"] and tt[idx] <= segments[0]["stop"]:
        plt.polar(doa_ref[0]*(np.pi/180.0)-np.pi, 0.1,'o',color='r')
    if tt[idx] >= segments[1]["start"] and tt[idx] <= segments[1]["stop"]:
        plt.polar(doa_ref[1]*(np.pi/180.0)-np.pi, 0.1,'o',color='g')    
    if tt[idx] >= segments[2]["start"] and tt[idx] <= segments[2]["stop"]:
        plt.polar(doa_ref[2]*(np.pi/180.0)-np.pi, 0.1,'o',color='b')
    if tt[idx] >= segments[3]["start"] and tt[idx] <= segments[3]["stop"]:
        plt.polar(doa_ref[3]*(np.pi/180.0)-np.pi, 0.1,'o',color='y')
    plt.xlabel("DOA [°]")
    plt.ylabel("power")
    plt.title(f"Window : {tt[idx]}")
    plt.show()
"""

# doa estimator
doaEngine = DoaMLE(microphone_array=mic_array,
                   grid=grid,
                   wave_reader=sig)


nb_loop = int(num_frame//num_snapshot)
for snp_idx in range(nb_loop):
    idx_frame_start = 0 + snp_idx * num_snapshot
    idx_frame_stop = idx_frame_start + num_snapshot
    H = doaEngine.broadBandHistogram(idx_frame_start=idx_frame_start,
                                     idx_frame_stop=idx_frame_stop,
                                     freq=freq,
                                     snr_thres=thres,
                                     freq_idx_vec=freq_idx_vec,
                                     ref_mic_idx=ref_mic_idx)
    doaEngine.time()

    ref_src1 = [doa_ref[0],doa_ref[0]]
    #ref_src2 = [doa_ref[1],doa_ref[1]]
    #ref_src3 = [doa_ref[2],doa_ref[2]]
    #ref_src4 = [doa_ref[3],doa_ref[3]]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(theta, H/np.max(np.abs(H)))
    plt.plot([doa_ref,doa_ref],[0,1],"k--")
    plt.xlabel("DOA [°]")
    plt.ylabel("Likelihood")
    plt.grid()
    plt.show(block=True)
