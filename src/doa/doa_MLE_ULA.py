"""
This code estimates de DOA of multiple sound sources based on the method described in [1]
[1] MAXIMUM LIKELIHOOD MULTI-SPEAKER DIRECTION OF ARRIVAL ESTIMATION UTILIZING A WEIGHTED HISTOGRAM, Hadad et Gannot (2020)
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from microphone_array import MicArray
from grid import CircularGrid2D
from wave_reader import WaveProcessorSlidingWindow
from doa_estimator import DoaMLE
from doa_estimator import DoaDelayAndSumBeamforming

"""
# real data from AMi corpus
wav_dir = "./data/AMI/"
audio_names = []
for i in range(8):
    audio_names.append(f"IS1000a.Array1-0{i+1}")
"""
# plot estimated dsp
dsp_trace=True
# Synthetic data
wav_dir = "./data/Synth_data/output/ULA/"
audio_names = ["IS1000a_TR300_T13_nch4_snr10_ola1_noise0"]

# True DOAs and microphone array parameters
doa_ref = [45.-90,125.-90]
nb_mic = 4
ref_mic_idx=0

# in case of ULA    
x_start = -0.1
x_stop = 0.1

# circular grid parameters
start = -90
step = 5
stop = 90
r = 1

# confidence measure threshold
thres = 1.0

# wave reader instance
sig = WaveProcessorSlidingWindow(wav_dir=wav_dir,
                                 audio_names=audio_names)

# wavform framing parameters
winlen = 512
winshift = winlen//2

# number of snapshots for doa estimation (i.e. number of frame)
duration = 4.0
num_snapshot = int(duration*16000//winlen)
num_snapshot=len(sig)
#num_snapshot = int(0.25*duration*16000//winlen)
#num_snapshot = 50

sig.load(winlen=winlen, shift=winshift)
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

# create linear microphone array
x_step = (x_stop-x_start)/nb_mic
x_mic = np.arange(x_start,x_stop,x_step)
y_mic = np.zeros(x_mic.shape)
z_mic = np.zeros(x_mic.shape)

micropnts = np.array([x_mic,y_mic,z_mic]).T

# microphone array instance
mic_array = MicArray(micropnts=micropnts)

# Grid instance
grid = CircularGrid2D(theta_start=start,
                      theta_stop=stop,
                      theta_step=step,
                      radius=r)

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
    plt.polar(doa_ref[0]*(np.pi/180.0)-np.pi, 0.1,'o',color='r')
    plt.polar(doa_ref[1]*(np.pi/180.0)-np.pi, 0.1,'o',color='g')
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
                                     ref_mic_idx=ref_mic_idx,
                                     keep_trace=dsp_trace)

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
