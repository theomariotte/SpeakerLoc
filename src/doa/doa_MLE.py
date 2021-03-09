"""
This code estimates de DOA of multiple sound sources based on the method described in [1]
[1] MAXIMUM LIKELIHOOD MULTI-SPEAKER DIRECTION OF ARRIVAL ESTIMATION UTILIZING A WEIGHTED HISTOGRAM, Hadad et Gannot (2020)
"""


# Synthetic data
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from microphone_array import MicArray
from grid import CircularGrid2D
from wave_reader import WaveProcessorSlidingWindow
from doa_estimator import DoaMLE
from doa_estimator import DoaDelayAndSumBeamforming

wav_dir = "./data/Synth_data/output/"
audio_names = ["IS1000a_TR300_T31_nch8_ola1_noise0"]

# real data from AMi corpus
#wav_dir = "../../03_DATA/AMI/"
#audio_names = []
#for i in range(8):
#    audio_names.append(f"IS1000a.Array1-0{i+1}")

# True DOAs
doa_ref = [45.,135.,225.,315.]

sig = WaveProcessorSlidingWindow(wav_dir=wav_dir,
                                 audio_names=audio_names)


# wavform framing parameters
winlen = 16000
winshift = winlen//2

# number of snapshots for doa estimation (i.e. number of frame)
duration = 20.
num_snapshot = int(0.5*duration*16000//winlen)
#num_snapshot = 20


sig.load(winlen=winlen, shift=winshift)
fs = sig.getFs()
num_frame = sig.frameNumber()

# build microphone array
micropnts = np.zeros((8, 3))
micropnts[0, :] = np.array([-0.1, 0, 0])
micropnts[1, :] = np.array([-0.1*np.sqrt(2)/2, -0.1*np.sqrt(2)/2, 0])
micropnts[2, :] = np.array([0, -0.1, 0])
micropnts[3, :] = np.array([0.1*np.sqrt(2)/2, -0.1*np.sqrt(2)/2, 0])
micropnts[4, :] = np.array([0.1, 0, 0])
micropnts[5, :] = np.array([0.1*np.sqrt(2)/2, 0.1*np.sqrt(2)/2, 0])
micropnts[6, :] = np.array([0, 0.1, 0])
micropnts[7, :] = np.array([-0.1*np.sqrt(2)/2, 0.1*np.sqrt(2)/2, 0])

mic_array = MicArray(micropnts=micropnts)

# build grid
start = 0.0
step = 5.0
stop = 360.0-step
r = 0.6
nfft = winlen//2+1
f_start_idx = 20
ff = np.linspace(f_start_idx, winlen//2, nfft).astype(np.int32)

grid = CircularGrid2D(theta_start=start,
                      theta_stop=stop,
                      theta_step=step,
                      radius=r)


rtf = grid.getRTF(freq=ff,
                  fs=fs,
                  array=mic_array)
coord = grid.components()
theta = coord["theta"]
theta *= 180. / np.pi
num_src = grid.shape()[0]


# baseline beamforming
doaBaseline = DoaDelayAndSumBeamforming(microphone_array=mic_array,
                                        grid=grid,
                                        wave_reader=sig)

for idx in range(sig.numel()):
    doaMap = doaBaseline.energyMap (idx_frame=idx)


    fig = plt.figure()
    plt.plot(theta,doaMap)
    plt.xlabel("DOA [°]")
    plt.ylabel("power")
    plt.show(block=False)
    plt.pause(1.0)
    plt.close(fig)

# frequency band used for loaclization
idx_fmin = 50
idx_fmax = len(ff)-1
freq_idx_vec = np.linspace(idx_fmin, idx_fmax, nfft-idx_fmin).astype(np.int32)
nb_freq = len(freq_idx_vec)




# threshold for confidence measure
thres = 1.0

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
                                     freq=ff,
                                     snr_thres=thres,
                                     freq_idx_vec=freq_idx_vec)

    ref_src1 = [doa_ref[0],doa_ref[0]]
    ref_src2 = [doa_ref[1],doa_ref[1]]
    ref_src3 = [doa_ref[2],doa_ref[2]]
    ref_src4 = [doa_ref[3],doa_ref[3]]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(theta, H,ref_src1,[0,50],"k--",ref_src2,[0,50],"k--",ref_src3,[0,50],"k--",ref_src4,[0,50],"k--")
    plt.xlabel("DOA [°]")
    plt.ylabel("Likelihood")
    plt.grid()
    plt.show(block=True)
