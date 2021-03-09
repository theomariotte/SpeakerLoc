"""
This code estimates de DOA of multiple sound sources based on the method described in [1]
[1] MAXIMUM LIKELIHOOD MULTI-SPEAKER DIRECTION OF ARRIVAL ESTIMATION UTILIZING A WEIGHTED HISTOGRAM, Hadad et Gannot (2020)
"""

"""
TODO:
        - Move doa estimation method into a class or another file reduce number of lines in main code
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
wav_dir = "./data/Synth_data/output/"
audio_names = ["IS1000a_TR300_T21_nch8_ola1_noise0"]

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
winlen = 1024
winshift = winlen//2

# number of snapshots for doa estimation (i.e. number of frame)
duration = 20.
num_snapshot = int(0.5*duration*16000//winlen)
num_snapshot = 30


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

"""
snp_cnt = 0
for idx in range(num_frame):

    logging.critical(f"Process frame {idx}/{num_snapshot}")

    xx, _ = sig.getAudioFrameSTFT(idx)

    broad_band_spectrum = np.zeros((num_src-1, nb_freq))
    conf_meas = np.zeros((nb_freq,))

    # Broad band histogram
    H = np.zeros((num_src,))
    k = 0

    for f_idx in freq_idx_vec:
        log_spectrum, snr = doaEngine.singleNarrowBandSpectrum(frame=xx,
                                                               rtf=rtf,
                                                               nb_freq=nb_freq,
                                                               freq_index=f_idx)

        #TODO (21/03/06) temporary remove first value -> problem here !
        log_spectrum = log_spectrum[1:]

        # confiudence measure on DOA estimation
        idx_doa_NB = np.argmax(log_spectrum)
        tmp_ = np.ma.array(log_spectrum, mask=False)
        tmp_.mask[idx_doa_NB] = True
        qD = log_spectrum[idx_doa_NB] / tmp_.sum()

        # confidence measure on SNR
        qSNR = int(snr > thres)

        # overall confidence measure (cf. Hadad et Gannot 2020)
        conf_meas[k] = qD * qSNR
        X = np.zeros((num_src,))
        X[idx_doa_NB] = 1

        # update histogram
        H += conf_meas[k] * X

        #print(f"Band {k} - aSNR = {snr} - qSNR = {qSNR} - qD = {qD}")
        broad_band_spectrum[:, k] = log_spectrum
        k += 1
"""