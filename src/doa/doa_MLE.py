"""
This code estimates de DOA of multiple sound sources based on the method described in [1]
[1] MAXIMUM LIKELIHOOD MULTI-SPEAKER DIRECTION OF ARRIVAL ESTIMATION UTILIZING A WEIGHTED HISTOGRAM, Hadad et Gannot (2020)
"""

"""
TODO / PROBLEMS :
        - Pyroom signals are sammpled at 48k -> not good since AMI is sampled at 16k
        - Move doa estimation method into a class or another file reduce number of lines in main code
        - Check pyroom simulation since the number of channels does not fit + mic coordinates
"""
import numpy as np
import matplotlib.pyplot as plt

from microphone_array import MicArray
from grid import CircularGrid2D
from wave_reader import WaveProcessorSlidingWindow



def singleNarrowBandSpectrum(frame,
                             rtf,
                             mic_array,
                             grid,
                             nb_freq,
                             freq_index,
                             fs):

    z = frame[:,freq_index]
    z = z[:,np.newaxis]
    R = np.dot(z,z.conj().T)
    B = mic_array.getSpatialCoherence(idx_freq=freq_index,
                                      num_freq=nb_freq,
                                      fs=fs,
                                      mode="sinc")
    
    B_inv = np.linalg.inv(B)

    mic_num = mic_array.micNumber()
    src_num = grid.shape()[0]

    log_spectrum = np.empty((src_num,))

    for ii in range(src_num):

        # RTF at the current frequency bin
        d = rtf[ii,freq_index,:]
        d = d[:,np.newaxis]

        # projection of d whitened by B
        proj_d = np.dot(d.conj().T,np.dot(B_inv,d))

        # inverse
        proj_d_inv = np.linalg.inv(proj_d) 

        # pseudo inversion
        db_inv = np.dot(proj_d_inv,np.dot(d.conj().T,B_inv))

        # projection of d
        P = np.dot(d,db_inv)

        # orthogonal projection of d
        P_orth = np.eye(mic_num) - P

        # Variance of the noise
        trace = np.trace(np.dot(P_orth,np.dot(R,B_inv)))
        sigma_v2 = 1/(mic_num-1) * trace

        # variance of the signal
        tmp_ = R - sigma_v2*B
        prod_ = np.dot(db_inv,np.dot(tmp_,db_inv.conj().T))
        sigma_s2 = prod_[0][0].astype(float)

        spectrum = np.dot(d,sigma_s2*d.conj().T) + sigma_v2 * B 
        log_spectrum[ii] = np.log(np.linalg.det(spectrum))

        # a posteriori SNR for confidence measure when weighting broadband histogram
        Sigma_V = sigma_v2 * B
        aSNR = np.dot(z.conj().T,z)/np.trace(Sigma_V).astype(float)


    return 1./log_spectrum, aSNR[0][0]

    

# Synthetic data
wav_dir = "../../03_DATA/Synth_data/output/"
audio_names = ["IS1000a_TR300"]

# real data from AMi corpus
#wav_dir = "../../03_DATA/AMI/"
#audio_names = []
#for i in range(8):
#    audio_names.append(f"IS1000a.Array1-0{i+1}")

sig = WaveProcessorSlidingWindow(wav_dir=wav_dir,
                                 audio_names=audio_names)


# wavform framing parameters
winlen = 1024
winshift = winlen

# number of snapshots for doa estimation (i.e. number of frame)
num_snapshot = 5


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
ff = np.linspace(f_start_idx,winlen//2,nfft).astype(np.int32)

grid = CircularGrid2D(theta_start=start,
                      theta_stop=stop,
                      theta_step=step,
                      radius=r)


rtf = grid.getRTF(freq=ff,
                  fs=fs,
                  array=mic_array)
coord = grid.components()
theta = coord["theta"]
theta*=180. / np.pi
num_src = grid.shape()[0]


# frequency band used for loaclization
idx_fmin = 50
idx_fmax = len(ff)-1
freq_idx_vec = np.linspace(idx_fmin,idx_fmax,nfft-idx_fmin).astype(np.int32)
nb_freq = len(freq_idx_vec)

# threshold for confidence measure
thres = 1.0

#TODO (2021/03/04) Make WaveProcessor object iterable to load segments more  efficiently
snp_cnt = 0
for idx in range(num_frame):

    xx, _ = sig.getAudioFrameSTFT(idx)

    broad_band_spectrum = np.zeros((num_src-1,nb_freq))
    conf_meas = np.zeros((nb_freq,))

    # Broad band histogram
    H = np.zeros((num_src,))
    # association function
    #X = np.zeros((num_src, nb_freq,num_snapshot)).astype(np.int32)
    # confidence measure
    #Q = np.zeros((nb_freq,num_snapshot))

    k = 0

    for f_idx in freq_idx_vec:
        log_spectrum, snr = singleNarrowBandSpectrum(frame=xx,
                                                rtf=rtf,
                                                mic_array=mic_array,
                                                grid=grid,
                                                nb_freq=nb_freq,
                                                freq_index=f_idx,
                                                fs=fs)    
        

        #TODO (21/03/06) temporary remove first value -> problem here !
        log_spectrum = log_spectrum[1:]

        # confiudence measure on DOA estimation 
        idx_doa_NB = np.argmax(log_spectrum)
        tmp_ = np.ma.array(log_spectrum,mask=False)
        tmp_.mask[idx_doa_NB]=True
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
        broad_band_spectrum[:,k] = log_spectrum
        k+=1

    
    snp_cnt+=1
    if snp_cnt == num_snapshot:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(theta,H)
        plt.xlabel("DOA [°]")
        plt.ylabel("Likelihood")
        plt.grid()
        plt.show(block=False)
        plt.pause(2.)
        plt.close(fig)

        H = np.zeros((num_src,))
        snp_cnt = 0



        

        



    

    
