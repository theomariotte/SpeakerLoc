import numpy as np
import logging
import matplotlib.pyplot as plt

import tqdm
from typing import Optional
from wave_reader import WaveProcessorSlidingWindow
from microphone_array import MicArray
from grid import CircularGrid2D

import time


class DoaBase():

    def __init__(self,
                 microphone_array,
                 grid,
                 wave_reader,
                 winlen=1024,
                 winshift=512):

        super().__init__()

        if not isinstance(microphone_array, MicArray):
            raise Exception("<mic_array> is not a <MicArray> object !")
        if not isinstance(grid, CircularGrid2D):
            raise Exception("<grid> is not a <CircularGrid2D> object !")
        if not isinstance(wave_reader, WaveProcessorSlidingWindow):
            raise Exception(
                "<wave_reader> is not a <WaveProcessorSlidingWindow> object !")

        self.micArray = microphone_array
        self.grid = grid
        self.grid.addMicArray(self.micArray)
        if not wave_reader.isLoad():
            wave_reader.load(winlen=winlen, winshift=winshift)
        self.waveProc = wave_reader
        self.fs = wave_reader.getFs()


class DoaMLE(DoaBase):
    """
    Method described in [1] by Hadad et Gannot. This method consist in localizing sources using a weighted histogram. 
    one localization spectrum (i.e. likelihood as a function of doa) is estimated per TF bin. It is supposed that one speaker 
    is active for each TF bin. Then, weight od the hitogram are computing using a confidence measure based on a posteriori SNR
    and   

    """

    def broadBandHistogram(self,
                           idx_frame_start,
                           idx_frame_stop,
                           freq,
                           snr_thres=1.0,
                           freq_idx_vec=None,
                           ref_mic_idx=None,
                           keep_trace=False):

        self.start_time=[]
        self.rtf_time=[]
        self.singleband_time=[]
        self.loadaudio_time=[]

        
        if freq_idx_vec is None:
            freq_idx_vec = range(len(freq))
        
        start_time = time.time()
        num_src = len(self.grid)
        if ref_mic_idx is not None:
            rtf = self.grid.getRDTF(freq=freq,
                                    fs=self.fs,
                                    reference_idx=ref_mic_idx)
        else:
            rtf = self.grid.getTF(freq=freq,
                                  fs=self.fs)
        self.rtf_time.append(time.time()-start_time)

        audioload_time = 0.0
        singleband_time = 0.0
        
        # to plot
        if keep_trace:
            self.sigma_sig = np.zeros((len(freq_idx_vec),idx_frame_stop-idx_frame_start)).T
            self.sigma_nn = np.zeros_like(self.sigma_sig)
            self.snr_vec = np.zeros_like(self.sigma_sig)
            self.ground_truth = np.zeros_like(self.sigma_sig)
        tt = 0

        for idx in tqdm.tqdm(range(idx_frame_start, idx_frame_stop), desc="Broad band processing", unit="frame"):

            tmp = time.time()
            xx, _ = self.waveProc.getAudioFrameSTFT(idx)
            audioload_time += (time.time() - tmp)

            # Broad band histogram
            H = np.zeros((num_src,))
            k = 0
            if keep_trace:
                self.ground_truth[tt,:] += np.squeeze(np.abs(xx[freq_idx_vec,0])**2)
            for f_idx in freq_idx_vec:
                tmp = time.time()
                log_spectrum, snr, ss2, sv2 = self._singleNarrowBandSpectrum(frame=xx,
                                                                            rtf=rtf,
                                                                            freq=freq[f_idx],
                                                                            freq_index=f_idx,
                                                                            ref_mic_idx=ref_mic_idx)
                singleband_time += (time.time() - tmp)

                #TODO (21/03/06) temporary remove first value -> problem here !
                log_spectrum = log_spectrum[1:]

                # confidence measure on DOA estimation
                idx_doa_NB = np.argmax(log_spectrum)
                #tmp_ = np.ma.array(log_spectrum, mask=False)
                #tmp_.mask[idx_doa_NB] = True
                qD = log_spectrum[idx_doa_NB] / log_spectrum.sum()

                #assert qD > 0
                if keep_trace: 
                    self.sigma_sig[tt,k]+=ss2
                    self.sigma_nn[tt,k]+=sv2
                    self.snr_vec[tt,k]+=snr

                # confidence measure on SNR
                qSNR = int(snr > snr_thres)

                # overall confidence measure (cf. Hadad et Gannot 2020)
                q_whole = qD * qSNR
                X = np.zeros((num_src,))
                X[idx_doa_NB] = 1

                # update histogram
                H += q_whole * X
                k += 1

                self.singleband_time.append(singleband_time/k)
            tt+=1
            self.loadaudio_time.append(audioload_time/(idx_frame_stop-idx_frame_start+1))
        if keep_trace:
            self.display(idx_frame_start,idx_frame_stop,freq_idx_vec,freq,slice_time=[0,int((idx_frame_stop-idx_frame_start)/2),int(idx_frame_stop-idx_frame_start)-1])
            
        return H

    def display(self,
                idx_frame_start,
                idx_frame_stop,
                freq_idx_vec,
                freq,
                slice_time=None):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        idx_ = np.arange(idx_frame_start,idx_frame_stop)
        ff_ = freq[freq_idx_vec]
        T,F = np.meshgrid(idx_,ff_)
        plt.pcolor(T.T,F.T,10*np.log10(np.abs(self.sigma_sig)))
        plt.xlabel('Time [samples]')
        plt.ylabel('Frequency [Hz')
        plt.title("Signal PSD")
        plt.colorbar(label="dB")
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.pcolor(T.T,F.T,10*np.log10(np.abs(self.sigma_nn)))
        plt.xlabel('Time [samples]')
        plt.ylabel('Frequency [Hz')
        plt.title("Noise PSD")
        plt.colorbar(label="dB")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.pcolor(T.T,F.T,10*np.log10(np.abs(self.snr_vec)))
        plt.xlabel('Time [samples]')
        plt.ylabel('Frequency [Hz')
        plt.title("SNR")
        plt.colorbar(label="dB")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.pcolor(T.T,F.T,10*np.log10(np.abs(self.ground_truth)))
        plt.xlabel('Time [samples]')
        plt.ylabel('Frequency [Hz')
        plt.title("Signal spectrogram")
        plt.colorbar(label="dB")

        if slice_time:
            fig=plt.figure()
            for slc in range(len(slice_time)):
                fig.add_subplot(len(slice_time),1,slc+1)
                plt.plot(ff_,10*np.log10(np.abs(self.sigma_sig[slice_time[slc],:])),linewidth=2,label="signal")
                plt.plot(ff_,10*np.log10(np.abs(self.sigma_nn[slice_time[slc],:])),linewidth=2,label="noise")
                plt.grid()
                plt.ylabel("DSP t={}".format(slice_time[slc]))
                plt.legend()
        plt.show()

    def _singleNarrowBandSpectrum(self,
                                 frame,
                                 rtf,
                                 freq,
                                 freq_index,
                                 ref_mic_idx=None):

        z = np.expand_dims((frame.T)[:, freq_index], axis=1)
        R = z @ z.conj().T
        B = self.micArray.getSpatialCoherence(freq=freq,
                                              fs=self.fs,
                                              mode="identity")

        B_inv = np.linalg.inv(B)
        mic_num = len(self.micArray)
        src_num = len(self.grid)
        log_spectrum = np.empty((src_num,))

        for ii in range(src_num):

            # RTF at the current frequency bin
            d = rtf[ii, freq_index, :]
            d = d[:, np.newaxis]

            #pseudo inverse of d
            #db = np.dot(B,d)
            #db_inv = np.linalg.pinv(db)
            tmp_1 = d.conj().T @ B_inv
            tmp_2 = 1.0/(np.matmul(d.conj().T, B_inv @ d))
            db_inv = tmp_2 * tmp_1
            #err_ = np.mean(np.mean(np.abs(db_inv-db_inv_2)**2))
            #db_inv = np.dot(d_inv,B_inv)

            # projection of d
            P = np.matmul(d, db_inv)

            # orthogonal projection of d
            P_orth = np.eye(mic_num) - P

            # Variance of the noise
            trace = np.trace( np.matmul(P_orth, R @ B_inv) )
            sigma_v2 = (1/(mic_num-1) * trace).astype(float)

            # DSP of the signal
            Sigma_V = sigma_v2 * B
            prod_ = np.matmul(db_inv, (R - Sigma_V) @ db_inv.conj().T)
            sigma_s2 = prod_[0][0].astype(float)

            spectrum = np.matmul(d, sigma_s2*d.conj().T) + Sigma_V
            log_spectrum[ii] = np.log(np.linalg.det(spectrum))

            # a posteriori SNR for confidence measure when weighting broadband histogram
            aSNR = (np.matmul(z.conj().T, z)/np.trace(Sigma_V)).astype(float)

        return 1./log_spectrum, aSNR[0][0], sigma_s2, sigma_v2

    def time(self):

        self.rtf_time = np.array(self.rtf_time)
        self.loadaudio_time = np.array(self.loadaudio_time)
        self.singleband_time = np.array(self.singleband_time)

        print("Mean RTF calculation time : {:1.6f}".format(np.mean(self.rtf_time)))
        print("Mean audio load time : {:1.6f}".format(np.mean(self.loadaudio_time)))
        print("Mean narrow band spectrum time : {:1.6f}".format(np.mean(self.singleband_time)))

class DoaDelayAndSumBeamforming(DoaBase):

    def energyMap(self,
                  idx_frame,
                  Nb: Optional[int] = 1024,
                  ref_index: Optional[int] = 0,
                  c0: Optional[float] = 343.0):

        Np = len(self.grid)
        frame = self.waveProc.getAudioFrame(index=idx_frame)
        coord = self.grid.components()
        X = coord["X"]
        Y = coord["Y"]
        powerMap = np.zeros((Np,))
        k = 0
        for x, y in zip(X, Y):
            bf_sig = self.micArray.beamformer(frame=frame,
                                              src_loc=np.array([x, y]),
                                              fs=self.fs)
            powerMap[k] = (bf_sig.sum())**2
            k += 1
        return powerMap
