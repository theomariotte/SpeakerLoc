import numpy as np
import logging

from wave_reader import WaveProcessorSlidingWindow
from microphone_array import MicArray
from grid import CircularGrid2D


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
        if not wave_reader.isLoad():
            wave_reader.load(winlen=winlen, winshift=winshift)
        self.waveProc = wave_reader
        self.fs = wave_reader.getFs()


class DoaMLE(DoaBase):

    def broadBandHistogram(self,
                           idx_frame_start,
                           idx_frame_stop,
                           freq,
                           snr_thres = 1.0,
                           freq_idx_vec=None):

        if freq_idx_vec is None:
            freq_idx_vec = range(0, len(freq)-1, len(freq))

        nb_freq = len(freq_idx_vec)
        num_src = self.grid.shape()[0]
        rtf = self.grid.getRTF(freq=freq,
                               fs=self.fs,
                               array=self.micArray)

        for idx in range(idx_frame_start, idx_frame_stop+1):

            logging.critical(
                f"Process frame {idx - idx_frame_start}/{idx_frame_stop-idx_frame_start}")

            xx, _ = self.waveProc.getAudioFrameSTFT(idx)

            # Broad band histogram
            H = np.zeros((num_src,))
            k = 0
            for f_idx in freq_idx_vec:
                log_spectrum, snr = self.singleNarrowBandSpectrum(frame=xx,
                                                                  rtf=rtf,
                                                                  nb_freq=nb_freq,
                                                                  freq_index=f_idx)

                #TODO (21/03/06) temporary remove first value -> problem here !
                log_spectrum = log_spectrum[1:]

                # confidence measure on DOA estimation
                idx_doa_NB = np.argmax(log_spectrum)
                tmp_ = np.ma.array(log_spectrum, mask=False)
                tmp_.mask[idx_doa_NB] = True
                qD = log_spectrum[idx_doa_NB] / tmp_.sum()

                # confidence measure on SNR
                qSNR = int(snr > snr_thres)

                # overall confidence measure (cf. Hadad et Gannot 2020)
                q_whole = qD * qSNR
                X = np.zeros((num_src,))
                X[idx_doa_NB] = 1

                # update histogram
                H += q_whole * X
                k += 1

        return H

    def singleNarrowBandSpectrum(self,
                                 frame,
                                 rtf,
                                 nb_freq,
                                 freq_index):

        z = frame[:, freq_index]
        z = z[:, np.newaxis]
        R = np.dot(z, z.conj().T)
        B = self.micArray.getSpatialCoherence(idx_freq=freq_index,
                                              num_freq=nb_freq,
                                              fs=self.fs,
                                              mode="sinc")

        B_inv = np.linalg.inv(B)
        mic_num = self.micArray.micNumber()
        src_num = self.grid.shape()[0]

        log_spectrum = np.empty((src_num,))

        for ii in range(src_num):

            # RTF at the current frequency bin
            d = rtf[ii, freq_index, :]
            d = d[:, np.newaxis]

            #pseudo inverse of d
            d_inv = np.linalg.pinv(d)
            db_inv = np.dot(d_inv,B_inv)

            # projection of d
            P = np.dot(d, db_inv)

            # orthogonal projection of d
            P_orth = np.eye(mic_num) - P

            # Variance of the noise
            trace = np.trace(np.dot(P_orth, np.dot(R, B_inv)))
            sigma_v2 = 1/(mic_num-1) * trace

            # variance of the signal
            tmp_ = R - sigma_v2*B
            prod_ = np.dot(db_inv, np.dot(tmp_, db_inv.conj().T))
            sigma_s2 = prod_[0][0].astype(float)

            spectrum = np.dot(d, sigma_s2*d.conj().T) + sigma_v2 * B
            log_spectrum[ii] = np.log(np.linalg.det(spectrum))

            # a posteriori SNR for confidence measure when weighting broadband histogram
            Sigma_V = sigma_v2 * B
            aSNR = np.dot(z.conj().T, z)/np.trace(Sigma_V).astype(float)

        return 1./log_spectrum, aSNR[0][0]
