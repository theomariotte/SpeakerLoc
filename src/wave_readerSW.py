from wave_reader import WaveProcessor

import numpy as np
import soundfile
import copy
from typing import Optional

def framing(sig, win_size, win_shift=1, context=(0, 0), pad='zeros'):
    """
    :param sig: input signal, can be mono or multi dimensional
    :param win_size: size of the window in term of samples
    :param win_shift: shift of the sliding window in terme of samples
    :param context: tuple of left and right context
    :param pad: can be zeros or edge
    """

    dsize = sig.dtype.itemsize
    if sig.ndim == 1:
        sig = sig[:, np.newaxis]

    data = copy.deepcopy(sig)
    data = data[: ((data.shape[0] - win_size) // win_shift) * win_shift + win_size, :]

    # Manage padding
    c = (context, ) + (sig.ndim - 1) * ((0, 0), )
    _win_size = win_size + sum(context)
    shape = (int((sig.shape[0] - win_size) / win_shift) + 1, 1, _win_size, sig.shape[1])
    strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
    print(f"strides = {strides}")
    if pad == 'zeros':
        return np.lib.stride_tricks.as_strided(np.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                                  shape=shape,
                                                  strides=strides).squeeze()
    elif pad == 'edge':
        return np.lib.stride_tricks.as_strided(np.lib.pad(data, cc, 'edge'),
                                                  shape=shape,
                                                  strides=strides).squeeze()


class WaveProcessorSlidingWindow(WaveProcessor):

    def load(self,
             winlen: Optional[int] = 16000,
             shift: Optional[int] = 16000):
        
        audio = []
        for name in self.audio_names:
            fname = f"{self.wav_dir}{name}.wav"
            tmp_,fs = soundfile.read(fname)
            audio.append(tmp_)

        tmp_ = framing(sig=np.array(audio,dtype="float64").T,
                      win_size=winlen,
                      win_shift=shift)

        self.data_sw = tmp_
        self.fs = fs
        self.nCh = len(self.audio_names)
        self.isLoaded = True
    
    def getAudio(self):

        if not self.isLoaded:
            self.load()
            raise Warning("Sliding window computed with default parameters !")
        
        return self.data_sw


    def getAudioFrame(self,index):

        if not self.isLoaded:
                    self.load()
                    raise Warning("Sliding window computed with default parameters !")
        
        return self.data_sw[index,:,:].T
        

    def getAudioFrameSTFT(self,index,nfft: Optional[None]=None ):

        if not self.isLoaded:
                    self.load()
                    raise Warning("Sliding window computed with default parameters !")

        frame = self.getAudioFrame(index=index)
        N = frame.shape[1]
        if nfft is not None:
            if nfft > N:
                zp = np.zeros((self.nCh,nfft-N))
                frame = np.hstack((frame,zp))
            elif nfft < N:
                nfft=N
                print("WARNING : NFFT is smaller than frame size. Set NFFT to frame.shape[1]")
            else:
                nfft=N
        else:
            nfft=N
        
        stft = np.fft.fft(frame,axis=1)
        freq = np.linspace(0,self.fs/2,nfft)

        return stft, freq

        
if __name__ == "__main__":
    wav_dir = "../../03_DATA/AMI/"
    audio_names = []
    for i in range(8):
        audio_names.append(f"IS1000a.Array1-0{i+1}")
    
    proc = WaveProcessorSlidingWindow(wav_dir=wav_dir,
                                      audio_names=audio_names)


    winlen = 2048
    winshift = 1024
    index = 10

    proc.load(winlen=winlen,shift=winshift)

    fs = proc.getFs()
    audio = proc.getAudio()
    frame = proc.getAudioFrame(index)
    nfft = 2048
    stft, freq = proc.getAudioFrameSTFT(index,nfft=nfft)

    print(audio_names)
    print(audio.shape)
    print(frame.shape)
    print(stft.shape)
