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
    data = data[: ((data.shape[0] - win_size) // win_shift)
                * win_shift + win_size, :]

    # Manage padding
    c = (context, ) + (sig.ndim - 1) * ((0, 0), )
    _win_size = win_size + sum(context)
    shape = (int((sig.shape[0] - win_size) /
                 win_shift) + 1, 1, _win_size, sig.shape[1])
    strides = tuple(
        map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
    print(f"strides = {strides}")
    if pad == 'zeros':
        return np.lib.stride_tricks.as_strided(np.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                               shape=shape,
                                               strides=strides).squeeze()
    elif pad == 'edge':
        return np.lib.stride_tricks.as_strided(np.lib.pad(data, c, 'edge'),
                                               shape=shape,
                                               strides=strides).squeeze()


class WaveProcessor:
    """
    Read one or multiple audio files (of the same length). Then, user can get the entire audio
    or only one frame between two time instant, or the Short Time Fourier Tranform of one frame
    """

    def __init__(self,
                 wav_dir,
                 audio_names):
        """
        Ctor
        Reads audio_names audio files from wav_dir and store them in a np array
        Note : audio files should be the same length

        :param wav_dir: path to the wave files directory
        :param audio_names: names of each audio file (stored in a list)
        """
        super().__init__()

        self.wav_dir = wav_dir
        self.audio_names = audio_names
        self.isLoaded = False

    def load(self):
        """
        Load audio file from file names
        """
        # case of one channel per file
        if len(self.audio_names) > 1:
            audio = []
            for name in self.audio_names:
                fname = f"{self.wav_dir}{name}.wav"
                tmp_, fs = soundfile.read(fname)
                audio.append(tmp_)
            audio = np.array(audio, dtype="float64").T
            self.nCh = len(self.audio_names)
        # case of multichannel audio file
        else:
            fname = f"{self.wav_dir}{self.audio_names[0]}.wav"
            nfo = soundfile.info(fname)
            self.duration = nfo.duration
            audio, fs = soundfile.read(fname)
            self.nCh = nfo.channels

        self.fs = fs
        self.data = audio
        self.isLoaded = True

    def getAudio(self):
        """
        Returns the entire audio signal as a np array and sampling rate.
        The audio signal is formated as a 2D array to handle multiple channels case
        """
        if not self.isLoaded:
            self.load()

        return self.data, self.fs

    def getAudioFrame(self,
                      start,
                      stop):
        """
        Returns one frame of the audio signal between two time instants

        :param start: start time (in sec)
        :param stop: stop time (in sec)
        """
        if not self.isLoaded:
            self.load()

        return self.data[int(np.ceil(start*self.fs)):int(np.ceil(stop*self.fs)),:]

    def getAudioFrameSTFT(self,
                          start,
                          stop,
                          nfft=None):
        """
        Returns one frame of the audio signal in the STFT domain

        :param start: start time (in sec)
        :param stop: stop time (in sec)
        :param nfft: number of samples for FFT calculation (if None, the length of the frame will be used)
        """

        frame = self.getAudioFrame(start, stop)
        N = frame.shape[0]
        if nfft is not None:
            if nfft >= N:
                zp = np.zeros((self.nCh, nfft-N)).T
                frame = np.vstack((frame, zp))
            elif nfft < N:
                nfft = N
                raise Warning(
                    "NFFT is smaller than frame size. Set NFFT to frame.shape[1]")
        else:
            nfft = N

        stft = np.fft.fft(frame, axis=0)
        freq = np.linspace(0, self.fs/2, nfft)

        return stft, freq

    def getFs(self):
        return self.fs


class WaveProcessorSlidingWindow(WaveProcessor):
    """
    Process audio file by exracting sliding windows
    The load method from WaveProcessor is overridden to allow windows extractions
    """

    def load(self,
             winlen: Optional[int] = 16000,
             shift: Optional[int] = 16000,
             avoid_null: Optional[bool] = False):
        """
        Load audio file and extract sliding windows from it as a 3-d numpy array

        :param winlen: length of the window in samples (default : 16000)
        :param shift: shift of the window in samples (default : 16000 - No overlap)
        :param avoid_null: add a very low gaussian noise to the signal to avoid null values in the signal
        """

        WaveProcessor.load(self)
        if avoid_null:
            self.data += 1e-6* np.random.randn(self.data.shape[0],self.data.shape[1])
        #tmp_ = framing(sig=self.data,
        #               win_size=winlen,
        #               win_shift=shift)
        tmp_ = np.lib.stride_tricks.sliding_window_view(self.data,
                                                        window_shape=winlen,
                                                        axis=0)
        self.data_sw = tmp_[::shift,:,:]
        self.shift = shift
        self.winlen = winlen
        #self.fs = fs
        #self.isLoaded = True

    def getAudio(self):
        """
        Return audio file as a 3-d numpy array (num channel x winlen x num win)
        """
        if not self.isLoaded:
            self.load()
            raise Warning("Sliding window computed with default parameters !")

        return self.data_sw

    def getAudioFrame(self, index):
        """
        Return a given frame from extracted sliding windows 

        :param index: index of the frame to be extracted
        """

        if not self.isLoaded:
            self.load()
            raise Warning(
                "Sliding window computed with default parameters !")

        return self.data_sw[index, :, :].T

    def getAudioFrameSTFT(self, index, nfft: Optional[None] = None):
        """
        Return a given frame in the STFT domain

        :param index: index of the frame to be returned
        :param nfft: number of samples for FFT calculation (default : None)
        """

        if not self.isLoaded:
            self.load()
            raise Warning(
                "Sliding window computed with default parameters !")

        frame = self.getAudioFrame(index=index)
        N = frame.shape[0]
        if nfft is not None:
            if nfft > N:
                zp = np.zeros((self.nCh, nfft-N))
                frame = np.hstack((frame, zp))
            elif nfft < N:
                nfft = N
                print(
                    "WARNING : NFFT is smaller than frame size. Set NFFT to frame.shape[1]")
            else:
                nfft = N
        else:
            nfft = N

        stft = np.fft.fft(frame, axis=0)

        return 2*stft[0:nfft//2+1,:]/nfft, nfft

    def isLoad(self):
        return self.isLoaded

    def __len__(self):
        return self.data_sw.shape[0]

    def timeSupport(self):
        N=self.__len__()
        return np.arange(start=0.0,stop=N*(self.shift),step=self.shift)/self.fs


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    wav_dir = "./data/Synth_data/output/CIRC/"
    audio_names = ["IS1000a_T23_nch8_snrinf_ola1_noise0"]
#    for i in range(8):
#        audio_names.append(f"IS1000a.Array1-0{i+1}")

    proc = WaveProcessor(wav_dir=wav_dir,
                         audio_names=audio_names)

    audio, fs = proc.getAudio()
    frame = proc.getAudioFrame(2.0, 2.005)
    nfft = 1024
    stft, freq = proc.getAudioFrameSTFT(0, 128./16000., nfft=nfft)

    print(audio_names)
    print(audio.shape)
    print(frame.shape)
    print(stft.shape)

    proc2 = WaveProcessorSlidingWindow(wav_dir=wav_dir,
                                       audio_names=audio_names)

    winlen = 512
    winshift = winlen//2
    index = 100

    proc2.load(winlen=winlen, shift=winshift)

    fs = proc2.getFs()
    audio = proc2.getAudio()
    frame = proc2.getAudioFrame(index)
    stft, nfft = proc2.getAudioFrameSTFT(index)

    freq = np.linspace(0,fs/2,nfft//2+1)

    print(audio_names)
    print(audio.shape)
    print(frame.shape)
    print(stft.shape)

    plt.figure(num=1)
    plt.plot(frame)
    plt.xlabel('samples')
    plt.ylabel('Amplitude')
    plt.title('Time')
    plt.grid()
    plt.show(block=False)

    plt.figure(num=2)
    plt.plot(freq,20*np.log10(abs(stft)))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('10*log10(|X|)')
    plt.title("STFT")
    plt.grid()
    plt.show(block=True)