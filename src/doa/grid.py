import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from microphone_array import MicArray


class Grid2D(object):
    """
    Candidate sources grid in 2-D
    This class implements a given grid in cartesian coordinates (e.g. square grid)
    """

    def __init__(self,
                 x_loc,
                 y_loc):
        """
        Conctructor

        :param x_lox: location of the sources on the x axis
        :param y_loc: location of the sources on the y axis
        """
        super(Grid2D, self).__init__()
        if not len(x_loc) == len(y_loc):
            raise Exception(
                "<x_loc> and <y_loc> parameters should be the same dimensions !!!")

        self.X, self.Y = np.meshgrid(x_loc, y_loc)

    def display(self,
                block=True,
                plt_duration=2.0):
        """
        Dislay the grid on a 2-D graph

        :param block (optional): block the display or not (default=True)
        :param plt_duration (otpional): display time before closing the figure, only when block==False (default=2.0s)
        """

        fig = plt.figure()
        ax = plt.plot(self.X, self.Y, 'bo')
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Sources grid")
        plt.show(block=block)
        if not block:
            plt.pause(plt_duration)
            plt.close(fig=fig)


class CircularGrid2D(object):
    """
    Circular grid

    This class implements a cricular grid in a 2 dimensional space
    """

    def __init__(self,
                 theta_start,
                 theta_stop,
                 theta_step: Optional[float] = 5.,
                 radius: Optional[float] = 0.5):
        """
        Constructor

        :param theta_start: starting angle of the grid [deg]
        :param theta_stop: stopping angle of the grid [deg]
        :param theta_step (optional): angle step between each source candidate [deg] (default=5)
        :param radius(optional): radius of the grid [m] (default=0.5) 
        """
        super(CircularGrid2D, self).__init__()
        self.theta = np.arange(theta_start, theta_stop+theta_step,
                               theta_step, dtype=float)
        # convert angles to radians
        self.theta *= np.pi/180.0
        self.src_num = len(self.theta)

        if radius > 0:
            self.radius = radius
        else:
            raise Exception("Radius should be positive !")

        # cartesian coordinates
        self.X = np.expand_dims(self.radius * np.cos(self.theta), axis=1)
        self.Y = np.expand_dims(self.radius * np.sin(self.theta), axis=1)
        self.is_array_init = False

    def addMicArray(self, array):
        """
        Adds a microphone array to the grid object to allow transfert functions calculations.

        :param array: MicArray object already defined
        """
        if not isinstance(array, MicArray):
            raise Exception("<array> param should be a <MicArray> object !!!")

        self.array = array
        self.is_array_init = True

    def getTDOA(self,
                ref_mic_idx: Optional[int] = 0,
                c0: Optional[float] = 343.0):
        """
        Computes TDOA between each microphone and each virtual source

        :param ref_mic_idx: reference microphone index for relative delay calculation (default : 0)
        :param c0: speed of sound in m/s (default: 343.0)

        Return : - matrix where each column is associated with one microphone of the array. 
        Each column contains TDOA between every source and the current microphone
                 - number of microrophones
        """

        if not self.is_array_init:
            raise Exception(
                'No MicArray object has been initialized. Please call addMicArray() before !')

        # microphone coordinates
        x_mic, y_mic, _ = self.array.getCoordinates()

        #distance between each source and each microphone
        R = np.sqrt((x_mic[:, np.newaxis].T - self.X)**2 +
                    (y_mic[:, np.newaxis].T - self.Y)**2)
        tdoa = R/c0

        return tdoa,len(self.array)

    def getRDTF(self,
                freq,
                fs,
                freq_idx_vec: Optional[None] = None,
                reference_idx: Optional[int] = 0,
                c0: Optional[float] = 343.0):
        """
        Compute Relative Direct Transfert Function (RDTF) with respect to a reference microphone.
        If there is N microphones in the array, the output array shape will be (K,N-1) where K is 
        the number of frequencies to be considered. In fact, relative delay between reference microphone and itself
        is always null, so the RDTF is 1 for all frequencies. This RDTF is removed to avoid useless computations.

        :param freq: np array containing frequencies where RTF is evaluated [Hz]
        :param fs: sampling rate [Hz]
        :param array: microphone array (as MicArray object) 
        :param freq_idx_vec: array containing indexes of frequency to be used (default None - all frequencies in <freq> are considered)
        :param reference_idx: reference microphone index for relative delay calculation (default : 0)
        :param c0: speed of sound in m/s (default: 343.0)
        """

        # if frequency band is constrained
        if freq_idx_vec is not None:
            freq = freq[freq_idx_vec]

        # check Nyquist criterion
        aliasing = np.where(freq > fs/2)[0]
        if aliasing.shape[0] > 0:
            raise Exception("Frequencies should respect Nyquist criterion !!!")

        # normalized frequency (with respect to fs/2)
        freq_norm = freq/fs*2

        # compute TDOA between each source of the grid and each microphone of the array
        tdoa, mic_num = self.getTDOA(
            ref_mic_idx=reference_idx,
            c0=c0)

        # compute relative delay between microphones and remove delay associated with reference mic (always zero)
        rel_tdoa = np.expand_dims(tdoa[:, reference_idx], axis=1)-tdoa

        freq_num = freq_norm.shape[0]
        src_num = self.__len__()

        RDTF = np.zeros((src_num, freq_num, mic_num), dtype=complex)

        # pas propre ça...
        for isrc in range(src_num):
            for imic in range(mic_num):
                jwt = -1j * 2 * np.pi * freq_norm * rel_tdoa[isrc, imic] * fs
                RDTF[isrc, :, imic] = np.exp(jwt)

        return RDTF

    def getTF(self,
              freq,
              fs,
              freq_idx_vec: Optional[None] = None,
              reference_idx: Optional[int] = 0,
              c0: Optional[float] = 343.0):
        """
        Compute relative Transfert Function between each microphone and each source on the grid

        :param freq: np array containing frequencies where RTF is evaluated [Hz]
        :param fs: sampling rate [Hz]
        :param array: microphone array (as MicArray object) 
        :param freq_idx_vec: array containing indexes of frequency to be used (default None - all frequencies in <freq> are considered)
        :param reference_idx: reference microphone index for relative delay calculation (default : 0)
        :param c0: speed of sound in m/s (default: 343.0)
        """
        # working frequency band
        if freq_idx_vec is not None:
            freq = freq[freq_idx_vec]

        aliasing = np.where(freq > fs/2)[0]
        if aliasing.shape[0] > 0:
            raise Exception("Frequencies should respect Nyquist criterion !!!")

        # normalized frequency (with respect to sampling rate)
        freq_norm = freq/fs*2

        tdoa,mic_num = self.getTDOA(
                            ref_mic_idx=reference_idx,
                            c0=c0)

        freq_num = freq_norm.shape[0]
        src_num = self.__len__()

        TF = np.zeros((src_num, freq_num, mic_num), dtype=complex)
        
        # pas propre no plus..
        for isrc in range(src_num):
            for imic in range(mic_num):
                jwt = -1j * 2 * np.pi * freq_norm * tdoa[isrc, imic] * fs
                TF[isrc, :, imic] = np.exp(jwt)

        return TF
    
    def display(self):
        """
        Displays grid and microphoene array of one has been defined
        """

        fig = plt.figure(num=12)
        ax=plt.plot(self.X,self.Y,'ro')
        if self.is_array_init:
            ar_x,ar_y,_=self.array.getCoordinates()
            plt.plot(ar_x,ar_y,'kx')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Configuration')
        plt.axis('equal')
        plt.show()
       


    def __len__(self):
        """
        Length of the grid (i.e. number of sources)
        """
        return self.src_num

    def components(self):
        """
        Returns grid components
        """
        return {"radius": np.zeros(self.theta.shape) + self.radius, "theta": self.theta, "X": self.X, "Y": self.Y}



if __name__ == "__main__":

    start = 0
    step = 10
    stop = 360
    r = 1.0
    fs = 16000
    #number of frequencies where to evaluate transfert function
    Nf = 128
    ff = np.linspace(0, fs/2, Nf)

    toplot="rdtf"

    grid = CircularGrid2D(theta_start=start,
                          theta_stop=stop,
                          theta_step=step,
                          radius=r)
    """
    x_vec = np.array([-1,-0.5,0,0.5,1])*1e-2
    #x_vec = np.linspace(-10,10,50)
    y_vec = np.zeros_like(x_vec)
    z_vec = np.ones_like(x_vec)
    """
    
    radius = 0.1
    x_vec = radius*np.array([1, 0.0, -1, 0.0])
    y_vec = radius*np.array([0.0, 1, 0.0, -1])
    z_vec = np.zeros(4)
    
    mics = np.array([x_vec, y_vec, z_vec]).T
    arr = MicArray(mics)

    # add microphone array to the grid
    grid.addMicArray(arr)

    # get time delay of arrival between each source and each microphone 
    # with default parameters
    tdoa,nb_mic = grid.getTDOA()

    # get transfert function between each source and each microphone 
    # for a given set of frequencies (with default parameters)
    tf = grid.getTF(freq=ff,
                     fs=fs)
    
    # get the relative direct transfert function 
    rdtf = grid.getRDTF(freq=ff,
                        fs=fs,
                        reference_idx=0)

    # get components of the grid (i.e. corrdinates of the sources in each frame)
    tmp = grid.components()
    theta = tmp["theta"] * 180./np.pi
    T, F = np.meshgrid(theta, ff)

    # normalized frequency (between 0 and 1)
    ff_norm = ff/fs*2

    grid.display()

    # theoretical phase to check
    if toplot=="tf":
        for i in range(tf.shape[2]):
            tf_phase_th = np.zeros((tdoa.shape[0], ff.shape[0]))
            ii = 0
            for f in ff_norm:
                jj = 0
                for t in tdoa[:, i]:
                    tf_phase_th[jj, ii] = -2 * np.pi * f * t * fs
                    jj += 1
                ii += 1
            th_rtf = np.exp(1j*tf_phase_th)
            fig = plt.figure()
            plt.subplot(2, 2, 1)
            plt.title(f"Microphone {i+1} - TF phase")

            #ax.set_ylim(0,Ly)
            plt.pcolor(T.T, F.T, np.unwrap(
                np.angle(tf[:, :, i]), axis=1), cmap="winter", shading="auto")
            #plt.pcolor(T.T,F.T,np.angle(rtf[:,:,i]),cmap="winter",shading="auto")
            plt.ylabel("Frequency [Hz]")

            plt.subplot(2, 2, 2)
            plt.title(f"Microphone {i+1} - Theoretical TF phase")
            #ax.set_ylim(0,Ly)
            #plt.pcolor(T.T,F.T,np.angle(th_rtf),cmap="winter",shading="auto")
            plt.pcolor(T.T, F.T, np.unwrap(np.angle(th_rtf)),
                    cmap="winter", shading="auto")
            plt.colorbar()
            plt.xlabel("DOA [°]")
            plt.ylabel("Frequency [Hz]")

            plt.subplot(2, 2, 3)
            plt.title(f"Microphone {i+1} - TDOA")
            plt.plot(theta, tdoa[:, i], 'k-')
            plt.xlabel("DOA [°]")
            plt.ylabel("Delay")
            plt.subplot(2, 2, 4)
            plt.title("Array geometry")
            plt.plot(x_vec, y_vec, 'ko')
            plt.xlabel("x")
            plt.ylabel("y")

            plt.show(block=True)

    if toplot=="rdtf":
        for i in range(len(arr)):
            fig = plt.figure()
            plt.title(f"Microphone {i+1} - RDTF phase")
            #ax.set_ylim(0,Ly)
            plt.pcolor(T.T, F.T, np.unwrap(
                np.angle(rdtf[:, :, i]), axis=1), cmap="winter", shading="auto")
            #plt.pcolor(T.T,F.T,np.angle(rtf[:,:,i]),cmap="winter",shading="auto")
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("DOA [deg]")
            plt.show(block=True)

