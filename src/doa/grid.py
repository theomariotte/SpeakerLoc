import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from microphone_array import MicArray
class Grid2D:

    def __init__(self,
                 x_loc,
                 y_loc):

        self.X,self.Y = np.meshgrid(x_loc,y_loc)

    def display(self,
                block=True,
                plt_duration=2.0):
        fig = plt.figure()
        ax = plt.plot(self.X,self.Y,'bo')
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Sources grid")
        plt.show(block=block)
        if not block:
            plt.pause(plt_duration)
            plt.close(fig=fig)

class CircularGrid2D:

    def __init__(self,
                 theta_start,
                 theta_stop,
                 theta_step: Optional[float] = 5.,
                 radius: Optional[float] = 0.5):

        super().__init__()

        theta = np.array(range(int(theta_start),int(theta_stop),int(theta_step)))

        self.theta = (theta * np.pi/180.0).astype(np.float)
        self.src_num = theta.shape[0]
        
        if radius > 0:
            self.radius = radius
        else:
            raise Exception("Radius should be positive !")

        # cartesian coordinates
        self.X = self.radius * np.cos(self.theta)
        self.Y = self.radius * np.sin(self.theta)

    def getTDOA(self,
                mic_array,
                ref_mic_idx: Optional[int] = 0,
                c0: Optional[float] = 343.0):
        """
        Computes TDOA between each microphone and each virtual source

        :param mic_array: microphone array (as MicArray object) 
        :param ref_mic_idx: reference microphone index for relative delay calculation (default : 0)
        :param c0: speed of sound in m/s (default: 343.0)

        Return : matrix where each column is associated with one microphone of the array. 
        Each column contains TDOA between every source and the current microphone
        """
        if not isinstance(mic_array, MicArray):
            raise Exception(
                "<array> parameter should be a microphone array object !")

        # distance between reference mic and others in the array
        dist = mic_array.getDistance(ref_mic_idx)
    
        # compute TDOA for each microphone
        tdoa = dist[:, np.newaxis] * np.cos(self.theta[:,np.newaxis].T) / c0 
        #tdoa = np.matmul(dist[:, np.newaxis], self.theta[:, np.newaxis].T).T

        return tdoa.T

    def getRTF(self,
               freq,
               fs,
               array,
               reference_idx: Optional[int] = 0,
               c0: Optional[float] = 343.0):
        """
        Compute relative Transfert Function between each microphone and each source on the grid

        :param freq: np array containing frequencies where RTF is evaluated
        :param fs: sampling rate
        :param array: microphone array (as MicArray object) 
        :param reference_idx: reference microphone index for relative delay calculation (default : 0)
        :param c0: speed of sound in m/s (default: 343.0)
        """

        aliasing = np.where(freq > fs/2)[0]
        if aliasing.shape[0] > 0:
            raise Exception("Frequencies should respect Nyquist criterion !!!")

        tdoa = self.getTDOA(mic_array=array,
                            ref_mic_idx=reference_idx,
                            c0=c0)

        freq_num = len(freq)
        mic_num = array.micNumber()
        src_num = self.shape()[0]

        RTF = np.zeros((src_num, freq_num, mic_num),dtype=complex)

        for isrc in range(src_num):
            for imic in range(mic_num):
                jwt = -1j * 2 * np.pi * freq * \
                    tdoa[isrc, imic] * fs / freq_num
                RTF[isrc, :, imic] = np.exp(jwt)

        return RTF

    def shape(self):
        """
        Returns grid shape
        """
        return self.theta.shape

    def components(self):
        """
        Returns grid components
        """
        return {"radius": np.zeros(self.theta.shape) + self.radius, "theta": self.theta}


if __name__ == "__main__":

    start = 0
    step = 5
    stop = 360
    r = 0.6
    fs = 16e3
    Nf = 128
    ff = np.linspace(0, fs/2, Nf)

    grid = CircularGrid2D(theta_start=start,
                          theta_stop=stop,
                          theta_step=step,
                          radius=r)

    x_vec = np.array([0., -1., 0., 1.])
    y_vec = np.array([1., 0., -1., 0.])
    z_vec = np.zeros(4)

    mics = np.array([x_vec,y_vec,z_vec]).T
    arr = MicArray(mics)

    tdoa = grid.getTDOA(mic_array=arr)

    rtf = grid.getRTF(freq=ff,
                      fs=fs,
                      array=arr)
    tmp = grid.components()
    theta = tmp["theta"] * 180./np.pi
    T,F = np.meshgrid(theta,ff) 

    for i in range(rtf.shape[2]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.set_ylim(0,Ly)
        plt.pcolor(T.T,F.T,np.angle(rtf[:,:,i]),cmap="winter",shading="auto")
        plt.xlabel("DOA [°]")
        plt.ylabel("RTF phase (wrapped)")
        plt.show()


    print(rtf)
    print(rtf.shape)
