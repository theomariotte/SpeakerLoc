import numpy as np
import logging
from typing import Optional

class MicArray:
    """
    Microphone array object
    """
    #def __init__(self,x_pos,y_pos,z_pos):
    #    super().__init__()
    #    self.x = x_pos
    #    self.y = y_pos
    #    self.z = z_pos

    def __init__(self,micropnts):
        super().__init__()
        if not len(micropnts.shape) == 2:
            raise Exception("Wrong microphone array definition !")
        
        self.x = micropnts[:,0]
        self.y = micropnts[:,1]
        self.z = micropnts[:,2]
    
    def getDistance(self,ref_mic_idx: Optional[int]=0):
        """
        Returns the distance between each microphone and the reference one
        """
        d = np.zeros(self.x.shape)
        idx = 0
        ref_x = self.x[ref_mic_idx]
        ref_y = self.y[ref_mic_idx]
        ref_z = self.z[ref_mic_idx]
        for x_i,y_i,z_i in zip(self.x,self.y,self.z):
            d[idx] = np.sqrt( (x_i-ref_x)**2 + (y_i-ref_y)**2 + (z_i-ref_z)**2 )
            idx+=1
        
        return d

    def getCoordinates(self):
        """
        Returns microphones coeerdinates over each axis of the frame
        """
        return self.x, self.y, self.z
    
    def micNumber(self):
        """
        Returns the number of microphones in the array
        """
        return self.x.shape[0]

    def getInterMicDelay(self,
                         fs,
                         ref_mic_idx: Optional[int]=0,
                         c0: Optional[float]=343.0):
        """
        Computes delay between reference microphone and others

        :param ref_mic_idx: index of the reference microphone (default 0)
        :param fs: sampling rate [Hz]
        :param c0: speed of sound [m/s] (default : c0=343.0 m/s)
        """

        if ref_mic_idx >= self.micNumber():
            ref_mic_idx = self.micNumber() - 1
        
        dist = self.getDistance(ref_mic_idx=ref_mic_idx)
        tdoa = dist / c0

        return tdoa



    def getSpatialCoherence(self,
                            idx_freq, 
                            num_freq, 
                            fs,
                            c0: Optional[float]= 343.0,
                            mode: Optional[str]="sinc"):
        """
        Computes the spatial coherence matrix between each microphones of the array 

        :param idx_freq: frequency index to be considered for the matrix computation
        :param num_freq: number of frequency index in the frequency domain
        :param fs: sampling rate
        :param c0: speed of sound in the medium (default = 343 m/s - air at 20°C)
        :param mode: spatial repartition mode (default="sinc" - sinus cardinal)
        """

        nbMic = self.micNumber()
        B = np.empty((nbMic,nbMic))

        if mode == "sinc":
            for ii in range(nbMic):
                for jj in range(nbMic):
                    d_ij = np.sqrt(
                        (self.x[jj]-self.x[ii])**2 + (self.y[jj]-self.y[ii])**2 + (self.z[jj]-self.z[ii])**2)
                    jwt = 2. * np.pi * idx_freq * d_ij * fs / num_freq / c0
                    B[ii, jj] = np.sinc(jwt)
        else:
            raise NotImplementedError(f"Mode <{mode}> does not exist !")

        return B


if __name__ == "__main__":
    x_vec = np.array([0.,-1.,0.,1.])
    y_vec = np.array([1.,0.,-1.,0.])
    z_vec = np.ones(4)
    X = np.array([x_vec,y_vec,z_vec]).T
    arr = MicArray(X)
    dist = arr.getDistance(0)
    ii=0
    for d in dist:
        logging.critical(f"Dist mic {ii} : {d}")
        ii+=1

    B = arr.getSpatialCoherence(idx_freq=1,num_freq=129,fs=16000.)
    print(B)
