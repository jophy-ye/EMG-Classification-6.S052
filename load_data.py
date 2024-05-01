import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import spectrogram
import os
import mat73

class Signal:
    def __init__(self, filename):
        try: 
            self.mat = scipy.io.loadmat(filename) #104 files
            self.signal = self.mat['signal']
            self.mat['signal'] = {'data': self.signal["data"].item()}
        except:
            self.mat = mat73.loadmat(filename) #20 files with extra meta data
        self.signal = self.mat['signal']

    
    @property
    def data(self):
        return self.signal['data']