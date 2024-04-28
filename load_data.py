import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import spectrogram
import os
import mat73

class Signal:
    def __init__(self, filename): 
        self.mat = mat73.loadmat(filename)
        self.signal = self.mat['signal']

    
    @property
    def data(self):
        return self.signal['data'].item()