import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import spectrogram
import os


class Signal:
    def __init__(self, filename): 
        self.mat = scipy.io.loadmat(filename)
        self.signal = self.mat['signal']
    
    @property
    def data(self):
        return self.signal['data'].item()