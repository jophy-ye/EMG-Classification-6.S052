import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import spectrogram
import re
import mat73

class Signal:
    def __init__(self, filepath):
        try: 
            self.mat = scipy.io.loadmat(filepath) #104 files
            self.signal = self.mat['signal']
            self.mat['signal'] = {'data': self.signal["data"].item()}
        except:
            self.mat = mat73.loadmat(filepath) #20 files with extra meta data
        self.signal = self.mat['signal']

    
    @property
    def data(self):
        return self.signal['data']
    
class Label:
    def __init__(self, filename):
        self.subject, self.force, self.task = Label.parse_labels(filename)
    
    @staticmethod
    def parse_labels(filename):
        text_content = re.split(r'[_.]', filename)
        subject = int(text_content[0][1:])
        force = int(text_content[1])
        task = text_content[2]
        return subject, force, task