from torch.utils.data import  Dataset
import torch
import numpy as np
from scipy.signal import spectrogram
from load_data import Signal, Label
import os, glob

SEED = 0
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
ELECRODES = 256
PAD_TARGET = 119296
X = 129
Y = 532
TESTTRAIN_SPLIT = {"train": "train_files.txt", "test": "test_files.txt"}
class EMGDataset(Dataset):
    def __init__(self, directory, format = "mat", mode="train", transform=None):
        """
        mode: "train" or "test"
        transfrom: apply transformation on emg_sample to transfrom to spectogram (last function must transfrom it to a spectogram)
        """
        self.directory = directory
        self.transform = transform
        self.emg_samples = []
        self.labels = []

        with open(TESTTRAIN_SPLIT[mode], "r") as f:
            for emg_file in f:
                emg_file = os.path.join(directory, emg_file.strip())
                self.emg_samples.append(Signal(emg_file, format=format).data)
                self.labels.append(int(Label(os.path.basename(emg_file)).task == 'DF')) #'DF'(1) and 'KE'(0)

    def __len__(self):
        return len(self.emg_samples)
    
    def __getitem__(self, idx):
        emg_sample = self.emg_samples[idx]
        if self.transform:
            for fn in self.transform:
                emg_sample = fn(emg_sample)
        spectogram = torch.tensor(emg_sample, dtype=torch.float32) # Note: maybe set datatype=torch.double (using datatype from output of spectrogram (dtype('float64')))
        label = self.labels[idx]
        
        return spectogram, label

def select_electrodes(dataset):
    electrodes, samples = dataset.shape
    return dataset[rng.choice(electrodes, size=ELECRODES, replace=False)]

def pad_data(dataset):
    electrodes, samples = dataset.shape
    pad = (PAD_TARGET - samples) // 2
    return np.pad(dataset, ((0,0), (pad,pad)))

def apply_spectrogram(dataset):
    fin = np.zeros((ELECRODES, X, Y))
    for (idx, row) in enumerate(dataset):
        _, _, sxx = spectrogram(row, fs=2048)
        fin[idx] = sxx
    return fin

data_transform = {'train': [select_electrodes, pad_data, apply_spectrogram], 
                  'test': [select_electrodes, pad_data, apply_spectrogram], }
