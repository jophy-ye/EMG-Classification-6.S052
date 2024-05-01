from torch.utils.data import Dataloader, Dataset
import torch
from load_data import Signal, Label
import os, glob

SEED = 0
torch.manual_seed(SEED)

class EMGDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.emg_samples = []
        self.labels = []

        with open("valid_files.py", "r") as f:
            for emg_file in f:
                self.emg_samples.append(Signal(emg_file).data)
                self.labels.append(Label(os.path.basename(emg_file).task))

    def __len__(self):
        return len(self.emg_samples)
    
    def __getitem__(self, idx):
        emg_sample_path = self.emg_samples[idx]
        emg_sample = Signal(emg_sample_path)
        label = self.labels[idx]
        if self.transform:
            spectogram = self.transform(spectogram)
        
        return spectogram, label

data_transforms = []