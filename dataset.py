from torch.utils.data import Dataloader, Dataset
import torch
from load_data import Signal

torch.manual_seed(0)
class EMGDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.emg_samples = []
        self.labels = []

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