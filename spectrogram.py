import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import spectrogram

SAVE_FOLDER = 'data_1'


class Spectrogram:
    def __init__(self, data):
        self.data = data

    def plot(self, save=False) -> None:
        for (i, arr) in enumerate(self.data):
            # Compute spectrogram for the current segment 'arr'
            f, t, Sxx = spectrogram(arr, fs=2048)   # calculated from dataset
            
            # Plot spectrogram
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.title('Spectrogram')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar(label='Power/Frequency [dB/Hz]')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            if save:
                plt.savefig(os.path.join(SAVE_FOLDER, f"spectrogram_{i}.png"),
                            bbox_inches='tight', pad_inches=0)
            plt.close()


from load_data import Signal
sig = Signal("S1_10_DF.otb%2B_decomp.mat_edited.mat")
a = Spectrogram(sig.data)
a.plot(save=True)