import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import spectrogram
import os

mat = scipy.io.loadmat("S1_20_DF.otb+_decomp.mat_edited.mat")

signal = mat['signal']['data']

data = signal[0][0]    # 2d array

save_folder = "data_1"

for i, arr in enumerate(data):
    # Compute spectrogram for the current segment 'arr'
    f, t, Sxx = spectrogram(arr, fs=2000)  # Assuming a sampling frequency of 2000 Hz
    
    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Power/Frequency [dB/Hz]')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"spectrogram_{i}.png"))
    plt.close()
    # put into tensor

# for arr in data:
# arr = data[0]
# transformed_arr = scipy.signal.ShortTimeFFT(arr, 1, 1/2000)
# transformed_arr.spectrogram(arr)


# print(type(transformed_arr))
# f, t, Sxx = spectrogram(transformed_arr, 1/2000)

# plt.figure(figsize=(10, 6))
# plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading = 'gouraud')
# plt.tile('spectrogram')
# plt.tight_layout()
# plt.show()





# mat.keys()
# signal = mat['signal']
# type(signal)
# signal['path']




