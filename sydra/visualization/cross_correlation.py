"""Plot the cross-correlation function between two or more signals.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import json

from scipy.signal import correlate, correlation_lags
from scipy.signal import find_peaks


def gcc_phat(signal1, signal2, abs=True, ifft=True, n_dft_bins=None):
    """Compute the generalized cross-correlation with phase transform (GCC-PHAT) between two signals.

    Parameters
    ----------
    signal1 : np.ndarray
        The first signal to correlate.
    signal2 : np.ndarray
        The second signal to correlate.
    abs : bool
        Whether to take the absolute value of the cross-correlation. Only used if ifft is True.
    ifft : bool
        Whether to use the inverse Fourier transform to compute the cross-correlation in the time domain,
        instead of returning the cross-correlation in the frequency domain.
    n_dft_bins : int
        The number of DFT bins to use. If None, the number of DFT bins is set to n_samples//2 + 1.
    """
    n_samples = len(signal1)

    if n_dft_bins is None:
        n_dft_bins = n_samples // 2 + 1

    signal1_dft = np.fft.rfft(signal1, n=n_dft_bins)
    signal2_dft = np.fft.rfft(signal2, n=n_dft_bins)

    gcc_ij = signal1_dft * np.conj(signal2_dft)
    gcc_phat_ij = gcc_ij / np.abs(gcc_ij)

    if ifft:
        gcc_phat_ij = np.fft.irfft(gcc_phat_ij)
        if abs:
            gcc_phat_ij = np.abs(gcc_phat_ij)

        gcc_phat_ij = np.concatenate((gcc_phat_ij[len(gcc_phat_ij) // 2:],
                                      gcc_phat_ij[:len(gcc_phat_ij) // 2]))

    return gcc_phat_ij
    
def plot_cross_correlation(signals, sr, plot_peaks=False, n_central_bins=64, output_path=""):
    if isinstance(signals, str):
        if os.path.isfile(signals):
            signals, sr = sf.read(signals)
        elif os.path.isdir(signals):
            signals_dir = signals
            signals = []
            for file in os.listdir(signals_dir):
                if file.endswith(".wav"):
                    signal, sr = sf.read(os.path.join(signals_dir, file))
                    signals.append(signal)
            if len(signals) == 1:
                signals = signals[0].T
            else:
                signals = np.stack(signals).transpose(1, 2)
        else:
            raise ValueError("The path provided is neither a file nor a directory.")
    elif not isinstance(signals, np.ndarray):
        raise TypeError("The signals provided must be either a path to a file or a numpy array.")
    
    if n_central_bins is None:
        n_central_bins = len(signals)//2
        
    n_signals, n_samples = signals.shape
    if n_signals < 2:
        raise ValueError("At least two signals must be provided.")
    
    num_peaks=[]
    x_corr = 1000*correlation_lags(n_samples, n_samples)/sr
    x_central = x_corr[len(x_corr)//2-n_central_bins//2:len(x_corr)//2+n_central_bins//2]
    x = np.arange(n_samples)/sr
    n_pairs = n_signals*(n_signals - 1)//2
    fig, axs = plt.subplots(n_pairs, 2, figsize=(10, 5))
    
    if n_pairs == 1:
        axs = np.expand_dims(axs, axis=0)

    n_pair = 0
    for i in range(n_signals):
        for j in range(i, n_signals):
            if i == j:
                continue
            # Plot correlation in the first column,
            corr = gcc_phat(signals[i], signals[j], abs=True, ifft=True, n_dft_bins=None)
            corr = corr[len(corr)//2-n_central_bins//2:len(corr)//2+n_central_bins//2]
            axs[n_pair, 0].plot(x_central, corr)
            max_corr_value = np.max(corr)
            threshold = max_corr_value / 1.5
            if threshold<0.03:
                threshold=0.03
            if plot_peaks:
                peaks, _ = find_peaks(corr, threshold=threshold)
                axs[n_pair, 0].plot(x_central[peaks], corr[peaks], "x", label="Peaks")
                axs[n_pair, 0].legend()
                num_peaks.append(len(peaks))
                

            # Plot the signals in the second column
            axs[n_pair, 1].plot(x, signals[i], label="Signal {}".format(i), alpha=0.5)
            axs[n_pair, 1].plot(x, signals[j], label="Signal {}".format(j), alpha=0.5)

            axs[n_pair, 0].set_title("Cross-correlation between signals {} and {}".format(i, j))
            axs[n_pair, 1].set_title("Signals {} and {}".format(i, j))
            axs[n_pair, 0].set_xlabel("Time (ms)")
            axs[n_pair, 0].set_ylabel("Correlation")
            axs[n_pair, 1].set_xlabel("Time (s)")
            axs[n_pair, 1].set_ylabel("Amplitude")
            
            axs[n_pair, 1].legend()

            n_pair += 1
    
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    # else:
    #    plt.show()
    plt.close()
    return num_peaks



def process_folder(config, folder_path, sr, plot_peaks=False, n_bins=512, output_path=""):
    with open(args.config, 'r') as f:
        config = json.load(f)
    num_sources = len(config[0]['source_coordinates'])  # Extract the number of sources from the config
    num_samples = 0
    num_peaks_count = {}
    num_peaks=[]
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            num_peaks= (plot_cross_correlation(subfolder_path, sr, plot_peaks, n_bins, output_path))
            for num_peak in num_peaks:
                if num_peak==num_sources:
                    num_samples+=1
                if num_peak in num_peaks_count:
                    num_peaks_count[num_peak]+=1
                else:
                    num_peaks_count[num_peak] = 1
    #prints the number of peaks and their frequency across the samples
    print(dict(sorted(num_peaks_count.items())))
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the cross-correlation function between two or more signals.")
    parser.add_argument("config", help="Path to the meta.json file.")
    parser.add_argument("folder", help="Path to the main folder containing subfolders with metadata.json and audio folders.")
    parser.add_argument("--sr", help="Sampling rate of the signals to be plotted.", default=16000, type=int)
    parser.add_argument("--plot_peaks", help="Plot the peaks of the cross-correlation function.", default=True, action="store_true")
    parser.add_argument("--n_bins", help="Number of bins to be plotted in the central part of the cross-correlation function.", default=256, type=int)
    parser.add_argument("--output_path", help="Output path to save the plots.", default="", type=str)
    args = parser.parse_args()

    process_folder(args.config, args.folder, args.sr, args.plot_peaks, args.n_bins, args.output_path)
