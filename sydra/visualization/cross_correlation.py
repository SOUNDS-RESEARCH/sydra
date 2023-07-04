"""Plot the cross-correlation function between two or more signals.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from scipy.signal import correlate, correlation_lags
from scipy.signal import find_peaks


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

            corr = np.abs(correlate(signals[i], signals[j]))
            corr = corr[len(corr)//2-n_central_bins//2:len(corr)//2+n_central_bins//2]
            axs[n_pair, 0].plot(x_central, corr)
            if plot_peaks:
                peaks, _ = find_peaks(corr)
                axs[n_pair, 0].plot(x_central[peaks], corr[peaks], "x", label="Peaks")
                axs[n_pair, 0].legend()

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
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the cross-correlation function between two or more signals.")
    parser.add_argument("signals", help="Path to a file or a directory containing the signals to be plotted.")
    parser.add_argument("--sr", help="Sampling rate of the signals to be plotted.", default=16000, type=int)
    parser.add_argument("--plot_peaks", help="Plot the peaks of the cross-correlation function.", action="store_true")
    parser.add_argument("--n_bins", help="Number of bins to be plotted in the central part of the cross-correlation function.", default=256, type=int)
    parser.add_argument("--output_path", help="Output path to save the plot at", default="", type=str)
    args = parser.parse_args()

    plot_cross_correlation(args.signals, args.sr, args.plot_peaks, args.n_bins, args.output_path)