import librosa
import numpy as np
import random


def generate_random_speech_signal(signal_duration, sr, speech_file_paths):
    total_duration_in_samples = int(signal_duration*sr)
    random_file_path = random.choice(speech_file_paths)
    source_signal, _ = librosa.load(random_file_path, sr=sr)
    # Improvement: Selecting the starting sample randomly instead of using 0
    start_idx = random.randint(0, source_signal.shape[0] - total_duration_in_samples)
    source_signal = source_signal[start_idx:start_idx+total_duration_in_samples]

    return source_signal


def generate_random_signal(n_samples:int, random_gain=True):
    """Generate a random signal to be emmited by the source.
    The signal is white gaussian noise distributed.
    The signal is also multiplied by an uniformly distributed gain to simulate
    the unknown source gain.
    """
    
    gain = np.random.uniform() if random_gain else 1
    source_signal = np.random.normal(size=n_samples)*gain

    return source_signal, gain


def generate_interference_signals(n_interferers, snr_range, source_signals):
    if n_interferers == 0:
        return [], None

    target_snr = random.uniform(snr_range[0], snr_range[1])

    interference_signals = np.random.randn(
                            n_interferers,
                            source_signals[0].shape[0])

    # Original SNR without scaling the interference signals
    snr_0 = _snr_db(source_signals, interference_signals)

    scaling_factors = 10**((snr_0 - target_snr)/20)

    interference_signals *= scaling_factors[:, np.newaxis]
    return interference_signals, target_snr


def _snr_db(signals, noises):
    "Compute the snr between signals and multiple noise sources"

    # Select first source as reference
    ref_signal = signals[0]
    sig_power = np.mean(ref_signal**2)
    
    noise_powers = np.mean(noises**2, axis=1)

    snr = 10*np.log10(sig_power/noise_powers) 

    return snr
