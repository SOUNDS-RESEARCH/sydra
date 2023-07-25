import numpy as np
import random

from .coordinates import generate_mic_and_source_coords
from .source_signals import (
    generate_interference_signals,
    generate_random_signal,
    generate_random_speech_signal
)
from .surface_absorptions import generate_random_surface_absorption


def generate_random_config_for_static_source(base_config):
    """Generate a random configuration for a static source scenario
    """

    source_coordinates = base_config["sources"]["source_coordinates"]

    # 1. Generate random room properties
    room_dims, surface_absorptions, rt60 = _generate_random_room_config(base_config)

    # 2. Generate random microphone and source locations
    (mic_coordinates,
     array_centres,
     source_coordinates,
     interferer_coordinates) = generate_mic_and_source_coords(
                room_dims,
                base_config["mics"],
                base_config["sources"]["source_coordinates"],
                base_config["sources"]["n_sources"],
                base_config["n_interferers"],
                base_config["min_wall_distance"],
                base_config["min_mic_source_dist"],
                base_config["default_device_height"]
    )

    # 3. Add asynchronous behaviour to microphones
    mic_delays, mic_sampling_rate_offsets, mic_gains = \
        generate_random_microphone_signal_degradations(base_config["mics"])

    # 4. Generate random source signals
    source_signals = _generate_random_source_signals(base_config, mic_delays)

    # 5. Generate interference signals at desired SNRs
    interference_signals, interferer_snr = generate_interference_signals(
                                                base_config["n_interferers"],
                                                base_config["interferers_snr_range"],
                                                source_signals
    )

    return {
        "room_dims": room_dims,
        "source_coordinates": source_coordinates,
        "mic_coordinates": mic_coordinates,
        "interferer_coordinates": interferer_coordinates,
        "mic_array_centres": array_centres,
        "mic_delays": mic_delays,
        "mic_gains": mic_gains,
        "mic_sampling_rate_offsets": mic_sampling_rate_offsets,
        "rt60": rt60,
        "surface_absorptions": surface_absorptions,
        "source_signals": source_signals,
        "interferer_signals": interference_signals,
        "interferer_snr": interferer_snr
    }


def _generate_random_room_config(base_config):
    """Generate random configuration for a shoebox room,
    given a base_config dictionary
    """

    room_dims = base_config["room_dims"]

    if len(np.array(room_dims).shape) == 2:
        # Each room coordinate represents an interval range to be sampled from 
        room_dims = [
            random.uniform(room_dims[0][0], room_dims[0][1]),
            random.uniform(room_dims[1][0], room_dims[1][1]),
            random.uniform(room_dims[2][0], room_dims[2][1])
        ]

    # Define absorption coefficients of the walls or reverberation time
    surface_absorptions = rt60 = None
    if base_config["anechoic"]:
        pass
    elif base_config["use_reflectivity_biased_sampling"]:
        surface_absorptions = generate_random_surface_absorption()
    else:
        rt60 = random.uniform(base_config["rt60"][0],
                        base_config["rt60"][1])

    return room_dims, surface_absorptions, rt60


def generate_random_microphone_signal_degradations(mic_config):
    n_mics = mic_config["n_mics"]
    delay_range = mic_config["mic_delay_ranges"]
    sr_offsets = mic_config["mic_sampling_rate_offsets"]
    gain_range = mic_config["mic_gain_ranges"]

    if mic_config["mic_type"] != "single":
        n_array = mic_config["n_array"]
    else:
        n_array = 1

    if delay_range is None:
        mic_delays = 0
    else:
        mic_delays = np.concatenate([
            np.random.uniform(delay_range[0], delay_range[1], n_array)
            for _ in range(n_mics)
        ])

    if sr_offsets is None:
        sampling_rate_offsets = 0
    else:
        sampling_rate_offsets = np.concatenate([
            np.random.uniform(sr_offsets[0], sr_offsets[1], n_array)
            for _ in range(n_mics)
        ])
    
    if gain_range is None:
        mic_gains = 1
    else:
        if type(gain_range[0]) in (int, float):
            mic_gains = np.concatenate([
                np.random.uniform(gain_range[0], gain_range[1], n_array)
                for _ in range(n_mics)
            ])
        else: # List of tuples
            mic_gains = np.concatenate([
                np.random.uniform(gain_range[i][0], gain_range[i][1], n_array)
                for i in range(n_mics)
            ])

    return mic_delays, sampling_rate_offsets, mic_gains


def _generate_random_source_signals(base_config, mic_delays):
    # Make signal's duration bigger to trim silent beginning
    if mic_delays == 0: # No async delay is simulated
        max_delay = 0
    else:
        max_delay = max(mic_delays)
    total_duration = base_config["signal_duration_in_seconds"] + max_delay
    sr = base_config["base_sampling_rate"]

    source_signals = []

    for i in range(base_config["sources"]["n_sources"]):
        if "speech_samples" in base_config["sources"]:
            source_signal = generate_random_speech_signal(
                total_duration, sr, base_config["sources"]["speech_samples"])
        else:
            source_signal, gain = generate_random_signal(int(sr*total_duration))

        source_signals.append(source_signal)
    
    source_signals = np.stack(source_signals)    
    return source_signals
