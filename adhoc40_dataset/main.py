import librosa
import numpy as np
import os
import pandas as pd
import random
import soundfile
import hydra

from joblib import Parallel, delayed
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
from typing import List
from .sampling import create_microphone_and_loudspeaker_combinations

from .settings import (
    LOUDSPEAKER_POSITIONS, MIN_DURATION_IN_SECS,
    POSITIONS, RT60_IN_SECS, START_TRIM, END_TRIM, SR, ROOM_DIMS,
)


def sample_adhoc40_phrases(config):
    """randomly select loudspeaker and microphone positions,
    as well as the speech sample used from the LibriAdhoc40 dataset"""

    random.seed(config["random_seed"])

    # 0. Load metadata as a pandas dataframe
    metadata = _load_adhoc40_metadata(config["input_dir"])

    # 1. Get loudspeaker and microphone combinations to use
    combinations = create_microphone_and_loudspeaker_combinations(
                        microphone_groups=config["microphone_groups"],
                        n_mics=config["n_mics"],
                        split_mode=config["mode"],
                        random_seed=config["random_seed"],
                        n=config["n_samples"])

    def get_loudspeaker_phrases(loudspeaker_position_id):
        loudspeaker_phrases = filter(
            lambda x: x["loudspeaker_position"] == loudspeaker_position_id, metadata)
        return list(loudspeaker_phrases)
    loudspeaker_phrases = {
        loudspeaker_position_id: get_loudspeaker_phrases(loudspeaker_position_id)
        for loudspeaker_position_id in LOUDSPEAKER_POSITIONS
    }

    n_mics = len(combinations[0]) - 1 # Every combination has one loudspeaker position and n_mics microphones
    samples = []

    count = 0
    for combination in tqdm(combinations):
        loudspeaker_position = combination[0]
        n_microphones = combination[1]
        
        for i in range(config["n_phrases_per_positioning"]):
            phrases = loudspeaker_phrases[loudspeaker_position]

            while True: # Loop until finding a signal with a minimal duration
                mic_str = '-'.join(str(x) for x in n_microphones)
                sample_id = f"{loudspeaker_position}-{mic_str}-{i}"
                sample = random.choice(phrases)
                phrase_name = os.path.basename(sample["path"])

                mic_paths = [
                    sample["path"] + "/" + f"{phrase_name}-ch-{n_mic}.wav"
                    for n_mic in n_microphones
                ]
                durations = [
                    librosa.get_duration(filename=mic_path)
                    for mic_path in mic_paths
                ]
                is_min_duration = list(filter(
                    lambda x: x >= MIN_DURATION_IN_SECS,
                    durations
                ))
                 
                if len(is_min_duration) < n_mics:
                    continue

                source_coordinates = POSITIONS[loudspeaker_position]
                mic_coordinates = [
                    POSITIONS[n_mic]
                    for n_mic in n_microphones
                ]

                sample_metadata = {
                    "source_coordinates": source_coordinates,
                    "mic_coordinates": mic_coordinates,
                    "sample_id": sample_id,
                    "mic_paths": mic_paths
                }
                # for i, mic_path in enumerate(mic_paths):
                #     sample_metadata[f"mic_{i}_path"] = mic_path

                samples.append(sample_metadata)

                break
   
    return samples


def create_dataset(samples: List[dict], config):
    dataset_dir = Path(config["dataset_dir"]) / config["mode"]
    os.makedirs(dataset_dir, exist_ok=True)

    def process(sample):
        sample_id = sample["sample_id"]
        signals_dir = f"samples/{sample_id}"
        os.makedirs(dataset_dir / signals_dir, exist_ok=True)

        mic_signals = np.stack([
            librosa.load(mic_path, sr=SR)[0]
            for mic_path in sample["mic_paths"]
        ])

        mic_signals = _trim_signal(mic_signals.T, START_TRIM, END_TRIM,
                                    config["signal_duration_in_seconds"], SR).T
        
        soundfile.write(dataset_dir / f"{signals_dir}/0.wav", mic_signals.T, SR) 

        # for i, mic_signal in enumerate(mic_signals):
        #     if len(mic_signal.shape) > 0:
        #         mic_signal = mic_signal.T # soundfile expects a NxChannels signal
        #     soundfile.write(dataset_dir / f"{signals_dir}/{i}.wav", mic_signal, SR) 

        return {
            "room_dims": ROOM_DIMS,
            "source_coordinates": sample["sources"]["source_coordinates"],
            "mic_coordinates": sample["mic_coordinates"],
            # "mic_delays": delays,
            # "mic_sampling_rate_offsets": sr_offsets,
            # "mic_gains": gains,
            "rt60": RT60_IN_SECS,
            "source_gain": 1,
            "signal_duration_in_seconds": config["signal_duration_in_seconds"],
            "mask_delay": False,
            "anechoic": False,
            "snr_in_db": None,
            "signals_dir": signals_dir,
            "sr": SR
        }

    df = Parallel(n_jobs=config["n_jobs"])(
        delayed(process)(sample)
        for sample in tqdm(samples))

    df = pd.DataFrame(df)
    df.to_csv(dataset_dir / "metadata.csv")


def _load_adhoc40_metadata(dataset_dir):
    "Load dataset metadata (such as path to audio signals)"
    samples = []
    for loudspeaker_position in LOUDSPEAKER_POSITIONS:
        loudspeaker_dir = dataset_dir + "/" + loudspeaker_position
        for speaker in os.listdir(loudspeaker_dir):
            speaker_dir = loudspeaker_dir + "/" + speaker
            if not os.path.isdir(speaker_dir):
                continue
            for chapter in os.listdir(speaker_dir):
                chapter_dir = speaker_dir + "/" + chapter
                if not os.path.isdir(chapter_dir):
                    continue
                for sentence in os.listdir(chapter_dir):
                    sentence_dir = chapter_dir + "/" + sentence
                    if not os.path.isdir(sentence_dir):
                        continue

                    samples.append({
                        "path": sentence_dir,
                        "loudspeaker_position": loudspeaker_position,
                        "speaker": speaker,
                        "chapter": chapter,
                        "sentence": sentence
                    })
    return samples


def _trim_signal(signal, start_trim, end_trim, duration=-1, sr=None):
    if sr is not None:
        start_trim = int(sr*start_trim)
        end_trim = int(sr*end_trim)
        duration = int(sr*duration)
    
    signal = signal[start_trim:-end_trim]

    if duration > 0:
        if signal.shape[0] < duration:
            return None
        signal = signal[:duration]  
    return signal


@hydra.main(config_path="config", config_name="adhoc40_dataset")
def main(config: DictConfig):
    print("Sampling dataset...")
    samples = sample_adhoc40_phrases(config)
    print("Saving signals...")
    create_dataset(samples, config)


if __name__ == "__main__":
    main()