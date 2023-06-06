import ast
import json
import os
import pandas as pd
import random
import torch
import torchaudio
import yaml

from functools import reduce
from pathlib import Path


class SydraDataset(torch.utils.data.Dataset):
    """Simple torch.Dataset class used to load signals and metadata
       from a dataset directory in the 'SYDRA' format 
    """
    def __init__(self, dataset_dir, sr=None,
                 trim_signals_duration=None, trim_signals_mode="end"):
        """Initialize the dataset

        Args:
            dataset_dir (str or Path): path to the dataset directory
            sr (float, optional): If not provided, will be inferred from data
            trim_signals_duration (float, optional): If provided, trim signals to this duration, in seconds.
            trim_signals_mode (str, optional): Mode of the trimming operation.
                                               "start" selects the first trim_signals_duration seconds of the signals, while
                                               "random" selects a random chunk of trim_signals_duration seconds out of the signals.
        """
        self.metadata = load_metadata(dataset_dir)
        self.sr = sr
        if sr is None:
            # If not provided, infer sampling rate from the dataset's first row
            self.sr = self.metadata[0]["sr"]

        self.n_mics = _infer_n_mics_in_dataset(self.metadata)
        
        self.trim_signals = trim_signals_duration is not None
        if self.trim_signals:
            self.trim_signals_duration_in_samples = int(self.sr*trim_signals_duration)
        self.trim_signals_mode = trim_signals_mode

    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        
        # 0. Load row containing signals' metadata
        sample_metadata = self.metadata[index]

        # 1. Load signals
        signal_file_names = sorted(os.listdir(sample_metadata["signals_dir"]))

        x = [
            _load_audio(
                sample_metadata["signals_dir"] / signal_file_name, self.sr)
            for signal_file_name in signal_file_names
        ]

        x = torch.stack(x, dim=0)
        # x.shape == (n_arrays, n_array, n_signal)

        # 1.1. (Optional) Trim signals
        if self.trim_signals:
            if self.trim_signals_mode == "start":
                start_idx = 0
                end_idx = self.trim_signals_duration_in_samples
            elif self.trim_signals_mode == "end":
                start_idx = x.shape[2] - self.trim_signals_duration_in_samples
                end_idx = x.shape[2] 
            elif self.trim_signals_mode == "random":
                max_end_idx = x.shape[2] - self.trim_signals_duration_in_samples
                start_idx = random.randint(0, max_end_idx)
                end_idx = start_idx + self.trim_signals_duration_in_samples

            x = x[:, :, start_idx:end_idx]

        return (x, sample_metadata)

    def __len__(self):
        return len(self.metadata)


def load_metadata(dataset_dir):
    "Load a single metadata file (see _load_metadata) or multiple ones and concatenate them."
    if type(dataset_dir) in [str, Path]:
        metadata = _load_metadata(dataset_dir)
    else: # Multiple datasets
        metadatas = [_load_metadata(d) for d in dataset_dir]
        metadata = reduce(lambda x,y: x + y, metadatas)

    return metadata


def _load_metadata(dataset_dir):
    "Load a single metadata file (csv, json or yaml)"
    dataset_dir = Path(dataset_dir)
    
    paths = {
        "csv": dataset_dir / "metadata.csv",
        "json": dataset_dir / "metadata.json",
        "yaml": dataset_dir / "metadata.yaml"
    }

    if os.path.exists(paths["csv"]):
        metadata = pd.read_csv(paths["csv"])
        metadata = list(metadata.T.to_dict().values())
    elif os.path.exists(paths["yaml"]):
        with open(paths["yaml"], "r") as f:
            metadata = yaml.load(f)
    elif os.path.exists(paths["json"]):
        with open(paths["json"], "r") as f:
            metadata = json.load(f)

    # Process dataset (convert lists to tensors, etc.)
    for row in metadata:
        _desserialize_lists_within_dict(row)
        row["n_mics"] = len(row["mic_coordinates"])
        row["signals_dir"] = dataset_dir / row["signals_dir"]

    return metadata


def _desserialize_lists_within_dict(d):
    """Lists were saved in pandas as strings.
       This small utility function transforms them into lists again.
    """
    for key, value in d.items():
        if type(value) == str:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass # Just keep it as a string
        
        if type(value) == list:
            value = torch.Tensor(value)
        
        # Update dict key
        d[key] = value


def _infer_n_mics_in_dataset(metadata):
    n_mics = set()

    for m in metadata:
        n_mics.add(m["n_mics"])
    
    n_mics = list(n_mics)
    if len(n_mics) == 1:
        n_mics = n_mics[0]
    return n_mics


def _load_audio(path, target_sr, normalize=True):
    signal, sr = torchaudio.load(path)
    if sr != target_sr:
        signal = torchaudio.functional.resample(signal, sr, target_sr)

    if normalize:
        signal /= signal.abs().max()

    return signal
