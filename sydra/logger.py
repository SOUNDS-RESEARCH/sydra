import numpy as np
import os
import soundfile
import pandas as pd
import json
import yaml

from omegaconf import OmegaConf
from omegaconf import DictConfig, ListConfig
from pathlib import Path

METADATA_FILENAME = "metadata.{}"


def save_signals(signals, sr, output_dir):
    
    for i, signal in enumerate(signals):
        file_name = output_dir / f"{i}.wav"

        soundfile.write(file_name, signal.T, sr)


def save_dataset_metadata(dataset_sample_configs, output_dir, mode):
    if mode == "csv":
        save_metadata_as_csv(dataset_sample_configs, output_dir)
    elif mode == "json":
        save_metadata_as_json(dataset_sample_configs, output_dir)
    elif mode == "yaml":
        save_metadata_as_yaml(dataset_sample_configs, output_dir)
    else:
        raise ValueError(f"Invalid metadata_format: {mode}")


def save_metadata_as_csv(dataset_sample_configs, output_dir):
    output_dicts = [_serialize_dict_csv(d) for d in dataset_sample_configs]

    df = pd.DataFrame(output_dicts)
    df.to_csv(output_dir / METADATA_FILENAME.format("csv"))


def save_metadata_as_json(dataset_sample_configs, output_dir):
    output_dicts = [_serialize_dict(d) for d in dataset_sample_configs]
    fn = output_dir / METADATA_FILENAME.format("json")
    with open(fn, "w") as file:
        json.dump(output_dicts, file, indent=4)


def save_metadata_as_yaml(dataset_sample_configs, output_dir):
    output_dicts = [_serialize_dict(d) for d in dataset_sample_configs]

    fn = output_dir / METADATA_FILENAME.format("yaml")
    with open(fn, "w") as file:
        yaml.dump(output_dicts, file)


def _serialize_dict(d):
    serialized_dict = {}
    for key, value in d.items():
        if isinstance(value, Path):
            serialized_dict[key] = str(value)
        elif isinstance(value, np.ndarray):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, (ListConfig, DictConfig)):
            serialized_dict[key] = OmegaConf.to_object(value)
        elif isinstance(value, (DictConfig, dict)):
            # Recursively serialize
            serialized_dict[key] = _serialize_dict(value)
        else:  # Int, float, str, list...
            serialized_dict[key] = value
    return serialized_dict


def _serialize_dict_csv(d):
    serialized_dict = {}
    for key, value in d.items():
        if isinstance(value, (int, float, str)):
            serialized_dict[key] = value
        elif isinstance(value, (Path, list, ListConfig)):
            serialized_dict[key] = str(value)

    return serialized_dict
