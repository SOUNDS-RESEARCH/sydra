# SYDRA - Synthetic Datasets for Room Acoustics

This project provides functionality for generating synthetic room acoustic datasets, which can be used for training and evaluating machine learning models. Furthermore, we take interest in asynchronous microphone networks, which may be also sometimes referred to as Wireless Acoustic Sensor Networks (WASNs). Every dataset sample is generated in two steps, firstly by simulating the acoustic propagation from the sound sources to all the microphones present in the room, and later by simulating the network propagation and microphone characteristics through the use of fractional delay filtering.

## Functionality
So far, Sydra is only able to generate datasets containing multiple microphones and sources. The parameters of these microphones may be either fixed or random, as well as the room dimensions and reverberation time (rt60). The input source signal may either be noise, or a directory containing signal samples (such as speech) may be provided.

We provide a SydraDataset class, a Pytorch Dataset, which allows you to easily load and train models using Pytorch.


## Installation
We recommend creating a virutal environment for this project.
If you are using Anaconda, this can be achieved by running the following command:

`conda create -n sydra python=3.8`

Then, activate the environment:

`conda activate sydra`

Then, install the requirements:

`pip install -r requirements.txt`

Finally, clone the package:

`git clone https://github.com/SOUNDS-RESEARCH/sydra --recurse-submodules`


The `--recurse-submodules` flag is necessary to clone the Pyroomasync submodule, a wrapper on top of [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics/) which facilitates generating behaviour
common to WASNs (Microphones with different sampling rates, gains, delay).

## Usage
You can configure your simulation in the `config/config.yaml` file. Most parameters in the config file have an inline comment describing it. Then, you can create a dataset by running:

`python main.py dataset_dir=/path/to/dataset n_samples=1000`,

where `dataset_dir` is the directory where the dataset will be saved, and `n_samples` is the number of samples to generate.

## Format

Every SydraDataset contains a `metadata.[csv|json|yaml]` file in its root, which contains the annotations for every dataset sample.
A SydraDataset also contains a `samples/` directory, which in turn contains many directories. Every one of such directories contains the recordings referring to each microphone.

### Author
Eric Grinstein, Imperial College London
Supervisor: Prof. Patrick A. Naylor
