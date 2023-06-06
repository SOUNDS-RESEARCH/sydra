import itertools
import random

from adhoc40_dataset.settings import LOUDSPEAKER_POSITIONS, N_MICROPHONES


def create_microphone_and_loudspeaker_combinations(
    microphone_groups=None, n_mics=None, split_mode="all",
    random_seed=0, n=None):
    """The LibriAdhoc40 dataset contains 40 microphones and 4 loudspeaker positions.
       There are millions of ways of subselecting microphones and loudspeakers from this dataset. 
       We implement two sampling modes.
       
       The first mode requires the variable microphone_groups to be a list of lists, where
       every list contains some microphone ids. The funcion will generate all combinations containing one microphone
       per group, as well as one loudspeaker.
       This is useful for when selecting one microphone for every corner of the room, for example.

       The second mode only requires the number of microphones we want for each combination. It does an "n-choose-k"
       combination of the 40 microphones available, and then combines it with all the available source positions.  
    """
    
    if microphone_groups is not None:
        groups = [LOUDSPEAKER_POSITIONS] + list(microphone_groups)
        combinations = list(itertools.product(*groups))
    elif n_mics is not None:
        mic_idxs = range(1, N_MICROPHONES + 1)
        mic_combinations = itertools.combinations(mic_idxs, n_mics)
        combinations = list(itertools.product(LOUDSPEAKER_POSITIONS, mic_combinations))

    random.seed(random_seed)
    random.shuffle(combinations)

    if n is not None:
        combinations = combinations[:n]

    return combinations
