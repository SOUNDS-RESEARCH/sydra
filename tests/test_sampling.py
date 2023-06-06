from ..adhoc40_dataset.sampling import create_microphone_and_loudspeaker_combinations


def test_create_microphone_and_loudspeaker_combinations_mode_1():
    combinations = create_microphone_and_loudspeaker_combinations(n_mics=4)
    assert len(combinations) == 4*91390 # 4 loudspeakers and (40 choose 4)=91390 microphone combinations 


def test_create_microphone_and_loudspeaker_combinations_mode_2():

    microphone_groups = [
    [40, 32, 24, 16],
    [8, 7, 6, 5, 4, 3, 2],
    [1, 9, 17, 25],
    [39, 38, 37, 36, 35, 34, 33]
    ]
    combinations = create_microphone_and_loudspeaker_combinations(microphone_groups=microphone_groups)
    
    assert len(combinations) == 4*(4*7*4*7) # 4 loudspeakers times the length of each group
