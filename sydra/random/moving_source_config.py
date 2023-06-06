import numpy as np

from ..mic_array_geometries import MIC_ARRAY_GEOMETRIES


class RandomParameter:
    """ Random parammeter class.
    You can indicate a constant value or a random range in its constructor and then
    get a value acording to that with sample_value(). It works with both scalars and vectors.
    """

    def __init__(self, *args):
        if len(args) == 1:
            self.random = False
            self.value = np.array(args[0])
            self.min_value = None
            self.max_value = None
        elif len(args) == 2:
            self. random = True
            self.min_value = np.array(args[0])
            self.max_value = np.array(args[1])
            self.value = None
        else:
            raise Exception(
                'Parammeter must be called with one (value) or two (min and max value) array_like parammeters')

    def sample_value(self):
        if self.random:
            return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
        else:
            return self.value


def generate_random_config_for_moving_source(config):

    room_dims = RandomParameter(*_sydra_range_array_to_gpu(config["room_dims"]))
    rt60 = RandomParameter(*config["rt60"])
    
    absorption_weights = RandomParameter([0.5]*6, [1.0]*6) # fixed for now.
    mic_array_centre = RandomParameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5])
    snr = RandomParameter(config["mics"]["snr_in_db"])
    mic_array_geometry = MIC_ARRAY_GEOMETRIES[config["mics"]["mic_type"]]

    # 1. Generate a random room
    room_dims = room_dims.sample_value()
    rt60 = rt60.sample_value()
    absorption_weights = absorption_weights.sample_value()

    # 2. randomly place the microphone within the room
    mic_array_centre = mic_array_centre.sample_value() * room_dims
    mic_coordinates = mic_array_centre + mic_array_geometry.mic_coordinates

    # 4. Generate a random source trajectory
    src_pos_min = np.array([0.0, 0.0, 0.0])
    src_pos_max = room_dims
    
    # 4.1. Limit the source minimum and maximum positions to something related to the array
    # I didn't quite understand what's going on here.
    if mic_array_geometry.arrayType == 'planar':
        if np.sum(mic_array_geometry.orV) > 0:
            src_pos_min[np.nonzero(mic_array_geometry.orV)] = mic_coordinates[np.nonzero(
                mic_array_geometry.orV)]
        else:
            src_pos_max[np.nonzero(mic_array_geometry.orV)] = mic_coordinates[np.nonzero(
                mic_array_geometry.orV)]
    
    # np.random.random(3) generates a 3D array with elements between 0-1
    src_pos_ini = src_pos_min + \
        np.random.random(3) * (src_pos_max - src_pos_min)
    src_pos_end = src_pos_min + \
        np.random.random(3) * (src_pos_max - src_pos_min)

    # Maximum oscilation amplitude
    Amax = np.min(np.stack((src_pos_ini - src_pos_min,
                            src_pos_max - src_pos_ini,
                            src_pos_end - src_pos_min,
                            src_pos_max - src_pos_end)), axis=0)

    # Oscilations with 1m as maximum in each axis
    A = np.random.random(3) * np.minimum(Amax, 1)
    # Between 0 and 2 oscilations in each axis
    w = 2*np.pi / config["sources"]["n_trajectory_points"] * np.random.random(3) * 2

    traj_pts = np.array([
        np.linspace(i, j, config["sources"]["n_trajectory_points"])
        for i, j in zip(src_pos_ini, src_pos_end)]
    ).transpose()
    traj_pts += A * np.sin(w * np.arange(config["sources"]["n_trajectory_points"])[:, np.newaxis])

    if np.random.random(1) < 0.25:
        traj_pts = np.ones((config["sources"]["n_trajectory_points"], 1)) * src_pos_ini

    # Fix the height of all devices to the same value
    if config["default_device_height"] is not None:
        mic_coordinates[:, -1] = config["default_device_height"]
        mic_array_centre[-1] = config["default_device_height"]
        src_pos_ini[-1] = config["default_device_height"]
        src_pos_end[-1] = config["default_device_height"]
        traj_pts[:, -1] = config["default_device_height"]


    # Load source signal and VAD
    idx = np.random.randint(0, len(config["librispeech_dataset"]) - 1)
    
    source_signal, vad = config["librispeech_dataset"][idx]
    
    return {
        "room_dims": room_dims,
        "rt60": rt60,
        "absorption_weights": absorption_weights,
        "mic_coordinates": mic_coordinates,
        "mic_array_centres": mic_array_centre[np.newaxis],
        "snr_in_db": snr.sample_value(),
        "src_pos_ini": src_pos_ini,
        "src_pos_end": src_pos_end,
        "trajectory_points": traj_pts,
        "source_signals": source_signal[np.newaxis],
        "vad": vad
    }


def _sydra_range_array_to_gpu(array):
    "In SYDRA, a range array is a list of tuples. In gpuRIR, it is a tuple of lists."

    min_limits = [elem[0] for elem in array]
    max_limits = [elem[1] for elem in array]

    return [min_limits, max_limits]
