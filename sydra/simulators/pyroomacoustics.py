from pyroomacoustics import Material

from ..random.surface_absorptions import SURFACE_NAMES, SURFACE_ABSORPTIONS

from .pyroomasync.pyroomasync import ConnectedShoeBox, simulate


def pyroomacoustics_simulator(config):
    """Simulate sound propagation from a sound source to a pair of microphones 

    Args:
        config (dict): Dictionary containing the following keys:
                        - room_dims
                        - sr
                        - anechoic
                        - mic_coordinates
                        - mic_delays
                        - source_coordinates
                        - source_signals
                        - rt60
                        - surface_absorptions
                        - snr_in_db
                        - ism_order

    Returns:
        numpy.array: matrix containing one microphone signal per row
    """

    sr = float(config["sr"])

    snr_in_db = config["snr_in_db"]

    kwargs = {
        "max_order": config["ism_order"]
    }
    if config["anechoic"]:
        kwargs["max_order"] = 0

    else:
        if config["surface_absorptions"] is not None:
            materials = _format_surface_absorptions_to_pyroomacoustics(
                                            config["surface_absorptions"])
            kwargs["materials"] = materials
        else:
            kwargs["rt60"] = float(config["rt60"])

    room = ConnectedShoeBox(config["room_dims"],
                        fs=sr,
                        **kwargs)

    for mic_coords in config["mic_coordinates"]:
        room.add_microphone_array(mic_coords,
                                  delay=config["mic_delays"],
                                  fs_offset=config["mic_sampling_rate_offsets"],
                                  gain=config["mic_gains"])

    for source_coords, source_signal in zip(config["source_coordinates"],
                                            config["source_signals"]):
        room.add_source(source_coords, source_signal)

    for interferer_coords, interferer_signal in zip(config["interferer_coordinates"],
                                                    config["interferer_signals"]):
        room.add_source(interferer_coords, interferer_signal)

    signals = simulate(room, snr=snr_in_db)
    n_signals, n_signal = signals.shape
    n_devices = len(config["mic_coordinates"])
    signals_per_device = n_signals//n_devices
    signals = signals.reshape((n_devices, signals_per_device, n_signal))

    return signals


def _format_surface_absorptions_to_pyroomacoustics(surface_absorptions):
    materials = {}
    for surface_name in SURFACE_NAMES:
        absorption_values = surface_absorptions[surface_name]
        
        materials[surface_name] = Material({
                "description": "",
                "coeffs": absorption_values,
                "center_freqs": SURFACE_ABSORPTIONS["frequency_bands"]
            }
        )
    return materials
