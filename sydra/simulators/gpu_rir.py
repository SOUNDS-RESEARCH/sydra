import gpuRIR
import numpy as np

from ..mic_array_geometries import MIC_ARRAY_GEOMETRIES


def gpu_rir_simulator(config):
    mic_array_geometry = MIC_ARRAY_GEOMETRIES[config["mic_type"]]

    n_points = len(config["trajectory_points"])
    # 1. Interpolate trajectory points
    timestamps = np.arange(n_points) * len(config["source_signals"][0]) / config["sr"] / n_points
    t = np.arange(len(config["source_signals"][0]))/config["sr"]

    surface_absorptions = gpuRIR.beta_SabineEstimation(
        config["room_dims"],
        config["rt60"],
        config["absorption_weights"]
    )
    
    if config["rt60"] == 0:
        Tdiff = 0.1
        Tmax = 0.1
        nb_img = [1, 1, 1]
    else:
        # Use ISM until the RIRs decay 12dB
        Tdiff = gpuRIR.att2t_SabineEstimator(12, config["rt60"])
        # Use diffuse model until the RIRs decay 40dB
        Tmax = gpuRIR.att2t_SabineEstimator(40, config["rt60"])
        if config["rt60"] < 0.15:
            Tdiff = Tmax  # Avoid issues with too short RIRs
        nb_img = gpuRIR.t2n(Tdiff, config["room_dims"])

    nb_mics = len(config["mic_coordinates"])
    nb_traj_pts = len(config["trajectory_points"])
    nb_gpu_calls = min(int(np.ceil(
        config["sr"] * Tdiff * nb_mics * nb_traj_pts * np.prod(nb_img) / 1e9)), nb_traj_pts)
    traj_pts_batch = np.ceil(
        nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls+1)).astype(int)

    RIRs_list = [
        gpuRIR.simulateRIR(config["room_dims"], surface_absorptions,
                            config["trajectory_points"][traj_pts_batch[0]:traj_pts_batch[1], :],
                            config["mic_coordinates"], nb_img, Tmax, config["sr"], Tdiff=Tdiff,
                            orV_rcv=mic_array_geometry.mic_orV,
                            mic_pattern=mic_array_geometry.mic_pattern)
    ]
    for i in range(1, nb_gpu_calls):
        RIRs_list += [
            gpuRIR.simulateRIR(config["room_dims"], surface_absorptions,
                                config["trajectory_points"][traj_pts_batch[i]:traj_pts_batch[i+1], :],
                                config["mic_coordinates"], nb_img, Tmax, config["sr"], Tdiff=Tdiff,
                                orV_rcv=mic_array_geometry.mic_orV,
                                mic_pattern=mic_array_geometry.mic_pattern)]
    RIRs = np.concatenate(RIRs_list, axis=0)
    mic_signals = gpuRIR.simulateTrajectory(
        config["source_signals"][0], RIRs, timestamps=timestamps, fs=config["sr"])
    mic_signals = mic_signals[0:len(t), :]

    # Omnidirectional noise
    dp_RIRs = gpuRIR.simulateRIR(
        config["room_dims"], surface_absorptions,
        config["trajectory_points"], config["mic_coordinates"], [1, 1, 1], 0.1, config["sr"],
        orV_rcv=mic_array_geometry.mic_orV, mic_pattern=mic_array_geometry.mic_pattern)
    dp_signals = gpuRIR.simulateTrajectory(
        config["source_signals"][0], dp_RIRs, timestamps=timestamps, fs=config["sr"])
    ac_pow = np.mean([_acoustic_power(dp_signals[:, i])
                        for i in range(dp_signals.shape[1])])
    noise = np.sqrt(ac_pow/10**(config["snr_in_db"]/10)) * \
        np.random.standard_normal(mic_signals.shape)
    mic_signals += noise

    # Apply the propagation delay to the VAD information if it exists
    if "vad" in config:
        vad = gpuRIR.simulateTrajectory(
            config["vad"], dp_RIRs, timestamps=timestamps, fs=config["sr"])
        vad = vad[0:len(t), :].mean(
            axis=1) > vad[0:len(t), :].max()*1e-3
        
        return mic_signals.T, vad
    return mic_signals.T


def _acoustic_power(s):
	""" Acoustic power of after removing the silences.
	"""
	w = 512  # Window size for silent detection
	o = 256  # Window step for silent detection

	# Window the input signal
	s = np.ascontiguousarray(s)
	sh = (s.size - w + 1, w)
	st = s.strides * 2
	S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]

	window_power = np.mean(S ** 2, axis=-1)
	th = 0.01 * window_power.max()  # Threshold for silent detection
	return np.mean(window_power[np.nonzero(window_power > th)])
