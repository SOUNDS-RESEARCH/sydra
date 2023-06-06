from ..moving_source.mic_array_geometries import MIC_ARRAY_GEOMETRIES
from ..moving_source.random_trajectory_generator import get_random_scene_config
from ..moving_source.visualization.moving_scene import plot_moving_source


def test_get_random_scene_config():
    output_plot_path = "tests/temp/random_scene_config.png"

    config = {
        "room_dims": [[3, 3, 3],[10, 8, 6]],
        "rt60": [0.2, 1.3],
        "absorption_weights": [[0.5]*6, [1.0]*6], # Fixed for now
        "mic_array_geometry": MIC_ARRAY_GEOMETRIES["benchmark"],
        "mic_coordinates": [[0.1, 0.1, 0.1], [0.9, 0.9, 0.5]],
        "snr": 30,
        "n_trajectory_points": 156,
        "default_device_height": 1
    }

    config = get_random_scene_config(
        config["mic_coordinates"],
        config["mic_array_geometry"],
        config["room_dims"], config["rt60"],
        config["absorption_weights"],
        config["snr"],
        config["n_trajectory_points"],
    )

    plot_moving_source(config["room_dims"], config["mic_coordinates"],
               config["trajectory_points"], config["rt60"], config["snr_in_db"],
               view="XYZ", output_filename=output_plot_path)