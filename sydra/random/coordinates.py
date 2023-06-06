import numpy as np
import random

from omegaconf import ListConfig
from pyroomacoustics.beamforming import circular_2D_array


def generate_mic_and_source_coords(room_dims, mic_config,
                                   base_source_coordinates,
                                   n_sources,
                                   n_interferers,
                                   min_wall_distance,
                                   min_mic_source_dist,
                                   default_device_height):

    if mic_config["mic_type"] == "circular":
        # Increase margins to the radius of the arrays
        if type(mic_config["radius_in_meters"]) in (tuple, list, ListConfig):
            # If radius is a random range, pick the maximum possible value
            max_radius = mic_config["radius_in_meters"][-1]
        else:
            max_radius = mic_config["radius_in_meters"]

        min_mic_source_dist += max_radius
        min_wall_distance += max_radius

    # 0.2 Prepare base_source_coordinates 
    if base_source_coordinates:
        base_source_coordinates = [base_source_coordinates]

    # 1. Loop until a valid configuration is found,
    # which means every (source, mic) pair is at least 'min_mic_source_dist' meters apart
    while True:
        # 1. Generate candidate source and microphone coordinates
        mic_coordinates = _generate_random_points(
                            room_dims, mic_config["n_mics"],
                            margin=min_wall_distance,
                            base_coordinates=mic_config["mic_coordinates"],
                            default_height=default_device_height)

        source_coordinates = _generate_random_points(
                                room_dims, n_sources, min_wall_distance,
                                base_coordinates=base_source_coordinates,
                                default_height=default_device_height)
        
        interferer_coordinates = _generate_random_points(
                    room_dims, n_interferers, min_wall_distance,
                    default_height=default_device_height)

        # 2. Verify if they are the required distance apart
        all_points = np.concatenate([
            np.array(mic_coordinates),
            np.array(source_coordinates)
        ])
        if interferer_coordinates:
            all_points = np.concatenate([
                all_points, np.array(interferer_coordinates)
            ])

        min_dist = _min_dist(all_points)
        if min_dist < min_mic_source_dist:
            continue
        
        # 3. Expand individual points into microphone arrays 
        array_centres = mic_coordinates
        if mic_config["mic_type"] == "circular":
            mic_coordinates = [
                _circular_2D_array(center,
                                   mic_config["n_array"],
                                   mic_config["phi"],
                                   mic_config["radius_in_meters"])
                for center in mic_coordinates
            ]
            mic_coordinates = np.array(mic_coordinates)
        else:
            mic_coordinates = np.array([mic_coordinates]) # An array of a single element

        return (
            mic_coordinates,
            array_centres,
            source_coordinates,
            interferer_coordinates
        )


def _generate_random_points(room_dims,
                            n_points,
                            margin=0,
                            base_coordinates=None,
                            default_height=None):
    
    """Generate random points within a shoebox enclosure

    Args:
        room_dims : 3-dimensional array-like
        n_points : Number of points to generate within the room
        margin : Minimum distance required for the points to be apart from the enclosure's limits. Defaults to 0.
        base_coordinates : List of tuples to draw the coordinate ranges from.
        default_height (float): If provided, all points will have this value as a fixed z coordinate.
    """
    
    if not base_coordinates:
        # If base coordinates are not provided, generate them randomly within the room
        points = [
            [
                random.uniform(margin, room_dims[0] - margin),
                random.uniform(margin, room_dims[1] - margin),
                random.uniform(margin, room_dims[2] - margin),
            ]
            for _ in range(n_points)
        ]
    else:
        # If mic coordinates are provided, use them according to the format provided.
        def _generate_mic_coordinate(coordinate, max_value):
            if coordinate is None:
                return random.uniform(margin, max_value - margin)
            elif type(coordinate) in (int, float): # Coordinate is fixed
                return min(max(coordinate, margin), max_value - margin)
            elif type(coordinate) in (tuple, list, ListConfig): # Coordinate is a range
                return random.uniform(coordinate[0], coordinate[1])

        points = [
            [
                _generate_mic_coordinate(base_coordinates[i][0], room_dims[0]),
                _generate_mic_coordinate(base_coordinates[i][1], room_dims[1]),
                _generate_mic_coordinate(base_coordinates[i][2], room_dims[2])
            ]
            for i in range(n_points)
        ]

    # Fix deterministic height, if provided
    if default_height is not None:
        for point in points:
            point[2] = default_height
    
    return points


def _generate_random_value(base_value):
    "Used to generate random radius and phi for the circular microphone arrays"
    if type(base_value) in (int, float): # base_value is fixed
        return base_value
    elif type(base_value) in (tuple, list, ListConfig): # base_value is a range
        return random.uniform(base_value[0], base_value[1])


def _min_dist(points):
    "Compute the minimum distance between a set of points"
    n_points = len(points)

    min_dist = np.sqrt(np.sum((points[0] - points[1])**2))
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.sqrt(np.sum((points[i] - points[j])**2))
            min_dist = min(dist, min_dist)
    
    return min_dist


def _circular_2D_array(center, n_array, phi, radius_in_meters):
    radius_in_meters = _generate_random_value(radius_in_meters)
    phi = _generate_random_value(phi)

    height = center[2]
    array = circular_2D_array(center[:2], n_array, phi, radius_in_meters)
    # Concatenate height back to array
    array = np.concatenate([
        array, np.ones((1, array.shape[1]))*height
    ])
    return array.T
