import numpy as np
#This file is used to perform a "reflectivity-biased" sampling procedure 
# as defined in https://asa.scitation.org/doi/10.1121/10.0005888.

SURFACE_ABSORPTIONS = {
    "frequency_bands": [125, 250, 500, 1000, 2000, 4000],
    "ranges": {
        "floor": [
            [0.01, 0.2] , [0.01, 0.3] , [0.05, 0.5] ,
            [0.15, 0.6] , [0.25, 0.75] , [0.3, 0.8]
        ],
   
        "wall": [
            [0, 0.5],[0, 0.5],[0, 0.3],
            [0, 0.12],[0, 0.12],[0, 0.12]
        ],
       "ceiling": [
            [0, 0.12], [0.15, 1], [0.4, 1],
            [0.4, 1], [0.4, 1], [0.3, 1]
        ],
        "reflective": [0, 0.12] # reflective has a single range
    }
}

WALL_SURFACE_NAMES = ["west", "east", "north", "south"]

SURFACE_NAMES = [
    "west", 
    "east", 
    "north", 
    "south", 
    "floor", 
    "ceiling"
]


def generate_random_surface_absorption():
    """Generate random surface absorption coefficients using
    The 'Reflectivity Biased' sampling approach defined by 
    https://asa.scitation.org/doi/10.1121/10.0005888.
    """

    bands = SURFACE_ABSORPTIONS["frequency_bands"]
    absorption_ranges = SURFACE_ABSORPTIONS["ranges"]
    n_bands = len(bands)

    surface_absorptions = {}
    for surface in ["floor", "wall", "ceiling"]:
        if surface == "wall":
            # 4 coefficients will be generated
            n_surfaces = 4
        else:
            n_surfaces = 1

        is_surface_hard = np.random.randint(0, 2) # Coin toss
        if is_surface_hard:
            # Draw a single value and assign it to all frequency bands
            absorptions = np.random.uniform(*absorption_ranges["reflective"], n_surfaces)
            absorptions = np.repeat(absorptions.reshape(n_surfaces, 1), n_bands, axis=1)
            # absorptions.shape == (n_surfaces, n_bands)

        else:
            absorptions = np.stack([
                np.random.uniform(*band_range, n_surfaces)
                for band_range in absorption_ranges[surface]
            ]).T
        
        if surface == "wall":
            for i, wall_name in enumerate(WALL_SURFACE_NAMES):
                surface_absorptions[wall_name] = absorptions[i]
        else:
            surface_absorptions[surface] = absorptions[0]
    
    return surface_absorptions