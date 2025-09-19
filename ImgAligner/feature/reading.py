import numpy as np
import tifffile as tf
from pathlib import Path

def load_round_channels(base_path, round_number, n_channels=3, ch_prefix='ch', file_suffix='.tif'):
    """
    Load all channels for a given round of images.

    Parameters:
    - base_path: str or Path, folder containing the image files
    - round_number: int, the round index (e.g., 0, 1, 2)
    - n_channels: int, number of channels per round (default 3)
    - ch_prefix: str, prefix for channel filenames (default 'ch')
    - file_suffix: str, file extension (default '.tif')

    Returns:
    - List of np.arrays, one per channel
    """
    base_path = Path(base_path)
    channels = []
    
    for ch in range(n_channels):
        file_name = f"Round{round_number}_{ch_prefix}{ch}{file_suffix}"
        file_path = base_path / file_name
        img = tf.imread(str(file_path))
        channels.append(img)
    
    return channels


