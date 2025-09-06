# --- Standard library ---
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- Third-party libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tifffile as tf
from tqdm import tqdm
from skimage.registration import phase_cross_correlation
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.ndimage import fourier_shift, shift
from scipy.interpolate import Rbf
from sklearn.neighbors import NearestNeighbors

# --- Add src folder to sys.path ---
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "src/"))
sys.path.append(parent_dir)

# --- Custom modules ---
from plotting import *
from reading import *
from processing import *
from image_transform import *
from image_register import *



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


