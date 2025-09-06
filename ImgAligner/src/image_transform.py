import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import xml.etree.ElementTree as ET
from xml.dom import minidom
import tifffile as tf
from pathlib import Path
import sys

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


def merge_channels_to_rgb(channels):
    """
    Sum a list of channels and create an RGB image where all channels are the same.

    Parameters:
    - channels: list of np.arrays (e.g., [ch1, ch2, ch3])

    Returns:
    - rgb_img: np.array, merged RGB image
    """
    if not channels:
        raise ValueError("Channel list is empty")
    
    # Sum all channels
    summed = np.zeros_like(channels[0], dtype=np.float32)
    for ch in channels:
        summed += ch.astype(np.float32)
    
    # Clip to valid range if needed (optional)
    summed = np.clip(summed, 0, 255)
    
    # Merge into 3-channel RGB
    rgb_img = cv2.merge([summed, summed, summed]).astype(np.uint8)
    
    return rgb_img

def normalize_uint16_to_uint8(rgb_img):
    """
    Normalize a 3-channel RGB image from uint16 to uint8.

    Parameters:
    - rgb_img: np.array, 3-channel RGB image (uint16)

    Returns:
    - rgb_img_uint8: np.array, normalized uint8 RGB image
    """
    return cv2.convertScaleAbs(rgb_img, alpha=(255.0 / 65535.0))

def normalize_and_clahe(tile,clipLimit=4.0,tileGridSize=(8,8)):
    tile = normalize_uint16_to_uint8(tile)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return cv2.merge([clahe.apply(c) for c in cv2.split(tile)])