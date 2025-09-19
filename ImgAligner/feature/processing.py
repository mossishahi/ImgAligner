import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import xml.etree.ElementTree as ET
from xml.dom import minidom

from tqdm import tqdm
from skimage.registration import phase_cross_correlation
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.ndimage import fourier_shift, shift
from scipy.interpolate import Rbf
from sklearn.neighbors import NearestNeighbors


def compute_homography_and_warp(img1, img2, src_pts, dst_pts, matches, ransac_thresh=3.0):
    """
    Estimate homography and warp img2 to align with img1 based on matched keypoints.

    Parameters:
    - img1, img2: Input images (img1 is reference)
    - src_pts: Keypoints in img1 (N x 1 x 2)
    - dst_pts: Corresponding keypoints in img2 (N x 1 x 2)
    - matches: List of cv2.DMatch objects
    - ransac_thresh: RANSAC reprojection threshold

    Returns:
    - H: Homography matrix
    - img2_aligned: Warped version of img2 aligned to img1
    - inlier_matches: Filtered matches that are consistent with the homography
    """
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)

    inlier_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]

    height, width = img1.shape[:2]
    img2_aligned = cv2.warpPerspective(img2, H, (width, height))

    return H, img2_aligned, inlier_matches


def adjust_homography_scale(H, scale):
    """
    Adjust a homography matrix estimated on downscaled images 
    to apply it to full-resolution images.

    Parameters:
    - H: 3x3 homography matrix (estimated on downscaled images)
    - scale: float, downscaling factor (e.g., 0.25 if image was 4x smaller)

    Returns:
    - H_fullres: 3x3 homography matrix adjusted for full-resolution images
    """
    if H is None:
        raise ValueError("Input homography matrix H is None.")

    S = np.diag([1 / scale, 1 / scale, 1.0])  # Upscaling matrix
    H_fullres = S @ H @ np.linalg.inv(S)      # Scale-adjusted homography
    return H_fullres