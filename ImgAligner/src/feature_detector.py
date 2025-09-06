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


from tqdm import tqdm
from skimage.registration import phase_cross_correlation
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.ndimage import fourier_shift, shift
from scipy.interpolate import Rbf
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn


def extract_features(tile1, tile2, label="", n_neighbors=10, error_threshold=50, method='akaze', **kwargs):
    """
    Unified interface for feature extraction and matching.
    
    Args:
        tile1, tile2: input grayscale or RGB images
        label: string identifier for logging
        n_neighbors: neighbors for consistency check
        error_threshold: threshold for consistency check
        method: one of {'akaze','orb','superpoint','brisk','daisy'}
        kwargs: passed to the selected feature function
    """
    try:
        if method == 'akaze':
            H, inliers, pts1, pts2, kp1, kp2 = akaze_feature_matching(tile1, tile2, **kwargs)
        elif method == 'orb':
            H, inliers, pts1, pts2, kp1, kp2 = orb_feature_matching(tile1, tile2, **kwargs)
        elif method == 'superpoint':
            H, inliers, pts1, pts2, kp1, kp2 = superpoint_feature_matching(tile1, tile2, **kwargs)
        elif method == 'brisk':
            H, inliers, pts1, pts2, kp1, kp2 = brisk_feature_matching(tile1, tile2, **kwargs)
        elif method == 'daisy':
            H, inliers, pts1, pts2, kp1, kp2 = daisy_pure_feature_matching(tile1, tile2, **kwargs)
        else:
            raise ValueError(f"Unknown feature extraction method '{method}'")

        # Optional: run your consistency filter
        if len(pts1) > 0 and len(pts2) > 0:
            consistency_flags = check_local_consistency(pts1, pts2, n_neighbors, error_threshold)
            consistent_matches = [m for m, c in zip(inliers, consistency_flags) if c]
        else:
            consistent_matches = []

        return consistent_matches, kp1, kp2

    except Exception as e:
        print(f"[Warning] {label}: failed with {method} — {e}")
        return [], [], []


def orb_feature_matching(tile_roi1, tile_roi2, nfeatures=5000, ransac_thresh=500.0, max_dist=400, verbose=True):
    """
    Perform ORB feature detection and matching with RANSAC-based homography.
    Returns the same format as other feature matching functions.
    """

    # ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    # Detect + compute descriptors
    kp1, des1 = orb.detectAndCompute(tile_roi1, None)
    kp2, des2 = orb.detectAndCompute(tile_roi2, None)

    if len(kp1) < 2 or len(kp2) < 2:
        if verbose:
            print("Not enough ORB keypoints")
        return None, [], [], [], kp1, kp2

    # Brute-force matcher with Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Distance filter
    good_matches = [m for m in matches if m.distance < max_dist]

    if len(good_matches) < 4:
        if verbose:
            print("Not enough ORB matches for homography")
        return None, [], [], [], kp1, kp2

    # Extract points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC homography
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return None, [], [], [], kp1, kp2

    mask = mask.ravel().astype(bool)
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
    pts1_inliers = pts1[mask]
    pts2_inliers = pts2[mask]

    if verbose:
        print(f"ORB total matches: {len(good_matches)}, inliers: {len(inlier_matches)}")

    return H, inlier_matches, pts1_inliers, pts2_inliers, kp1, kp2







def akaze_feature_matching(tile_roi1, tile_roi2, ratio_thresh=0.8, ransac_thresh=500.0, verbose=True):
    """
    Perform AKAZE feature detection and matching between two images, then filter matches by Lowe's ratio test and RANSAC.
    
    Args:
        tile_roi1 (np.ndarray): First grayscale image (reference).
        tile_roi2 (np.ndarray): Second grayscale image (to match).
        ratio_thresh (float): Lowe's ratio test threshold (default 0.8).
        ransac_thresh (float): RANSAC reprojection threshold (default 500.0).
        verbose (bool): If True, print summary information.
        
    Returns:
        H (np.ndarray): Homography matrix estimated from inlier matches.
        good_matches (list of cv2.DMatch): List of inlier matches after RANSAC filtering.
        pts1 (np.ndarray): Matched points coordinates from first image (Nx2).
        pts2 (np.ndarray): Matched points coordinates from second image (Nx2).
        kp1 (list of cv2.KeyPoint): Keypoints in first image.
        kp2 (list of cv2.KeyPoint): Keypoints in second image.
    """
    # Initialize AKAZE detector
    detector = cv2.AKAZE_create(threshold=0.0005)
    kp1, des1 = detector.detectAndCompute(tile_roi1, None)
    kp2, des2 = detector.detectAndCompute(tile_roi2, None)

    # FLANN matcher setup for AKAZE (Lowe's recommended parameters)
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors with k=2 for ratio test
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test safely
    good_matches = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < ratio_thresh * n.distance]

    if len(good_matches) < 4:
        raise RuntimeError("Not enough good matches for homography estimation.")

    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Estimate homography using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)

    # Filter matches by inliers mask
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]

    # Update matched points for inliers only
    pts1 = np.float32([kp1[m.queryIdx].pt for m in inlier_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in inlier_matches])

    if verbose:
        print(f"Total matches: {len(good_matches)}; Inliers after RANSAC: {len(inlier_matches)}")

    return H, inlier_matches, pts1, pts2, kp1, kp2

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = (scores == max_pool(scores))
    for _ in range(2):
        supp_mask = (max_pool(max_mask.float()) > 0)
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = (supp_scores == max_pool(supp_scores))
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2018. https://arxiv.org/abs/1712.07629

    """
    def __init__(self, config={}):
        super().__init__()
        self.config = {
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
            'remove_borders': 4,
        }
        self.config.update(config)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = '/ictstr01/home/icb/mostafa.shahhosseini/code/repos/ImgAligner/ImgAligner/models/superpoint_v1.pth'
        self.load_state_dict(torch.load(path))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or -1')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        } 

def superpoint_feature_matching(tile_roi1, tile_roi2, model=None, device=None, ratio_thresh=0.8, ransac_thresh=500.0, verbose=True):
    """
    Perform SuperPoint feature detection and matching between two images, then filter matches by
    Lowe's ratio test and RANSAC.

    Returns the same tuple format as `akaze_feature_matching`:
        (H, inlier_matches, pts1, pts2, kp1, kp2)
    """
    # Resolve defaults from globals if not provided
    if model is None:
        if 'superpoint_model' in globals():
            model = globals()['superpoint_model']
        else:
            raise RuntimeError("SuperPoint model is not provided and not found as 'superpoint_model'.")
    if device is None:
        if 'device' in globals():
            device = globals()['device']
        else:
            device = 'cpu'

    # Ensure grayscale inputs for SuperPoint
    if tile_roi1.ndim == 3 and tile_roi1.shape[2] == 3:
        img1_gray = cv2.cvtColor(tile_roi1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = tile_roi1
    if tile_roi2.ndim == 3 and tile_roi2.shape[2] == 3:
        img2_gray = cv2.cvtColor(tile_roi2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = tile_roi2

    # Convert numpy arrays to torch tensors [1, 1, H, W] and normalize to float32
    img1_tensor = torch.from_numpy(img1_gray).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
    img2_tensor = torch.from_numpy(img2_gray).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0

    # Run the model
    with torch.no_grad():
        pred1 = model({'image': img1_tensor})
        pred2 = model({'image': img2_tensor})

    # Extract keypoints and descriptors
    kps1_t = pred1['keypoints'][0]            # (N1, 2) tensor of (x, y)
    kps2_t = pred2['keypoints'][0]            # (N2, 2)
    des1_t = pred1['descriptors'][0].t()      # (N1, 256)
    des2_t = pred2['descriptors'][0].t()      # (N2, 256)

    kps1_np = kps1_t.cpu().numpy()
    kps2_np = kps2_t.cpu().numpy()
    des1 = np.ascontiguousarray(des1_t.cpu().numpy().astype(np.float32))
    des2 = np.ascontiguousarray(des2_t.cpu().numpy().astype(np.float32))

    # Convert to OpenCV KeyPoint lists for downstream visualization compatibility
    kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kps1_np]
    kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kps2_np]

    if len(kp1) == 0 or len(kp2) == 0:
        raise RuntimeError("No SuperPoint keypoints detected in one of the images.")

    # Match features with BF + Lowe's ratio test
    good_matches, pts1, pts2 = match_features(kp1, des1, kp2, des2, ratio_thresh=ratio_thresh)

    if len(good_matches) < 4:
        raise RuntimeError("Not enough good matches for homography estimation.")

    # Estimate homography using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        raise RuntimeError("Homography estimation failed.")

    # Filter matches by inliers mask (ensure 1D boolean)
    mask = mask.ravel().astype(bool)
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]

    # Update matched points for inliers only
    pts1_inliers = pts1[mask]
    pts2_inliers = pts2[mask]

    if verbose:
        print(f"Total matches: {len(good_matches)}; Inliers after RANSAC: {len(inlier_matches)}")

    return H, inlier_matches, pts1_inliers, pts2_inliers, kp1, kp2

def brisk_feature_matching(tile_roi1, tile_roi2, ratio_thresh=0.8, ransac_thresh=500.0, verbose=True):
    """
    Perform BRISK feature detection and matching between two images, then filter matches by
    Lowe's ratio test and RANSAC to estimate homography.
    
    Parameters:
    - tile_roi1: First grayscale image (reference)
    - tile_roi2: Second grayscale image (to match)
    - ratio_thresh: Lowe's ratio test threshold (default: 0.8)
    - ransac_thresh: RANSAC threshold for homography estimation (default: 500.0)
    - verbose: Whether to print progress information (default: True)
    
    Returns:
    - H: 3x3 homography matrix (None if estimation fails)
    - inlier_matches: List of inlier matches after RANSAC
    - pts1_inliers: Source points (from tile_roi1) for inlier matches
    - pts2_inliers: Destination points (from tile_roi2) for inlier matches
    - kp1: Keypoints from tile_roi1
    - kp2: Keypoints from tile_roi2
    """
    
    # Initialize BRISK detector
    # BRISK is patent-free and provides good performance
    brisk = cv2.BRISK_create(thresh=30, octaves=3, patternScale=1.0)
    
    # FLANN matcher setup for BRISK (optimized for binary descriptors)
    # BRISK uses binary descriptors, so we use FLANN with Hamming distance
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, 
                        table_number=6, 
                        key_size=12, 
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Detect and compute descriptors
    kp1, des1 = brisk.detectAndCompute(tile_roi1, None)
    kp2, des2 = brisk.detectAndCompute(tile_roi2, None)
    
    if verbose:
        print(f"BRISK detected {len(kp1)} keypoints in image 1, {len(kp2)} keypoints in image 2")
    
    # Check if we have enough keypoints
    if len(kp1) < 2 or len(kp2) < 2:
        if verbose:
            print("Not enough keypoints detected for matching")
        return None, [], [], [], kp1, kp2
    
    # Match descriptors with k=2 for ratio test
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test safely
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    
    if verbose:
        print(f"Lowe's ratio test: {len(good_matches)} good matches out of {len(matches)} total matches")
    
    # Check if we have enough good matches
    if len(good_matches) < 4:
        if verbose:
            print("Not enough good matches for homography estimation (need at least 4)")
        return None, [], [], [], kp1, kp2
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Estimate homography using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    
    # Check if homography estimation was successful
    if H is None or mask is None:
        if verbose:
            print("Homography estimation failed")
        return None, [], [], [], kp1, kp2
    
    # Filter matches by inliers mask (ensure 1D boolean)
    mask = mask.ravel().astype(bool)
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    
    # Extract inlier points
    pts1_inliers = pts1[mask]
    pts2_inliers = pts2[mask]
    
    if verbose:
        print(f"RANSAC: {len(inlier_matches)} inlier matches out of {len(good_matches)} good matches")
        print(f"Homography matrix:\n{H}")
    
    return H, inlier_matches, pts1_inliers, pts2_inliers, kp1, kp2

def daisy_pure_feature_matching(tile_roi1, tile_roi2, ratio_thresh=0.8, ransac_thresh=500.0, verbose=True):
    """
    Pure DAISY implementation using only DAISY for both keypoint detection and description.
    This approach creates a dense grid of keypoints and computes DAISY descriptors.
    """
    
    # Create DAISY descriptor extractor
    daisy = cv2.xfeatures2d.DAISY_create(
        radius=15,
        q_radius=3,
        q_theta=8,
        q_hist=8,
        norm=cv2.NORM_L1
    )
    
    def create_dense_keypoints(image, step_size=15):
        """Create dense keypoints across the image"""
        height, width = image.shape
        keypoints = []
        
        for y in range(step_size, height - step_size, step_size):
            for x in range(step_size, width - step_size, step_size):
                kp = cv2.KeyPoint(x, y, size=step_size)
                keypoints.append(kp)
        
        return keypoints
    
    # Create dense keypoints
    kp1 = create_dense_keypoints(tile_roi1)
    kp2 = create_dense_keypoints(tile_roi2)
    
    if verbose:
        print(f"DAISY dense keypoints: {len(kp1)} in image 1, {len(kp2)} in image 2")
    
    # Compute DAISY descriptors
    kp1, des1 = daisy.compute(tile_roi1, kp1)
    kp2, des2 = daisy.compute(tile_roi2, kp2)
    
    # Rest of the function remains the same...
    # (matching, RANSAC, etc.)



