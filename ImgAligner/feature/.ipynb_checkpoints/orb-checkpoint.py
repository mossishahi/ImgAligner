import cv2
import numpy as np

def orb_feature_matching(img1, img2, nfeatures=10000, max_dist=400):
    """
    Detect and match ORB features between two images.

    Parameters:
    - img1, img2: Input images (grayscale or single-channel)
    - nfeatures: Number of ORB features to detect
    - max_dist: Maximum distance to keep good matches

    Returns:
    - good_matches: Filtered list of cv2.DMatch objects
    - kp1, kp2: Keypoints from img1 and img2
    - src_pts: Matched keypoints in img1 (Nx1x2)
    - dst_pts: Matched keypoints in img2 (Nx1x2)
    """
    # ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    # Detect and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Brute-force matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Filter by distance
    good_matches = [m for m in matches if m.distance < max_dist]
    
    # Extract point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    return good_matches, kp1, kp2, src_pts, dst_pts

