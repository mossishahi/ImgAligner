import cv2
import numpy as np

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
