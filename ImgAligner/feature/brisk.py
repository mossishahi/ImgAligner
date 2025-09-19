import cv2
import numpy as np

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
