import cv2
import numpy as np

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
