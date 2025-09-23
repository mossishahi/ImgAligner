import cv2
import numpy as np

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
    
    # Brute-force matcher with Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Distance filter   
    good_matches = [m for m in matches if m.distance < ratio_thresh]

    # Extract points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    if len(pts1) < 2 or len(pts2) < 2:
        if verbose:
            print("Not enough DAISY matches for homography")
        return None, [], [], [], kp1, kp2

    # RANSAC homography
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return None, [], [], [], kp1, kp2

    mask = mask.ravel().astype(bool)
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
    pts1_inliers = pts1[mask]
    pts2_inliers = pts2[mask]

    if verbose:
        print(f"DAISY total matches: {len(good_matches)}, inliers: {len(inlier_matches)}")

    return H, inlier_matches, pts1_inliers, pts2_inliers, kp1, kp2