from image_register import check_local_consistency
from akaze import akaze_feature_matching
from superpoint import superpoint_feature_matching
from brisk import brisk_feature_matching
from daisy import daisy_pure_feature_matching
from orb import orb_feature_matching


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




