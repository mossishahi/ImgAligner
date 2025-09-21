from .akaze import akaze_feature_matching
from .superpoint import superpoint_feature_matching
from .brisk import brisk_feature_matching
from .daisy import daisy_pure_feature_matching
from .orb import orb_feature_matching
import numpy as np
import cv2
from .image_register import compute_homography_and_warp, adjust_homography_scale, extract_and_warp_roi_correct, check_local_consistency
from ..utils.plotting import plot_match_consistency

def extract_features(tile1, tile2, label="", method='akaze', **kwargs):
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
    features = None
    try:
        if method == 'akaze':
            features = akaze_feature_matching(tile1, tile2, **kwargs)
        elif method == 'orb':
            features = orb_feature_matching(tile1, tile2, **kwargs)
        elif method == 'superpoint':
            features = superpoint_feature_matching(tile1, tile2, **kwargs)
        elif method == 'brisk':
            features = brisk_feature_matching(tile1, tile2, **kwargs)
        elif method == 'daisy':
            features = daisy_pure_feature_matching(tile1, tile2, **kwargs)
        else:
            raise ValueError(f"Unknown feature extraction method '{method}'")
    except Exception as e:
        print(f"[Warning] {label}: failed with {method} — {e}")
    return features


def search_scales(img1, img2, x0, y0, tile_size, scales=[0.01], n_neighbors=3, error_thresh=50.0):
    pts_mov_all = {}
    pts_ref_all = {}
    for scale in scales:
        img1_down = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        img2_down = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        orb_features = orb_feature_matching(img1_down, img2_down)
        good_matches, kp1, kp2, src_pts, dst_pts = orb_features
        H, _, _ = compute_homography_and_warp(img1_down, img2_down, src_pts, dst_pts, good_matches, ransac_thresh=3.0)
        H_fullres_rigid=adjust_homography_scale(H, scale)
        tile_roi1, tile_roi2, tile_roi1_full, tile_roi2_full = extract_and_warp_roi_correct(img1, 
                                                                                            img2, 
                                                                                            x0, y0, 
                                                                                            tile_size, 
                                                                                            H_fullres_rigid)
        H, inliers, pts1, pts2, kp1, kp2 = akaze_feature_matching(tile_roi1, tile_roi2)
        consistency_flags = check_local_consistency(pts1, pts2, n_neighbors=n_neighbors, error_thresh=error_thresh)
        consistent_matches = [m for m, c in zip(inliers, consistency_flags) if c]
        inconsistent_matches = [m for m, c in zip(inliers, consistency_flags) if not c]

        pts_mov = np.float32([kp1[m.queryIdx].pt for m in consistent_matches]) 
        pts_ref = np.float32([kp2[m.trainIdx].pt for m in consistent_matches]) 
        pts_mov_all[scale] = pts_mov
        pts_ref_all[scale] = pts_ref
    return pts_mov_all, pts_ref_all, tile_roi1, tile_roi2

    #     n_neighbors = 4
    #     error_threshold = 50.0  # pixels, adjust based on your data
    #     consistency_flags = check_local_consistency(pts1, pts2, n_neighbors, error_threshold)
    #     consistent_matches = [m for m, c in zip(inliers, consistency_flags) if c]
    #     inconsistent_matches = [m for m, c in zip(inliers,consistency_flags) if not c]
    #     consistent_matches_list.extend(consistent_matches)
    #     inconsistent_matches_list.extend(inconsistent_matches)
    #     kp1_list.extend(kp1)
    #     kp2_list.extend(kp2)
    #     print(f'{len(consistent_matches)} consistent matches and {len(inconsistent_matches)} inconsistent matches')
    #     plot_match_consistency(tile_roi1, tile_roi2, kp1, kp2, consistent_matches, inconsistent_matches)
    #     plt.show()
    #     # for m in consistent_matches:
    #     #     all_pts1.append(kp1[m.queryIdx].pt)
    #     #     all_pts2.append(kp2[m.trainIdx].pt)
    # # warped = piecewise_affine_warp(all_pts1, all_pts2, tile_roi2)
    # # plot_overlay(tile_roi1, warped, figsize=(15, 15))
    # # plt.show()
