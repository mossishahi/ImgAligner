import cv2
import numpy as np


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


def add_vectors_to_df(df,tag='roi'):
    """
    Compute vector components from matched points.
    Adds columns dx, dy to the DataFrame.
    """
    df["dx"] = df["x2_"+tag] - df["x1_"+tag]
    df["dy"] = df["y2_"+tag] - df["y1_"+tag]
    df["magnitude"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    return df

import pandas as pd
import numpy as np

def matches_to_dataframe(all_matches):
    """
    Convert the output of get_global_matches_for_tile (list of dicts)
    into a single pandas DataFrame where each match is a row.
    """
    rows = []
    for match_dict in all_matches:
        pts1 = match_dict["pts1_global"]
        pts2 = match_dict["pts2_global_orig"]
        pts1_roi = match_dict["pts1_roi"]
        pts2_roi = match_dict["pts2_roi"]
        n_matches = pts1.shape[0]

        for i in range(n_matches):
            row = {
                "x1_global": float(pts1[i,0]),
                "y1_global": float(pts1[i,1]),
                "x2_global": float(pts2[i,0]),
                "y2_global": float(pts2[i,1]),
                "x1_roi": float(pts1_roi[i,0]),
                "y1_roi": float(pts1_roi[i,1]),
                "x2_roi": float(pts2_roi[i,0]),
                "y2_roi": float(pts2_roi[i,1]),
                "scale": match_dict["scale"],
                "detector": match_dict["detector"],
                "channel": match_dict["channel"],
                "tile_size": match_dict["tile_size"],
                "roi_x": match_dict["roi"][0],
                "roi_y": match_dict["roi"][1],
                "roi_size": match_dict["roi"][2],
                "clahe_clipLimit": match_dict["normalization"].get("clipLimit", np.nan),
                "clahe_tileGridSize": match_dict["normalization"].get("tileGridSize", np.nan)
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df
