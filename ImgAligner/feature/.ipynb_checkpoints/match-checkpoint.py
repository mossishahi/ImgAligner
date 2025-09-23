from .feature_detector import extract_features
from .image_register import check_local_consistency
import numpy as np
import cv2
from tqdm import tqdm

def get_consistent_matches(tile1, tile2, method='akaze', label="", n_neighbors=10, error_threshold=50):
    try:
        H, inliers, pts1, pts2, kp1, kp2 = extract_features(tile1, tile2, method=method)
        consistency_flags = check_local_consistency(pts1, pts2, n_neighbors, error_threshold)
        consistent_matches = [m for m, c in zip(inliers, consistency_flags) if c]
        return consistent_matches, kp1, kp2
    except Exception as e:
        print(f"[Warning] {label}: failed to match — {e}")
        return [], [], []

def get_global_matches_for_tile(
    x0, y0, tile_size,
    img1_enh, img2_enh,
    img1_chs, img2_chs,
    H_fullres,
    n_neighbors=5,
    error_threshold=50.0,feature_detector='akaze'
):
    """
    Process a tile to extract consistent feature matches between img1 and img2 (merged + channels),
    return matched points in global coordinates.

    Parameters:
    - x0, y0: top-left tile coordinates
    - tile_size: size of square tile
    - img1_enh, img2_enh: merged/combined channels for visualization/matching
    - img1_chs, img2_chs: lists of 3 individual channels (uint16 or uint8)
    - H_fullres: homography from img2 to img1 (fullres)
    - n_neighbors: number of neighbors for local consistency filtering
    - error_threshold: max error for filtering matches

    Returns:
    - pts1_global: consistent keypoints in img1 (global coordinates)
    - pts2_global_original: corresponding keypoints in original img2 (global coordinates)
    """

    # Step 1: Extract and normalize merged channel
    tile_roi1_merged, tile_roi2_merged = extract_and_warp_roi_correct(
        img1_enh, img2_enh, x0, y0, tile_size, H_fullres
    )

    all_pts1 = []
    all_pts2 = []

    print("Processing merged image...")
    matches, kp1, kp2 = extract_features(tile_roi1_merged, tile_roi2_merged, label="merged")
    for m in matches:
        all_pts1.append(kp1[m.queryIdx].pt)
        all_pts2.append(kp2[m.trainIdx].pt)

    # Step 2: Process each individual channel
    for i, (ch1, ch2) in enumerate(zip(img1_chs, img2_chs), start=1):
        print(f"Processing channel {i}...")
        try:
            tile_roi1_ch, tile_roi2_ch = extract_and_warp_roi_correct(ch1, ch2, x0, y0, tile_size, H_fullres)
            tile_roi1_ch = normalize_and_clahe(tile_roi1_ch)
            tile_roi2_ch = normalize_and_clahe(tile_roi2_ch)
            matches, kp1, kp2 = extract_features(tile_roi1_ch, tile_roi2_ch, label=f"channel {i}",method=feature_detector)
            for m in matches:
                all_pts1.append(kp1[m.queryIdx].pt)
                all_pts2.append(kp2[m.trainIdx].pt)
        except Exception as e:
            print(f"[Warning] Channel {i}: extraction or normalization failed — {e}")

    # Step 3: Combine and filter globally
    pts1 = np.float32(all_pts1)
    pts2 = np.float32(all_pts2)
    print(f"Total combined matches before global consistency check: {len(pts1)}")

    if len(pts1) < n_neighbors + 1:
        print("[Error] Not enough combined matches for global consistency filtering.")
        return np.empty((0, 2)), np.empty((0, 2))

    consistency_flags = check_local_consistency(pts1, pts2, n_neighbors, error_threshold)
    pts1_consistent = pts1[consistency_flags]
    pts2_consistent = pts2[consistency_flags]
    print(f"After global consistency filtering: {len(pts1_consistent)} matches retained")

    # Step 4: Convert to global coordinates
    pts1_global = pts1_consistent + np.array([x0, y0])
    pts2_global = pts2_consistent + np.array([x0, y0])

    # Optional: Visual debug
    if len(pts1_consistent) >= 3:
        try:
            warped = piecewise_affine_warp(pts1_consistent, pts2_consistent, tile_roi2_merged)
            plot_overlay(tile_roi1_merged, warped, figsize=(10, 10))
            # Step 5: Map pts2 to original img2 space
            H_inv = np.linalg.inv(H_fullres)
            pts2_global_original = cv2.perspectiveTransform(pts2_global[None, :, :], H_inv)[0]
            return pts1_global, pts2_global_original
        except Exception as e:
            print(f"[Warning] Piecewise warp failed: {e}")
    else:
        print("[Warning] Not enough points for piecewise affine warp — skipping overlay.")


def extract_matches_over_image(
    img1_enh, img2_enh,
    img1_chs, img2_chs,
    H_fullres,
    tile_size=4000,
    stride=None,
    n_neighbors=5,
    error_threshold=50.0,feature_detector='akaze'):
    """
    Extract consistent feature matches over the entire image using overlapping tiles.

    Parameters:
    - img1_enh, img2_enh: merged image channels
    - img1_chs, img2_chs: list of individual channels (len=3 each)
    - H_fullres: homography matrix
    - tile_size: size of tile (square)
    - stride: step size for tiling (default = tile_size // 2 for 50% overlap)
    - n_neighbors, error_threshold: params for consistency filtering

    Returns:
    - all_pts1_global, all_pts2_global_original: concatenated matches in global coordinates
    """

    if stride is None:
        stride = tile_size // 2  # default to 50% overlap

    h, w = img1_enh.shape[:2]

    all_pts1_global = []
    all_pts2_global_original = []

    print(f"Tiling image of size {w}x{h} with tile size {tile_size} and stride {stride}...")

    for y0 in tqdm(range(0, h - tile_size + 1, stride)):
        for x0 in range(0, w - tile_size + 1, stride):
            print(f"\n==> Processing tile at ({x0}, {y0})")
            pts1, pts2 = get_global_matches_for_tile(
                x0=x0, y0=y0, tile_size=tile_size,
                img1_enh=img1_enh, img2_enh=img2_enh,
                img1_chs=img1_chs, img2_chs=img2_chs,
                H_fullres=H_fullres,
                n_neighbors=n_neighbors,
                error_threshold=error_threshold,feature_detector=feature_detector
            )

            if len(pts1) > 0:
                all_pts1_global.append(pts1)
                all_pts2_global_original.append(pts2)

    if all_pts1_global:
        all_pts1_global = np.concatenate(all_pts1_global, axis=0)
        all_pts2_global_original = np.concatenate(all_pts2_global_original, axis=0)
    else:
        all_pts1_global = np.empty((0, 2))
        all_pts2_global_original = np.empty((0, 2))

    print(f"\n✅ Total matched points collected: {len(all_pts1_global)}")

    return all_pts1_global, all_pts2_global_original
