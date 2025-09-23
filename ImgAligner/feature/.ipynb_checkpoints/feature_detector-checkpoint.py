from .akaze import akaze_feature_matching
from .superpoint import superpoint_feature_matching
from .brisk import brisk_feature_matching
from .daisy import daisy_pure_feature_matching
from .orb import orb_feature_matching
import numpy as np
import cv2
from .image_register import compute_homography_and_warp, adjust_homography_scale, extract_and_warp_roi_correct, check_local_consistency
from ..utils.image_transform import normalize_uint16_to_uint8
from ..utils.plotting import plot_match_consistency
from tqdm import tqdm

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


def extract_matches_over_image(
    img1_enh, img2_enh,
    img1_chs, img2_chs,
    H_fullres,
    tile_size=4000,
    stride=None,
    scales_list=[[1.0]],                 # list of lists of scales
    feature_extractors_list=[["ORB","AKAZE"]],  # list of lists of extractors
    clahe_params_list=[{"clipLimit": 4.0, "tileGridSize": (8, 8)}],
    n_neighbors=5,
    error_threshold=50.0,
    min_matches=3
):
    """
    Extract consistent feature matches over the entire image using overlapping tiles.

    Supports multiple parameter sets: tries all combinations of scales, feature extractors,
    and CLAHE parameters per tile.

    Returns:
        all_matches: list of dicts (each contains coordinates + metadata for matches)
    """

    if stride is None:
        stride = tile_size // 2  # default = 50% overlap

    h, w = img1_enh.shape[:2]
    all_matches = []

    print(f"Tiling image of size {w}x{h} with tile size {tile_size} and stride {stride}...")

    for y0 in tqdm(range(0, h - tile_size + 1, stride)):
        for x0 in range(0, w - tile_size + 1, stride):
            print(f"\n==> Processing tile at ({x0}, {y0})")

            # Loop over parameter combinations
            for scales in scales_list:
                for feature_extractors in feature_extractors_list:
                    for clahe_params in clahe_params_list:
                        try:
                            matches, roi1, roi2 = get_global_matches_for_tile(
                                x0=x0, y0=y0, tile_size=tile_size,
                                img1_enh=img1_enh, img2_enh=img2_enh,
                                img1_chs=img1_chs, img2_chs=img2_chs,
                                H_fullres=H_fullres,
                                scales=scales,
                                feature_extractors=feature_extractors,
                                n_neighbors=n_neighbors,
                                error_threshold=error_threshold,
                                min_matches=min_matches,
                                clahe_params=clahe_params
                            )

                            if matches:
                                all_matches.extend(matches)

                        except Exception as e:
                            print(f"[Warning] Tile ({x0}, {y0}) failed for params "
                                  f"{scales}, {feature_extractors}, {clahe_params}: {e}")

    print(f"\n✅ Total match sets collected: {len(all_matches)}")

    return all_matches


import cv2
import numpy as np

def detect_features_in_pair(
    roi1, roi2,
    H_fullres,
    x0, y0, tile_size,
    scales,
    feature_extractors,
    n_neighbors,
    error_threshold,
    min_matches,
    clahe_params,
    channel_label="merged"
):
    """
    Detect consistent matches between roi1 and roi2 across multiple scales and detectors.
    Returns a list of match dicts with metadata.
    """
    matches_out = []

    # Ensure grayscale
    def to_gray(img):
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    for scale in scales:
        if scale <= 0:
            continue

        # Downsample ROIs
        roi1_scaled = cv2.resize(roi1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        roi2_scaled = cv2.resize(roi2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        roi1_gray = to_gray(roi1_scaled)
        roi2_gray = to_gray(roi2_scaled)

        for extractor in feature_extractors:
            try:
                if extractor.upper() == "ORB":
                    results = orb_feature_matching(roi1_gray, roi2_gray)
                elif extractor.upper() == "AKAZE":
                    results = akaze_feature_matching(roi1_gray, roi2_gray)
                else:
                    continue

                # Normalize return signature
                if len(results) == 6:
                    H_local, inlier_matches, pts1_local, pts2_local, kp1, kp2 = results
                elif len(results) == 5:
                    inlier_matches, kp1, kp2 ,pts1_local, pts2_local = results
                    #kp2 = kp1
                else:
                    raise ValueError(f"[Error] Unexpected return format from {extractor}: {len(results)}")

                if len(inlier_matches) == 0:
                    continue

                # Handle both DMatch or tuple list formats
                if isinstance(inlier_matches[0], cv2.DMatch):
                    pts1_scaled = np.float32([kp1[m.queryIdx].pt for m in inlier_matches])
                    pts2_scaled = np.float32([kp2[m.trainIdx].pt for m in inlier_matches])
                else:
                    pts1_scaled = np.float32([m[0] for m in inlier_matches])
                    pts2_scaled = np.float32([m[1] for m in inlier_matches])

                if len(pts1_scaled) < n_neighbors + 1:
                    continue

                # Local consistency filtering
                consistency_flags = check_local_consistency(
                    pts1_scaled, pts2_scaled, n_neighbors, error_threshold
                )
                consistency_flags = np.asarray(consistency_flags, dtype=bool)
                pts1_consistent = pts1_scaled[consistency_flags]
                pts2_consistent = pts2_scaled[consistency_flags]

                if len(pts1_consistent) < min_matches:
                    continue

                # Map back to full resolution and global coordinates
                pts1_roi_fullres = pts1_consistent / scale
                pts2_roi_fullres = pts2_consistent / scale
                pts1_global = pts1_roi_fullres + np.array([x0, y0], dtype=np.float32)
                pts2_global_in_img1 = pts2_roi_fullres + np.array([x0, y0], dtype=np.float32)

                try:
                    H_inv = np.linalg.inv(H_fullres)
                    pts2_global_original = cv2.perspectiveTransform(
                        pts2_global_in_img1.reshape(1, -1, 2), H_inv
                    )[0]
                except Exception:
                    pts2_global_original = pts2_global_in_img1

                matches_out.append({
                    "pts1_global": pts1_global,
                    "pts2_global_orig": pts2_global_original,
                    "pts1_roi": pts1_roi_fullres,
                    "pts2_roi": pts2_roi_fullres,
                    "scale": scale,
                    "detector": extractor,
                    "channel": channel_label,
                    "roi": (x0, y0, tile_size),
                    "normalization": clahe_params
                })

            except Exception as e:
                print(f"[Warning] {extractor} matching failed at scale {scale}: {e}")

    return matches_out


def get_global_matches_for_tile(
    x0, y0, tile_size,
    img1_enh, img2_enh,
    img1_chs, img2_chs,
    H_fullres,
    scales=[1.0],
    feature_extractors=None,
    n_neighbors=5,
    error_threshold=50.0,
    min_matches=3,
    clahe_params=None
):
    """
    Extract consistent matches for a tile using per-tile CLAHE, multi-scale feature extraction,
    and configurable feature extractors.

    Returns:
        all_matches: list of dicts with match coordinates + metadata
    """
    if feature_extractors is None:
        feature_extractors = ["ORB", "AKAZE"]

    if clahe_params is None:
        clahe_params = {"clipLimit": 4.0, "tileGridSize": (8, 8)}

    clahe = cv2.createCLAHE(
        clipLimit=clahe_params.get("clipLimit", 4.0),
        tileGridSize=clahe_params.get("tileGridSize", (8, 8))
    )

    def apply_clahe_to_img(img):
        if img.ndim == 2:
            return clahe.apply(img)
        channels = cv2.split(img)
        enhanced = [clahe.apply(c) for c in channels]
        return cv2.merge(enhanced)

    # --- Merged ROI ---
    rois = extract_and_warp_roi_correct(img1_enh, img2_enh, x0, y0, tile_size, H_fullres)
    tile_roi1, tile_roi2 = rois[:2]
    tile_roi1 = apply_clahe_to_img(tile_roi1)
    tile_roi2 = apply_clahe_to_img(tile_roi2)

    print(tile_roi1.shape)
    all_matches = []

    # Detect on merged ROI
    all_matches.extend(
        detect_features_in_pair(
            tile_roi1, tile_roi2,
            H_fullres, x0, y0, tile_size,
            scales, feature_extractors,
            n_neighbors, error_threshold, min_matches,
            clahe_params, channel_label="merged"
        )
    )

    # --- Per-channel ROI ---
    for i, (ch1, ch2) in enumerate(zip(img1_chs, img2_chs), start=1):
        rois_ch = extract_and_warp_roi_correct(ch1, ch2, x0, y0, tile_size, H_fullres)
        roi1_ch, roi2_ch = rois_ch[:2]
        roi1_ch=normalize_uint16_to_uint8(roi1_ch)
        roi2_ch=normalize_uint16_to_uint8(roi2_ch)
        roi1_ch = apply_clahe_to_img(roi1_ch)
        roi2_ch = apply_clahe_to_img(roi2_ch)

        
        all_matches.extend(
            detect_features_in_pair(
                roi1_ch, roi2_ch,
                H_fullres, x0, y0, tile_size,
                scales, feature_extractors,
                n_neighbors, error_threshold, min_matches,
                clahe_params, channel_label=f"ch{i}"
            )
        )

    return all_matches,tile_roi1,tile_roi2

