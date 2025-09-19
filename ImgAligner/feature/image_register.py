import cv2
import numpy as np
from tqdm import tqdm
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.interpolate import Rbf
from sklearn.neighbors import NearestNeighbors
import numpy as np
from plotting import *
from image_transform import *
from feature_detector import *

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

def extract_and_warp_roi_correct(img1, img2, x0, y0, size, H_fullres):
    """
    Extract ROI from two images, warp img2 ROI to img1 ROI using homography,
    convert to grayscale, and crop to valid overlapping region.

    Parameters:
    - img1: reference image (full resolution, color or grayscale)
    - img2: moving image (full resolution, color or grayscale)
    - x0, y0: top-left corner coordinates for ROI extraction
    - size: size of square ROI (int)
    - H_fullres: 3x3 homography matrix mapping img2 to img1 coordinates (global)

    Returns:
    - tile_roi1_cropped: grayscale ROI from img1 cropped to valid overlap
    - tile_roi2_cropped: grayscale warped ROI from img2 cropped to valid overlap
    """

    # Extract ROIs
    tile_roi1 = img1[y0:y0+size, x0:x0+size]
    tile      = img2[y0:y0+size, x0:x0+size]

    # Translation matrices to move between global and local coordinates
    T     = np.array([[1, 0, -x0],
                      [0, 1, -y0],
                      [0, 0,  1 ]], dtype=np.float32)  # shift global → local
    T_inv = np.array([[1, 0, x0],
                      [0, 1, y0],
                      [0, 0, 1 ]], dtype=np.float32)   # shift local → global

    # Adapt global homography to this local tile
    H_local = T @ H_fullres @ T_inv

    # Warp img2 tile into tile_roi1 coordinates
    tile_roi2 = cv2.warpPerspective(tile, H_local, (tile_roi1.shape[1], tile_roi1.shape[0]))

    # Convert to grayscale if needed
    if tile_roi1.ndim == 3 and tile_roi1.shape[2] == 3:
        tile_roi1 = cv2.cvtColor(tile_roi1, cv2.COLOR_BGR2GRAY)
    if tile_roi2.ndim == 3 and tile_roi2.shape[2] == 3:
        tile_roi2 = cv2.cvtColor(tile_roi2, cv2.COLOR_BGR2GRAY)

    # Mask out valid overlapping region
    valid_mask = tile_roi2 > 0
    ys, xs = np.where(valid_mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No valid overlapping region found in warped tile.")
    # Crop both ROIs to the valid region
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    tile_roi1_cropped = tile_roi1[y_min:y_max+1, x_min:x_max+1]
    tile_roi2_cropped = tile_roi2[y_min:y_max+1, x_min:x_max+1]

    return tile_roi1_cropped, tile_roi2_cropped


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


def check_local_consistency(pts1, pts2, n_neighbors=5, error_thresh=3.0):
    """
    Check local geometric consistency of point correspondences using local affine transforms.

    Args:
        pts1 (np.ndarray): Nx2 array of points in image 1 (reference).
        pts2 (np.ndarray): Nx2 array of corresponding points in image 2 (aligned).
        n_neighbors (int): Number of neighbors to use for local transformation.
        error_thresh (float): Pixel error threshold for consistency.

    Returns:
        List[bool]: List of consistency flags for each point pair.
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(pts1)
    local_consistency = []

    for i in range(len(pts1)):
        pt1 = pts1[i].reshape(1, -1)
        pt2 = pts2[i].reshape(1, -1)

        distances, indices = nbrs.kneighbors(pt1)
        neighbor_indices = indices[0]

        neigh_pts1 = pts1[neighbor_indices]
        neigh_pts2 = pts2[neighbor_indices]

        local_H, inliers = cv2.estimateAffinePartial2D(
            neigh_pts1, neigh_pts2, method=cv2.RANSAC, ransacReprojThreshold=error_thresh
        )

        if local_H is None:
            local_consistency.append(False)
            continue

        neigh_pts1_h = np.hstack([neigh_pts1, np.ones((len(neigh_pts1), 1))])
        transformed_pts1 = (local_H @ neigh_pts1_h.T).T
        errors = np.linalg.norm(transformed_pts1 - neigh_pts2, axis=1)
        consistent = np.mean(errors) < error_thresh
        local_consistency.append(consistent)

    return local_consistency

def piecewise_affine_warp(pts_src, pts_dst, image):
    """
    Warp the given image from pts_src to pts_dst using Piecewise Affine Transform.
    pts_src: Nx2 array of control points in the moving image
    pts_dst: Nx2 array of control points in the fixed/reference image
    image: input 2D image to warp (grayscale or single-channel)
    """
    # Define the transform
    tform = PiecewiseAffineTransform()
    tform.estimate(pts_src, pts_dst)

    # Warp the image
    warped_image = warp(image, tform, output_shape=image.shape, order=1, mode='constant', cval=0)

    # Convert to original dtype (e.g. uint8)
    if np.issubdtype(image.dtype, np.integer):
        warped_image = (warped_image * 255).astype(image.dtype)
    return warped_image


def tps_warp(pts_src, pts_dst, image_to_warp):
    """
    Apply Thin Plate Spline (TPS) warp from source to destination points.

    Parameters:
    - pts_src: Nx2 array of control points in the source image (moving image)
    - pts_dst: Nx2 array of control points in the target image (reference image)
    - image_to_warp: 2D NumPy array (grayscale) to warp

    Returns:
    - warped_image: 2D NumPy array of the warped image
    """
    # Separate coordinates
    x_src, y_src = pts_src[:, 0], pts_src[:, 1]
    x_dst, y_dst = pts_dst[:, 0], pts_dst[:, 1]

    # Create RBF interpolators
    rbf_x = Rbf(x_src, y_src, x_dst, function='thin_plate')
    rbf_y = Rbf(x_src, y_src, y_dst, function='thin_plate')

    # Create meshgrid of image_to_warp coordinates
    h, w = image_to_warp.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    points_grid = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Warp all points
    x_warped = rbf_x(points_grid[:, 0], points_grid[:, 1])
    y_warped = rbf_y(points_grid[:, 0], points_grid[:, 1])
    warped_points = np.vstack([x_warped, y_warped]).T

    # Build remap grids
    map_x = warped_points[:, 0].reshape(h, w).astype(np.float32)
    map_y = warped_points[:, 1].reshape(h, w).astype(np.float32)

    # Warp image
    warped_image = cv2.remap(
        image_to_warp,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_image


