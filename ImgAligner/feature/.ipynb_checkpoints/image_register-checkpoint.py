import cv2
import numpy as np
from tqdm import tqdm
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.interpolate import Rbf
from sklearn.neighbors import NearestNeighbors
import numpy as np
from ..utils.image_transform import *
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors

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

    return tile_roi1_cropped, tile_roi2_cropped, tile_roi1, tile_roi2




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


def check_local_consistency_df(df, n_neighbors=5, error_thresh=3.0,tag='global'):
    """
    Check local geometric consistency of point correspondences in a matches DataFrame
    using local affine transforms.

    Args:
        df (pd.DataFrame): Must contain columns ['x1_roi', 'y1_roi', 'x2_roi', 'y2_roi'].
        n_neighbors (int): Number of neighbors for local transformation.
        error_thresh (float): Pixel error threshold for consistency.

    Returns:
        np.ndarray: Boolean array of consistency flags for each row in the DataFrame.
    """
    pts1 = df[['x1_'+tag, 'y1_'+tag]].to_numpy()
    pts2 = df[['x2_'+tag, 'y2_'+tag]].to_numpy()
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(pts1)
    local_consistency = []

    for i in range(len(pts1)):
        pt1 = pts1[i].reshape(1, -1)
        pt2 = pts2[i].reshape(1, -1)

        distances, indices = nbrs.kneighbors(pt1)
        neighbor_indices = indices[0]

        neigh_pts1 = pts1[neighbor_indices]
        neigh_pts2 = pts2[neighbor_indices]

        # Estimate local affine transform
        local_H, inliers = cv2.estimateAffinePartial2D(
            neigh_pts1, neigh_pts2, method=cv2.RANSAC, ransacReprojThreshold=error_thresh
        )

        if local_H is None:
            local_consistency.append(False)
            continue

        # Transform neighbor points and compute error
        neigh_pts1_h = np.hstack([neigh_pts1, np.ones((len(neigh_pts1), 1))])
        transformed_pts1 = (local_H @ neigh_pts1_h.T).T
        errors = np.linalg.norm(transformed_pts1 - neigh_pts2, axis=1)
        consistent = np.mean(errors) < error_thresh
        local_consistency.append(consistent)

    return np.array(local_consistency)

