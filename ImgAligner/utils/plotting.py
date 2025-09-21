
import cv2
import numpy as np
import matplotlib.pyplot as plt

def _ensure_uint8_image(img):
    """
    Convert input image to uint8 (grayscale or color) suitable for cv2.drawMatches.
    Supports uint8 (no-op), uint16, float types, and other integer types.
    """
    if img is None:
        return img
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
    if np.issubdtype(img.dtype, np.floating):
        imgf = np.nan_to_num(img, copy=False)
        min_val = float(np.min(imgf))
        max_val = float(np.max(imgf))
        if max_val > min_val:
            imgf = (imgf - min_val) / (max_val - min_val)
        else:
            imgf = np.zeros_like(imgf, dtype=np.float32)
        return cv2.convertScaleAbs(imgf, alpha=255.0)
    # Fallback: normalize other integer types to 0..255
    imgf = img.astype(np.float32)
    min_val = float(np.min(imgf))
    max_val = float(np.max(imgf))
    if max_val > min_val:
        imgf = (imgf - min_val) / (max_val - min_val)
    else:
        imgf = np.zeros_like(imgf, dtype=np.float32)
    return cv2.convertScaleAbs(imgf, alpha=255.0)

def plot_inlier_matches(img1, img2, kp1, kp2, matches, title="Inlier Matches", figsize=(12, 8)):
    """
    Draw and display inlier feature matches between two images.

    Parameters:
    - img1, img2: Input images
    - kp1, kp2: Keypoints from img1 and img2
    - matches: List of cv2.DMatch objects (e.g., inlier matches)
    - title: Title for the plot
    - figsize: Size of the matplotlib figure

    Returns:
    - img_matches: Image with drawn matches (BGR format)
    """
    img1_u8 = _ensure_uint8_image(img1)
    img2_u8 = _ensure_uint8_image(img2)
    img_matches = cv2.drawMatches(
        img1_u8, kp1, img2_u8, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_overlay(tile_roi1, tile_roi2, figsize=(10, 10), title="Overlay: Reference (Green) & Registered (Magenta)"):
    """
    Create and plot an RGB overlay image:
    - Reference image in green channel
    - Registered image in magenta (red + blue channels)
    
    Parameters:
    - tile_roi1: 2D array, reference grayscale image
    - tile_roi2: 2D array, registered grayscale image
    - figsize: tuple, figure size
    - title: str, plot title
    """
    # Ensure images are float32 for overlay composition
    tile_roi1 = tile_roi1.astype(np.float32)
    tile_roi2 = tile_roi2.astype(np.float32)

    # Initialize empty RGB overlay
    overlay = np.zeros((tile_roi1.shape[0], tile_roi1.shape[1], 3), dtype=np.float32)
    overlay[..., 1] = tile_roi1  # Green channel for reference
    overlay[..., 0] = tile_roi2  # Red channel for registered
    overlay[..., 2] = tile_roi2  # Blue channel for registered

    # Normalize overlay to [0, 255]
    max_val = np.max(overlay)
    if max_val == 0:
        max_val = 1  # avoid division by zero

    overlay_uint8 = cv2.convertScaleAbs(overlay, alpha=(255.0 / max_val))

    # Plot
    fig = plt.figure(figsize=figsize)
    plt.imshow(overlay_uint8)
    plt.title(title)
    plt.axis('off')
    # plt.show()
    return fig

def plot_match_consistency(tile_roi1, tile_roi2, kp1, kp2, consistent_matches, inconsistent_matches, title="Matches: Green = consistent; Magenta = inconsistent"):
    """
    Plot consistent and inconsistent feature matches between two image tiles.

    Parameters:
    - tile_roi1: Grayscale or RGB image patch from image 1.
    - tile_roi2: Grayscale or RGB image patch from image 2.
    - kp1: Keypoints from image 1.
    - kp2: Keypoints from image 2.
    - consistent_matches: List of cv2.DMatch objects for consistent matches.
    - inconsistent_matches: List of cv2.DMatch objects for inconsistent matches.
    - title: Title of the plot (default provided).
    """
    
    # Ensure images are uint8 for drawing
    tile_roi1_u8 = _ensure_uint8_image(tile_roi1)
    tile_roi2_u8 = _ensure_uint8_image(tile_roi2)

    # Draw consistent matches in green
    img_matches = cv2.drawMatches(
        tile_roi1_u8, kp1,
        tile_roi2_u8, kp2,
        consistent_matches, None,
        matchColor=(0, 255, 0),  # Green
        singlePointColor=None,
        matchesThickness=8,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Draw inconsistent matches in magenta (red + blue)
    img_matches = cv2.drawMatches(
        tile_roi1_u8, kp1,
        tile_roi2_u8, kp2,
        inconsistent_matches, img_matches,
        matchColor=(255, 0, 255),  # Magenta
        singlePointColor=None,
        matchesThickness=8,
        flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Convert BGR to RGB for matplotlib
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    # Plot
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_point_mappings(pts1, pts2, background_img1=None, background_img2=None, title="Point correspondences", alpha1=0.8, alpha2=0.8):
    plt.figure(figsize=(10, 10))

    # Plot first background image if provided
    if background_img1 is not None:
        plt.imshow(background_img1, cmap='gray', alpha=alpha1)

    # Plot second background image if provided
    if background_img2 is not None:
        plt.imshow(background_img2, cmap='hot', alpha=alpha2)  # or another colormap

    # Plot pts1 as red dots
    plt.scatter(pts1[:, 0], pts1[:, 1], color='red', label='pts1 (reference)', s=10)

    # Plot pts2 as blue arrows starting from pts1
    deltas = pts2 - pts1
    plt.quiver(
        pts1[:, 0], pts1[:, 1],   # origins
        deltas[:, 0], deltas[:, 1],  # displacements
        angles='xy', scale_units='xy', scale=1, color='blue', width=0.0025
    )

    plt.legend()
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    #plt.gca().invert_yaxis()
    plt.show()


def plot_overlay(tile_roi1, tile_roi2, pts=None, figsize=(10, 10), 
                 title="Overlay: Reference (Green) & Registered (Magenta)"):
    """
    Create and plot an RGB overlay image:
    - Reference image in green channel
    - Registered image in magenta (red + blue channels)
    - Optionally plot points used for registration
    
    Parameters:
    - tile_roi1: 2D array, reference grayscale image
    - tile_roi2: 2D array, registered grayscale image
    - pts1: Nx2 array of points in reference image (optional)
    - pts2: Nx2 array of points in registered image (optional)
    - figsize: tuple, figure size
    - title: str, plot title
    """
    # Ensure images are float32 for overlay composition
    tile_roi1 = tile_roi1.astype(np.float32)
    tile_roi2 = tile_roi2.astype(np.float32)

    # Initialize empty RGB overlay
    overlay = np.zeros((tile_roi1.shape[0], tile_roi1.shape[1], 3), dtype=np.float32)
    overlay[..., 1] = tile_roi1  # Green channel for reference
    overlay[..., 0] = tile_roi2  # Red channel for registered
    overlay[..., 2] = tile_roi2  # Blue channel for registered

    # Normalize overlay to [0, 255]
    max_val = np.max(overlay)
    if max_val == 0:
        max_val = 1
    overlay_uint8 = cv2.convertScaleAbs(overlay, alpha=(255.0 / max_val))

    # Plot
    fig = plt.figure(figsize=figsize)
    plt.imshow(overlay_uint8)
    plt.title(title)
    plt.axis('off')

    # # Plot points if provided
    # if pts is not None:
    #     plt.scatter(pts[:, 0], pts[:, 1], s=20, c='lime', marker='o', label='Reference points')
    #     plt.legend(loc='upper right')

    return fig
