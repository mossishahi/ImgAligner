import torch
import torch.nn as nn
import cv2
import numpy as np

SUPERPOINT_PATH = '/home/icb/mostafa.shahhosseini/models/superpoint/superpoint_v1.pth'

def match_features(kp1, des1, kp2, des2, ratio_thresh=0.8):
    """Match features using BF + Lowe's ratio test"""
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = []
    for m in matches:
        if m.distance < ratio_thresh * matches[0].distance:
            good_matches.append(m)
    return good_matches, kp1, kp2
    
def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = (scores == max_pool(scores))
    for _ in range(2):
        supp_mask = (max_pool(max_mask.float()) > 0)
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = (supp_scores == max_pool(supp_scores))
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2018. https://arxiv.org/abs/1712.07629

    """
    def __init__(self, config={}):
        super().__init__()
        self.config = {
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
            'remove_borders': 4,
        }
        self.config.update(config)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = SUPERPOINT_PATH
        self.load_state_dict(torch.load(path))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or -1')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        } 

def superpoint_feature_matching(tile_roi1, tile_roi2, model=None, device=None, ratio_thresh=0.8, ransac_thresh=500.0, verbose=True):
    """
    Perform SuperPoint feature detection and matching between two images, then filter matches by
    Lowe's ratio test and RANSAC.

    Returns the same tuple format as `akaze_feature_matching`:
        (H, inlier_matches, pts1, pts2, kp1, kp2)
    """
    # Resolve defaults from globals if not provided
    if model is None:
        if 'superpoint_model' in globals():
            model = globals()['superpoint_model']
        else:
            raise RuntimeError("SuperPoint model is not provided and not found as 'superpoint_model'.")
    if device is None:
        if 'device' in globals():
            device = globals()['device']
        else:
            device = 'cpu'

    # Ensure grayscale inputs for SuperPoint
    if tile_roi1.ndim == 3 and tile_roi1.shape[2] == 3:
        img1_gray = cv2.cvtColor(tile_roi1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = tile_roi1
    if tile_roi2.ndim == 3 and tile_roi2.shape[2] == 3:
        img2_gray = cv2.cvtColor(tile_roi2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = tile_roi2

    # Convert numpy arrays to torch tensors [1, 1, H, W] and normalize to float32
    img1_tensor = torch.from_numpy(img1_gray).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0
    img2_tensor = torch.from_numpy(img2_gray).unsqueeze(0).unsqueeze(0).to(device).float() / 255.0

    # Run the model
    with torch.no_grad():
        pred1 = model({'image': img1_tensor})
        pred2 = model({'image': img2_tensor})

    # Extract keypoints and descriptors
    kps1_t = pred1['keypoints'][0]            # (N1, 2) tensor of (x, y)
    kps2_t = pred2['keypoints'][0]            # (N2, 2)
    des1_t = pred1['descriptors'][0].t()      # (N1, 256)
    des2_t = pred2['descriptors'][0].t()      # (N2, 256)

    kps1_np = kps1_t.cpu().numpy()
    kps2_np = kps2_t.cpu().numpy()
    des1 = np.ascontiguousarray(des1_t.cpu().numpy().astype(np.float32))
    des2 = np.ascontiguousarray(des2_t.cpu().numpy().astype(np.float32))

    # Convert to OpenCV KeyPoint lists for downstream visualization compatibility
    kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kps1_np]
    kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kps2_np]

    if len(kp1) == 0 or len(kp2) == 0:
        raise RuntimeError("No SuperPoint keypoints detected in one of the images.")

    # Match features with BF + Lowe's ratio test
    good_matches, pts1, pts2 = match_features(kp1, des1, kp2, des2, ratio_thresh=ratio_thresh)

    if len(good_matches) < 4:
        raise RuntimeError("Not enough good matches for homography estimation.")

    # Estimate homography using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        raise RuntimeError("Homography estimation failed.")

    # Filter matches by inliers mask (ensure 1D boolean)
    mask = mask.ravel().astype(bool)
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]

    # Update matched points for inliers only
    pts1_inliers = pts1[mask]
    pts2_inliers = pts2[mask]

    if verbose:
        print(f"Total matches: {len(good_matches)}; Inliers after RANSAC: {len(inlier_matches)}")

    return H, inlier_matches, pts1_inliers, pts2_inliers, kp1, kp2
