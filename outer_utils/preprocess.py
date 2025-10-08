import numpy as np
import cv2
import torch
from plyfile import PlyData

from .joint_mapping import get_openpose_part


# codes from https://github.com/mks0601/Hand4Whole_RELEASE/blob/main/common/utils/preprocessing.py
# and https://github.com/haofanwang/CLIFF/blob/main/common/imutils.py
def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img


def load_obj(file_name):
    v = []
    obj_file = open(file_name)
    for line in obj_file:
        words = line.split(' ')
        if words[0] == 'v':
            x, y, z = float(words[1]), float(words[2]), float(words[3])
            v.append(np.array([x, y, z]))
    return np.stack(v)


def load_ply(file_name):
    plydata = PlyData.read(file_name)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    v = np.stack((x, y, z), 1)
    return v


def compute_bbox(keypoints_list):
    all_keypoints = keypoints_list
    bbox_list = []

    for batch_id, keypoints in enumerate(all_keypoints):
        visible_keypoints = keypoints[keypoints[:, 2] > 0]

        if len(visible_keypoints) == 0:
            continue

        x_coords = visible_keypoints[:, 0]
        y_coords = visible_keypoints[:, 1]

        min_x = np.min(x_coords)
        min_y = np.min(y_coords)
        max_x = np.max(x_coords)
        max_y = np.max(y_coords)

        # [batch_id, min_x, min_y, max_x, max_y]
        bbox = [batch_id, min_x, min_y, max_x, max_y]
        bbox_list.append(bbox)

    bbox_array = np.array(bbox_list)
    return bbox_array


def get_best_hand(keypoints_list, hand='rhand'):
    hand_idx = get_openpose_part(hand)
    all_conf = []
    for keypoints in keypoints_list:
        hand_keypoints = keypoints[hand_idx]
        mean_conf = np.mean(hand_keypoints[:, 2])
        # sanity check
        x_coords = hand_keypoints[:, 0]
        y_coords = hand_keypoints[:, 1]
        if abs(np.max(x_coords) - np.min(x_coords)) < 2.0 or abs(np.max(y_coords) - np.min(y_coords)) < 2.0:
            mean_conf = 0
        all_conf.append(mean_conf)
    if np.max(all_conf) == 0:
        raise ValueError('All hand keypoints are invisible')
    best_idx = np.argmax(all_conf)

    return best_idx


def get_best_face(keypoints_list, from_wholebody=False):
    face_idx = get_openpose_part('face')
    all_area = []
    for keypoints in keypoints_list:
        face_keypoints = keypoints[face_idx] if from_wholebody else keypoints
        face_area = (face_keypoints[:, 0].max() - face_keypoints[:, 0].min()) * (
                face_keypoints[:, 1].max() - face_keypoints[:, 1].min())
        all_area.append(face_area)
    best_idx = np.argmax(all_area)

    return best_idx


def crop_img_tensor(ori_image_tensor, rect, cropped_size, kpts=None):
    image_tensor = ori_image_tensor.clone()
    l, t, r, b = rect
    center_x = int(r - (r - l) // 2)
    center_y = int(b - (b - t) // 2)
    w = int((r - l) * 1.2)
    h = int((b - t) * 1.2)
    crop_size = max(w, h, cropped_size)  # Ensure crop_size is at least as large as cropped_size

    # Calculate padding needs
    pad_top = max(0, -(center_y - crop_size // 2))
    pad_left = max(0, -(center_x - crop_size // 2))
    pad_bottom = max(0, (center_y + crop_size // 2) - image_tensor.shape[1])
    pad_right = max(0, (center_x + crop_size // 2) - image_tensor.shape[2])

    # Pad the image if necessary using 'constant' mode equivalent in PyTorch
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    image_tensor = torch.nn.functional.pad(image_tensor, padding, mode='constant', value=0)

    # Update center coordinates to new padded image dimensions
    center_y += pad_top
    center_x += pad_left

    # Recalculate crop coordinates based on padded image
    crop_ly = int(center_y - crop_size // 2)
    crop_lx = int(center_x - crop_size // 2)

    # Crop the image using PyTorch tensor indexing
    crop_image_tensor = image_tensor[:, crop_ly: crop_ly + crop_size, crop_lx: crop_lx + crop_size]

    # Calculate the new rectangle in the cropped image coordinates
    new_rect = [l + pad_left - crop_lx, t + pad_top - crop_ly, r + pad_left - crop_lx, b + pad_top - crop_ly]

    # Adjust keypoints if provided
    if kpts is not None:
        new_kpts = kpts.copy()
        new_kpts[:, 0] = kpts[:, 0] + pad_left - crop_lx
        new_kpts[:, 1] = kpts[:, 1] + pad_top - crop_ly
        return crop_image_tensor, new_rect, new_kpts

    return crop_image_tensor, new_rect