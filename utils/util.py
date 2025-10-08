from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import os
from skimage.io import imsave
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def update_losses_log(losses_log, losses):
    for key in losses:
        if key not in losses_log:
            losses_log[key] = []
        losses_log[key].append(losses[key].item())
    return losses_log


def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def check_mkdir(path):
    if not os.path.exists(path):
        print('making %s' % path)
        os.makedirs(path)


def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2) ** 2).sum(2)).mean(1).mean()


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def loss_plotting(title, iters: List[int], losses: Dict[str, List[float]], save_path=None):
    plt.figure(figsize=(10, 5))
    plt.title(title, fontsize=24)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Loss", fontsize=20)

    # Plot each type of loss from the dictionary
    for label, values in losses.items():
        plt.plot(iters, values, label=label)

    # Add a legend to clarify which curve corresponds to which loss
    plt.legend(fontsize=14)

    # Save or show the plot
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def batch_rodrigues(theta):
    # theta N x 3
    batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def batch_orth_proj(X, camera):
    '''
        X is N x num_points x 3
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
    shape = X_trans.shape
    # Xn = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn


def batch_persp_proj(vertices, cam, f, t, orig_size=256, eps=1e-9):
    '''
    Calculate projective transformation of vertices given a projection matrix
    Input parameters:
    f: torch tensor of focal length
    t: batch_size * 1 * 3 xyz translation in world coordinate
    K: batch_size * 3 * 3 intrinsic camera matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
    dist_coeffs: vector of distortion coefficients
    orig_size: original size of image captured by the camera
    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''
    device = vertices.device

    K = torch.tensor([f, 0., cam['c'][0], 0., f, cam['c'][1], 0., 0., 1.]).view(3, 3)[None, ...].repeat(
        vertices.shape[0], 1).to(device)
    R = batch_rodrigues(cam['r'][None, ...].repeat(vertices.shape[0], 1)).to(device)
    dist_coeffs = cam['k'][None, ...].repeat(vertices.shape[0], 1).to(device)

    vertices = torch.matmul(vertices, R.transpose(2, 1)) + t
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + eps)
    y_ = y / (z + eps)

    # Get distortion coefficients from vector
    k1 = dist_coeffs[:, None, 0]
    k2 = dist_coeffs[:, None, 1]
    p1 = dist_coeffs[:, None, 2]
    p2 = dist_coeffs[:, None, 3]
    k3 = dist_coeffs[:, None, 4]

    # we use x_ for x' and x__ for x'' etc.
    r = torch.sqrt(x_ ** 2 + y_ ** 2)
    x__ = x_ * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)) + 2 * p1 * x_ * y_ + p2 * (r ** 2 + 2 * x_ ** 2)
    y__ = y_ * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)) + p1 * (r ** 2 + 2 * y_ ** 2) + 2 * p2 * x_ * y_
    vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
    vertices = torch.matmul(vertices, K.transpose(1, 2))
    u, v = vertices[:, :, 0], vertices[:, :, 1]
    v = orig_size - v
    # map u,v from [0, img_size] to [-1, 1] to be compatible with the renderer
    u = 2 * (u - orig_size / 2.) / orig_size
    v = 2 * (v - orig_size / 2.) / orig_size
    vertices = torch.stack([u, v, z], dim=-1)

    return vertices


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                   vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                   vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                   vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


def tensor_vis_landmarks(images, landmarks, landmarks_conf=None, gt_landmarks=None, color='g', isScale=True):
    # visualize landmarks
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()
    if gt_landmarks is not None:
        gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]].copy();
        image = (image * 255)
        if isScale:
            predicted_landmark = predicted_landmarks[i] * image.shape[0] / 2 + image.shape[0] / 2
        else:
            predicted_landmark = predicted_landmarks[i]
        predicted_landmark_conf = landmarks_conf[i] if landmarks_conf is not None else None

        if predicted_landmark.shape[0] == 68:
            image_landmarks = plot_kpts(image, predicted_landmark, predicted_landmark_conf, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(image_landmarks,
                                             gt_landmarks_np[i] * image.shape[0] / 2 + image.shape[0] / 2, 'r')
        else:
            image_landmarks = plot_verts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(image_landmarks,
                                             gt_landmarks_np[i] * image.shape[0] / 2 + image.shape[0] / 2, 'r')

        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(
        vis_landmarks[:, :, :, [2, 1, 0]].transpose(0, 3, 1, 2)) / 255.  # , dtype=torch.float32)
    return vis_landmarks


end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def plot_kpts(image, kpts, kpts_conf=None, color='r'):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpts: (68, 3).
        kpts_conf: (68).
    '''
    if color == 'r':
        c = (0, 0, 255)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1] == 4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        if kpts_conf is not None:
            if kpts_conf[i] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        image = cv2.circle(image, (int(st[0]), int(st[1])), 1, c, 2)
        if i in end_list:
            continue
        ed = kpts[i + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), 1)

    return image


def save_obj(filename, vertices, faces, textures=None, uvcoords=None, uvfaces=None, texture_type='surface'):
    assert vertices.ndimension() == 2
    assert faces.ndimension() == 2
    assert texture_type in ['surface', 'vertex']
    # assert texture_res >= 2

    if textures is not None and texture_type == 'surface':
        textures = textures.detach().cpu().numpy().transpose(1, 2, 0)
        filename_mtl = filename[:-4] + '.mtl'
        filename_texture = filename[:-4] + '.png'
        material_name = 'material_1'
        # texture_image, vertices_textures = create_texture_image(textures, texture_res)
        texture_image = textures
        texture_image = texture_image.clip(0, 1)
        texture_image = (texture_image * 255).astype('uint8')
        imsave(filename_texture, texture_image)

    faces = faces.detach().cpu().numpy()

    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')

        if textures is not None:
            f.write('mtllib %s\n\n' % os.path.basename(filename_mtl))

        if textures is not None and texture_type == 'vertex':
            for vertex, color in zip(vertices, textures):
                f.write('v %.8f %.8f %.8f %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2],
                                                               color[0], color[1], color[2]))
            f.write('\n')
        else:
            for vertex in vertices:
                f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
            f.write('\n')

        if textures is not None and texture_type == 'surface':
            for vertex in uvcoords.reshape((-1, 2)):
                f.write('vt %.8f %.8f\n' % (vertex[0], vertex[1]))
            f.write('\n')

            f.write('usemtl %s\n' % material_name)
            for i, face in enumerate(faces):
                f.write('f %d/%d %d/%d %d/%d\n' % (
                    face[0] + 1, uvfaces[i, 0] + 1, face[1] + 1, uvfaces[i, 1] + 1, face[2] + 1, uvfaces[i, 2] + 1))
            f.write('\n')
        else:
            for face in faces:
                f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))

    if textures is not None and texture_type == 'surface':
        with open(filename_mtl, 'w') as f:
            f.write('newmtl %s\n' % material_name)
            f.write('map_Kd %s\n' % os.path.basename(filename_texture))


def face_seg(img, net, cropped_size: int):
    device = next(net.parameters()).device
    face_area = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resize_pil_image = pil_image.resize((512, 512), Image.BILINEAR)
    tensor_image = to_tensor(resize_pil_image)
    tensor_image = torch.unsqueeze(tensor_image, 0)
    tensor_image = tensor_image.to(device)
    out = net(tensor_image)[0]
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    vis_parsing_anno = parsing.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]))
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        if pi in face_area:
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1]] = 1
    image_mask = cv2.resize(vis_parsing_anno_color, (cropped_size, cropped_size))
    image_mask = image_mask[..., None].astype('float32')
    image_mask = image_mask.transpose(2, 0, 1)
    image_mask_bn = np.zeros_like(image_mask)
    image_mask_bn[np.where(image_mask != 0)] = 1.

    return image_mask_bn[None, :, :, :]


def crop_img(ori_image, rect, cropped_size, kpts=None):
    image = ori_image.copy()
    l, t, r, b = rect
    center_x = int(r - (r - l) // 2)
    center_y = int(b - (b - t) // 2)
    w = int((r - l) * 1.2)
    h = int((b - t) * 1.2)
    crop_size = max(w, h, cropped_size)  # Ensure crop_size is at least as large as cropped_size

    # Calculate padding needs
    pad_top = max(0, -(center_y - crop_size // 2))
    pad_left = max(0, -(center_x - crop_size // 2))
    pad_bottom = max(0, (center_y + crop_size // 2) - image.shape[0])
    pad_right = max(0, (center_x + crop_size // 2) - image.shape[1])

    # Pad the image if necessary
    if any([pad_top, pad_left, pad_bottom, pad_right]):
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    # Update center coordinates to new padded image dimensions
    center_y += pad_top
    center_x += pad_left

    # Recalculate crop coordinates based on padded image
    crop_ly = int(center_y - crop_size // 2)
    crop_lx = int(center_x - crop_size // 2)

    crop_image = image[crop_ly: crop_ly + crop_size, crop_lx: crop_lx + crop_size, :]

    # Calculate the new rectangle in the cropped image coordinates
    new_rect = [l + pad_left - crop_lx, t + pad_top - crop_ly, r + pad_left - crop_lx, b + pad_top - crop_ly]

    # Adjust keypoints if provided
    if kpts is not None:
        new_kpts = kpts.copy()
        new_kpts[:, 0] = kpts[:, 0] + pad_left - crop_lx
        new_kpts[:, 1] = kpts[:, 1] + pad_top - crop_ly
        return crop_image, new_rect, new_kpts

    return crop_image, new_rect


def resize_para(ori_frame):
    w, h, c = ori_frame.shape
    d = max(w, h)
    scale_to = 640 if d >= 1280 else d / 2
    scale_to = max(64, scale_to)
    input_scale = d / scale_to
    w = int(w / input_scale)
    h = int(h / input_scale)
    image_info = [w, h, input_scale]
    return image_info


def draw_train_process(title, iters: List[int], loss: Dict[str, List[float]], label_loss, save_path=None):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.plot(iters, loss, color='red', label=label_loss)
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
