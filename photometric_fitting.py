from typing import List, Dict
from tqdm import tqdm
import argparse
import json
import os, sys
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime

sys.path.append('.')
from models.FLAME import FLAME, FLAMETex
from utils.renderer import Renderer
from utils import util
from utils.config import cfg

from transformers import SegformerForSemanticSegmentation

from outer_utils.joint_mapping import vitpose_to_openpose, mmpose_to_openpose
from outer_utils.preprocess import get_best_face, get_openpose_part
from utils.face_parser import face_seg_hf

torch.backends.cudnn.benchmark = True


class PhotometricFitting(object):
    def __init__(self, device='cuda', vis_num=10, batchsize=1, verbose=True,
                 video_writer=None, savefolder="./output", save_static=True):
        self.verbose = verbose
        self.vis_num = vis_num
        self.batchsize = batchsize
        self.config = cfg
        self.device = device
        self.video_writer = video_writer
        self.savefolder = savefolder
        self.save_static = save_static
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)

        self._setup_renderer()
        self.prior_models = {}
        self.loss_weights = {'landmark_loss': [1.0, 1.0, 0.1, 0.1],
                             'photometric_loss': [50.0] * 4,
                             'pose_prior': [0.1, 0.1, 0.01, 0.01],
                             'expr_prior': [0.1, 0.1, 0.01, 0.01],
                             'shape_prior': [0.01, 0.01, 0.001, 0.001],
                             }
        self.stages = len(self.loss_weights['landmark_loss'])
        self.num_iters = self.config.stage_iter
        self.t_max = 0.12
        self.t_min = 0.08

        self.unreliable_indices = [17, 21, 22, 26]  # These annotations are not reliable
        self.reliable_indices = [1, 2, 3, 6, 7, 8, 13, 14, 15]   # Necessary face contour for fitting
        # coutour, eyebrow, upper_mouth, eye, mouth
        self.left_face_indices = (list(range(0, 7)) + list(range(17, 22)) + [31, 32]
                                  + list(range(36, 42)) + [48, 49, 59, 60, 61, 67])
        self.right_face_indices = (list(range(10, 17)) + list(range(22, 27)) + [34, 35]
                                   + list(range(42, 48)) + [53, 54, 55, 63, 64, 65])
        self.joints_conf_thsh = self.config.joints_conf_thsh

    def load_prior(self, prior_models: Dict[str, nn.Module]):
        support_keys = ['shape', 'expression', 'pose', 'pose_expression']
        assert all([key in support_keys for key in prior_models.keys()]), \
            f'Unsupported prior model keys {prior_models.keys()}'
        self.prior_models = prior_models
        print(f'Loaded prior models: {prior_models.keys()}')

    def _setup_renderer(self):
        self.render = Renderer(cfg.image_size, obj_filename=cfg.mesh_file).to(self.device)

    def sample_continuous_time(self, iteration):
        total_steps = self.stages * self.num_iters
        t = self.t_min + torch.tensor(total_steps - iteration - 1) / total_steps * (self.t_max - self.t_min)

        return t

    def initialize_parameters(self, batch_init, bz):
        """ Initialize the model parameters, potentially with batch data """
        shape = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(self.device))
        pose = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(self.device))
        cam = torch.zeros(bz, self.config.camera_params).float().to(self.device)
        cam[:, 0] = 5.  # Set initial camera parameter
        cam = nn.Parameter(cam)

        # Use initial batch data if provided
        if batch_init:
            for key, param in zip(['pose', 'cam'], [pose, cam]):
                if key in batch_init and batch_init[key] is not None:
                    param.data = batch_init[key].to(self.device)

            for key, param, param_dim in zip(['betas', 'exp'], [shape, exp],
                                             [self.config.shape_params, self.config.expression_params]):
                if key in batch_init and batch_init[key] is not None:
                    param_data = self.pad_data(batch_init[key], bz, param_dim)
                    param.data = param_data

        tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
        lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))

        # Split the pose as global and jaw pose for fine-grained optimization
        global_orient = nn.Parameter(pose[:, :3].data)
        jaw_pose = nn.Parameter(pose[:, 3:].data)

        return [shape, exp, jaw_pose, global_orient, cam, tex, lights]

    def pad_data(self, data, bz, target_params):
        """ Pad data if its dimension is less than the expected dimension """
        if data.shape[1] < target_params:
            padding = torch.zeros(bz, target_params - data.shape[1], device=self.device)
            data = torch.cat((data, padding), dim=1)
        return data

    def fit_camera(self, gt_landmark, joints_conf, batchsize, batch_init=None):
        shape, exp, jaw_pose, global_orient, cam, tex, lights = self.initialize_parameters(batch_init, batchsize)
        full_pose = torch.cat((global_orient, jaw_pose), dim=1)
        # Fit the camera translation first
        param_groups = [{'params': cam, 'lr': 0.5}]
        optimizer = torch.optim.Adam(param_groups)
        for k in range(100):
            vertices, landmarks2d, landmarks3d = self.flame(shape, exp, pose_params=full_pose)
            trans_vertices, landmarks2d, landmarks3d = self.transform_points(vertices, landmarks2d, landmarks3d, cam)
            landmark_loss = joints_conf * util.gmof(landmarks2d[:, :, :2] - gt_landmark[:, :, :2], sigma=50).sum(dim=-1)
            all_loss = landmark_loss.sum(dim=-1).mean()
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        # Then fit the global orient jointly
        param_groups = [{'params': global_orient, 'lr': 0.1},
                        {'params': cam, 'lr': 0.1}]
        optimizer = torch.optim.Adam(param_groups)
        for k in range(200):
            full_pose = torch.cat((global_orient, jaw_pose), dim=1)
            vertices, landmarks2d, landmarks3d = self.flame(shape, exp, pose_params=full_pose)
            trans_vertices, landmarks2d, landmarks3d = self.transform_points(vertices, landmarks2d, landmarks3d, cam)
            landmark_loss = joints_conf * util.gmof(landmarks2d[:, :, :2] - gt_landmark[:, :, :2], sigma=50).sum(dim=-1)
            all_loss = landmark_loss.sum(dim=-1).mean()
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        print('Camera fitting done')
        return global_orient.data.clone(), cam.data.clone()

    def fit(self, images, landmarks, image_masks, batch_id=0, batch_init=None):
        """ Fit the model to the image and landmark data """
        train_iters, losses_log = [], {}
        gt_landmark = landmarks[:, :, :2].clone()  # [b, 68, 3]
        joints_conf = landmarks[:, :, -1].clone()  # [b, 68]

        # binarize the confidence and strengthen the eyes
        joints_conf = (joints_conf > self.joints_conf_thsh).float()
        joints_conf[:, 36:48] *= 2.0
        joints_conf[:, 7:10] *= 2.0
        joints_conf[:, self.unreliable_indices] = 0.0
        joints_conf[:, self.reliable_indices] = 1.0
        stage_weights = [dict(zip(self.loss_weights.keys(), vals)) for vals in zip(*self.loss_weights.values())]

        # Fit the camera pose and cam first
        global_orient, cam = self.fit_camera(gt_landmark, joints_conf, images.shape[0], batch_init)

        # Decide whether the face are occluded
        left_indices = torch.where(global_orient[:, 1] < -1.0)[0]  # left face under occlusion
        if len(left_indices) > 0:
            joints_conf[left_indices[:, None], self.left_face_indices] = 0.0
        right_indices = torch.where(global_orient[:, 1] > 1.0)[0]  # right face under occlusion
        if len(right_indices) > 0:
            joints_conf[right_indices[:, None], self.right_face_indices] = 0.0
        print(f"Left face occlusion: {len(left_indices)}, Right face occlusion: {len(right_indices)}")
        vis_landmarks = torch.cat((landmarks[:, :, :2], joints_conf[:, :, None]), dim=-1)

        # Update the global_orient and cam trans
        params = self.initialize_parameters(batch_init, images.shape[0])
        params[3].data = global_orient
        params[4].data = cam
        # [shape, exp, jaw_pose, global_orient, cam, tex, lights]
        learning_rates = [0.005, 0.005, 0.01, 0.05, 0.0, 0.005, 0.01]
        param_groups = [{'params': param, 'lr': lr} for param, lr in zip(params, learning_rates)]
        optimizer = torch.optim.Adam(param_groups)

        # Fit all model parameters
        for stage, current_weights in enumerate(tqdm(stage_weights, desc='Stage')):
            pbar = tqdm(total=self.num_iters//10, desc=f'Iter (Stage {stage + 1})')
            if stage == len(stage_weights[:-1]) - 1:  # Last stage
                optimizer.param_groups[3]['lr'] = 0.01
                optimizer.param_groups[4]['lr'] = 0.002
            for iteration in range(self.num_iters):
                t = self.sample_continuous_time(iteration=stage * self.num_iters + iteration)
                losses, ops, albedos, verts_tuple = self.compute_losses(images, image_masks, gt_landmark, joints_conf,
                                                                        *params, current_weights, t)
                self.update_params(optimizer, losses)

                if iteration % 10 == 0:
                    self.log_progress(pbar, iteration, losses, losses_log, images, image_masks,
                                      vis_landmarks, ops, *verts_tuple)
                    train_iters.append(stage * self.num_iters + iteration)
            pbar.close()

        loss_fig_path = os.path.join(self.savefolder, f'loss_batch{batch_id}.svg')
        util.loss_plotting("fitting", train_iters, losses_log, save_path=loss_fig_path)
        fitted_params = self.collect_fitted_params(*params, verts_tuple[1], albedos)
        vis_grid = self.collect_vis_frame(images, image_masks, vis_landmarks, *verts_tuple)

        return fitted_params, vis_grid

    def compute_losses(self, images, image_masks, gt_landmark, joints_conf,
                       shape, exp, jaw_pose, global_orient, cam, tex, lights,
                       loss_weights, t):
        """ Calculate losses for the current model parameters """
        pose = torch.cat((global_orient, jaw_pose), dim=1)
        vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
        trans_vertices, landmarks2d, landmarks3d = self.transform_points(vertices, landmarks2d, landmarks3d, cam)

        # Compute landmark loss, sum along x-y, joints and then mean over batch
        errors = (landmarks2d[:, :, :2] - gt_landmark[:, :, :2])
        landmark_loss = (joints_conf * util.gmof(errors, sigma=50).sum(dim=-1)).sum(dim=-1).mean() * loss_weights['landmark_loss']
        # Render images and compute photometric texture loss
        albedos = self.flametex(tex) / 255.
        ops = self.render(vertices, trans_vertices, albedos, lights)
        photometric_texture_loss = F.smooth_l1_loss(image_masks * ops['images'], image_masks * images) * loss_weights['photometric_loss']
        # Compute the fitting loss
        fitting_loss = landmark_loss + photometric_texture_loss

        # Compute the prior loss
        prior_losses = self.compute_prior_losses(shape, exp, jaw_pose, loss_weights, t)

        all_loss = fitting_loss + sum(prior_losses.values())
        loss_log = {'all_loss': all_loss, 'photometric': photometric_texture_loss,
                    'landmark': landmark_loss, **prior_losses}

        return loss_log, ops, albedos, (vertices, trans_vertices, landmarks2d, landmarks3d)

    def compute_prior_losses(self, shape, exp, jaw_pose, prior_weights, t):
        prior_losses = {}
        if 'shape' in self.prior_models:
            prior_losses['shape_prior'] = self.prior_models['shape'](shape, t) * prior_weights['shape_prior']
        if 'pose_expression' in self.prior_models:
            input_data = torch.cat((jaw_pose, exp), dim=1)
            prior_losses['pose_expr_prior'] = self.prior_models['pose_expression'](input_data, t) * prior_weights['pose_prior']
        if 'expression' in self.prior_models:
            prior_losses['expr_prior'] = self.prior_models['expression'](exp, t) * prior_weights['expr_prior']
        if 'pose' in self.prior_models:
            prior_losses['pose_prior'] = self.prior_models['pose'](jaw_pose, t) * prior_weights['pose_prior']
        return prior_losses

    def transform_points(self, vertices, landmarks2d, landmarks3d, cam):
        trans_vertices = util.batch_orth_proj(vertices, cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]
        landmarks2d = util.batch_orth_proj(landmarks2d, cam)
        landmarks2d[..., 1:] = - landmarks2d[..., 1:]
        landmarks3d = util.batch_orth_proj(landmarks3d, cam)
        landmarks3d[..., 1:] = - landmarks3d[..., 1:]
        return trans_vertices, landmarks2d, landmarks3d

    def update_params(self, optimizer, losses):
        """ Update model parameters based on computed losses """
        optimizer.zero_grad()
        losses['all_loss'].backward()
        optimizer.step()

    def log_progress(self, pbar, iteration, losses, losses_log, images, image_masks,
                     landmarks, ops, vertices, trans_vertices, landmarks2d, landmarks3d):
        """ Log fitting progress and optionally save to video """
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        loss_info = f'iter: {iteration}, time: {current_time}'
        loss_details = ', '.join(f'{key}: {float(value):.4f}' for key, value in losses.items())
        loss_info += f', {loss_details}'

        if self.verbose:
            tqdm.write(loss_info)
        else:
            pbar.set_postfix_str(loss_details)
            pbar.update(1)
        util.update_losses_log(losses_log, losses)

        if self.video_writer:
            self.record_video_frame(images, image_masks, landmarks, ops, vertices, trans_vertices, landmarks2d, landmarks3d)

    def record_video_frame(self, images, image_masks, landmarks, ops, vertices, trans_vertices, landmarks2d, landmarks3d):
        grids = {}
        bz = images.shape[0]
        visind = range(torch.min(torch.tensor([bz, self.vis_num])))

        grids['images'] = torchvision.utils.make_grid(images[visind], nrow=1).detach().cpu()
        grids['image_masks'] = torchvision.utils.make_grid(image_masks[visind], nrow=1).detach().cpu()
        grids['landmarks_gt'] = torchvision.utils.make_grid(
            util.tensor_vis_landmarks(images[visind], landmarks[visind], landmarks[visind, :, 2]), nrow=1)
        grids['landmarks2d'] = torchvision.utils.make_grid(
            util.tensor_vis_landmarks(images[visind], landmarks2d[visind]), nrow=1)
        grids['landmarks3d'] = torchvision.utils.make_grid(
            util.tensor_vis_landmarks(images[visind], landmarks3d[visind]), nrow=1)
        grids['albedoimage'] = torchvision.utils.make_grid(
            (ops['albedo_images'])[visind].detach().cpu(), nrow=1)
        grids['render'] = torchvision.utils.make_grid(
            ops['images'][visind].detach().float().cpu(), nrow=1)
        shape_images = self.render.render_shape(vertices, trans_vertices, images)
        grids['shape'] = torchvision.utils.make_grid(
            F.interpolate(shape_images[visind], [cfg.image_size, cfg.image_size]), nrow=1).detach().float().cpu()

        grid = torch.cat(list(grids.values()), 2)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        self.video_writer.write(grid_image)

    def collect_fitted_params(self, shape, exp, jaw_pose, global_orient, cam, tex, lights, trans_vertices, albedos):
        """ Collect and return fitted parameters from the optimization """
        pose = torch.cat((global_orient, jaw_pose), dim=1)
        fitted_params = {
            'shape': shape.detach().cpu().numpy(),
            'exp': exp.detach().cpu().numpy(),
            'pose': pose.detach().cpu().numpy(),
            'cam': cam.detach().cpu().numpy(),
            'tex': tex.detach().cpu().numpy(),
            'lights': lights.detach().cpu().numpy(),
            'verts': trans_vertices.detach().cpu().numpy(),
            'albedos': albedos.detach().cpu().numpy(),
        }

        static_exp, static_pose = torch.zeros_like(exp), torch.zeros_like(pose)
        vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=static_exp,
                                                        pose_params=static_pose)
        fitted_params['static_verts'] = vertices.detach().cpu().numpy()
        fitted_params['static_landmarks3d'] = landmarks3d.detach().cpu().numpy()

        return fitted_params

    def collect_vis_frame(self, images, image_masks, landmarks, vertices, trans_vertices, landmarks2d, landmarks3d):
        """ Collect and return visualization frames from the optimization """
        shape_images = self.render.render_shape(vertices, trans_vertices, images)
        vis_frames = {
            'images': images.detach().cpu(),
            'image_masks': image_masks.repeat(1, 3, 1, 1).detach().cpu(),
            'landmarks_gt': util.tensor_vis_landmarks(images, landmarks, landmarks[:, :, 2]),
            'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
            'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
            'shape': shape_images.detach().float().cpu(),
        }
        # concat them together
        grid = torch.cat(list(vis_frames.values()), 3)  # [b, c, h, w*6]
        return grid

    def fetch_dict(self, dict, start, end, selected_keys=None):
        sub_dict = {}
        if selected_keys is None:
            selected_keys = dict.keys()
        for key in selected_keys:
            sub_dict[key] = dict[key][start:end]
        return sub_dict

    def run(self, imgs, kpts, base_paths, seg_net, init_results=None):
        batch_images, batch_landmarks, batch_image_masks = [], [], []
        dataset_len = len(imgs)
        fitted_params = {}

        for idx, img in enumerate(imgs):
            kpt = kpts[idx]  # [68, 3]
            bbox = [np.min(kpt[:, 0]), np.min(kpt[:, 1]), np.max(kpt[:, 0]), np.max(kpt[:, 1])]
            # bbox format: [left, top, right, bottom]
            crop_image, new_bbox, kpt = util.crop_img(img, bbox, cfg.cropped_size, kpt)

            landmark = kpt  # [68, 3]
            landmark[:, 0] = landmark[:, 0] / float(crop_image.shape[1]) * 2 - 1
            landmark[:, 1] = landmark[:, 1] / float(crop_image.shape[0]) * 2 - 1
            batch_landmarks.append(torch.from_numpy(landmark)[None, :, :].double().to(self.device))

            # Prepare images
            image = cv2.resize(crop_image, (cfg.cropped_size, cfg.cropped_size)).astype(np.float32) / 255.
            image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            batch_images.append(torch.from_numpy(image[None, :, :, :]).double().to(self.device))

            # Segmentation masks
            image_mask = face_seg_hf(crop_image, seg_net, cfg.cropped_size)
            if isinstance(image_mask, np.ndarray):
                batch_image_masks.append(torch.from_numpy(image_mask).double().to(self.device))
            else:   # tensor
                batch_image_masks.append(image_mask)

            # Process as a batch
            if idx % self.batchsize == self.batchsize - 1 or idx == dataset_len - 1:
                batch_landmarks = torch.cat(batch_landmarks, dim=0)
                batch_images = torch.cat(batch_images, dim=0)
                batch_images = F.interpolate(batch_images, [cfg.image_size, cfg.image_size])
                batch_image_masks = torch.cat(batch_image_masks, dim=0)
                batch_image_masks = F.interpolate(batch_image_masks, [cfg.image_size, cfg.image_size])

                if init_results is not None:
                    batch_init = self.fetch_dict(init_results, start=(idx // self.batchsize) * self.batchsize,
                                                 end=idx + 1)
                else:
                    batch_init = None
                batch_id = idx // self.batchsize
                # ============================   Core fitting process   ============================
                batch_fitted_params, batch_vis_grid = self.fit(batch_images, batch_landmarks, batch_image_masks, batch_id, batch_init)
                # ============================   Core fitting process   ============================
                # save the visualization results
                batch_base_paths = base_paths[(idx // self.batchsize) * self.batchsize: idx + 1]
                for i, base_path in enumerate(batch_base_paths):
                    prefix_directory = '/'.join(base_path.split('/')[:-1])
                    os.makedirs(os.path.join(self.savefolder, prefix_directory), exist_ok=True)
                    vis_path = os.path.join(self.savefolder, f'{base_path}.jpg')
                    torchvision.utils.save_image(batch_vis_grid[i], vis_path)

                for key in batch_fitted_params.keys():
                    if key not in fitted_params:
                        fitted_params[key] = []
                    fitted_params[key].append(batch_fitted_params[key])

                batch_images, batch_landmarks, batch_image_masks = [], [], []

        for key in fitted_params.keys():
            fitted_params[key] = np.concatenate(fitted_params[key], axis=0)

        for i, base_path in enumerate(base_paths):
            # save fitted obj and vis results
            prefix_directory = '/'.join(base_path.split('/')[:-1])
            os.makedirs(os.path.join(self.savefolder, prefix_directory), exist_ok=True)
            if self.save_static:  # results for Now evaluation
                self.render.save_obj(filename=os.path.join(self.savefolder, f'{base_path}.obj'),
                                     vertices=torch.from_numpy(fitted_params['static_verts'][i]).to(self.device),
                                     textures=None)
                kpt3d = fitted_params['static_landmarks3d'][i]
                indices = [x + 17 for x in [19, 22, 25, 28, 16, 31, 37]]
                kpt3d = kpt3d[indices]
                np.savetxt(os.path.join(self.savefolder, f'{base_path}.txt'), kpt3d)
            else:
                self.render.save_obj(filename=os.path.join(self.savefolder, f'{base_path}.obj'),
                                     vertices=torch.from_numpy(fitted_params['verts'][i]).to(self.device),
                                     textures=torch.from_numpy(fitted_params['albedos'][i]).to(self.device))


def main_demo(args):
    """
    A demo function to run photometric fitting on two specific samples
    from the NoW benchmark.
    """
    # --- 1. Define Output Folder and Demo Samples ---
    save_folder = args.output_folder
    save_video_path = os.path.join(save_folder, 'vis_video.avi')
    os.makedirs(save_folder, exist_ok=True)
    print(f"Results will be saved in: {save_folder}")

    select_list = [
        'IMG_0067.jpg',
        'IMG_7344.jpg'
    ]

    # --- 2. Prepare Paths and Load Data ---
    rgb_dir = os.path.join(args.data_dir, 'images')

    img_paths = [os.path.join(rgb_dir, x) for x in select_list]
    base_paths = [os.path.splitext(p)[0] for p in select_list]
    
    print(f"Processing {len(img_paths)} images...")
    
    # Load BGR images
    imgs = [cv2.imread(img_path) for img_path in img_paths]

    # Load keypoints
    all_kpts = []
    face_idx = get_openpose_part('face')
    for path in base_paths:
        # 'mmpose_face' keypoints
        kpt_path = os.path.join(args.data_dir, 'landmarks', f'{path}.json')
        if not os.path.exists(kpt_path):
            raise FileNotFoundError(f"Keypoint file not found: {kpt_path}")
        
        json_data = json.load(open(kpt_path))
        keypoints = np.array(json_data[0]['keypoints'])
        keypoint_scores = np.array(json_data[0]['keypoint_scores'])
        converted_keypoints = np.hstack((keypoints, keypoint_scores.reshape(-1, 1)))
        
        all_kpts.append(converted_keypoints)

    # --- 3. Setup Models and Fitting Class ---
    # Load Segmentation Network
    seg_net = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    seg_net = seg_net.to(args.device)

    # Initialize Fitting Class
    if args.save_video:
        vis_image_num = min(args.max_vis_num, args.batchsize)
        video_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 16,
                                       ((cfg.image_size + 4) * 8, (cfg.image_size + 2) * vis_image_num + 2))
    else:
        video_writer = None
    fitting = PhotometricFitting(device=args.device, vis_num=args.max_vis_num, verbose=args.verbose,
                                 batchsize=args.batchsize, video_writer=video_writer,
                                 savefolder=save_folder, save_static=args.save_static)

    # Load Priors
    from prior import L2Prior, DPoser_pose_exp, DPoser_beta
    if args.prior == 'DPoser':
        DPoser_pose_exp_model = DPoser_pose_exp(args.exp_config_path, args)
        DPoser_beta_model = DPoser_beta(args.beta_config_path, args)
        prior_dict = {'shape': DPoser_beta_model, 'pose_expression': DPoser_pose_exp_model}
    elif args.prior == 'L2':
        prior_dict = {'shape': L2Prior(), 'expression': L2Prior(), 'pose': L2Prior()}
    else:
        prior_dict = {}
    fitting.load_prior(prior_dict)

    # --- 4. Run the Fitting Process ---
    print("\nStarting the fitting process...")
    init_results = None
    fitting.run(imgs, all_kpts, base_paths, seg_net, init_results)
    print("Fitting process completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run photometric fitting on an image.")
    parser.add_argument('-i', '--data_dir', default='./demo_inputs', type=str)

    parser.add_argument('--batchsize', type=int, default=2, help='batch size for fitting')

    parser.add_argument("--output_folder", type=str, default='demo_output')
    parser.add_argument("--verbose", type=str, default=False)
    parser.add_argument("--save-static", type=bool, default=True)
    parser.add_argument("--save-video", type=bool, default=False)

    parser.add_argument("--max_vis_num", type=int, default=5, help="Maximum number of images to visualize.")
    parser.add_argument("--device", type=str, default="cuda:7", )

    # params for DPoser
    parser.add_argument('--prior', type=str, default='DPoser', choices=['DPoser', 'L2'],
                        help='Our prior model or competitors')
    parser.add_argument('--exp-ckpt-path', type=str,
                        default='/data4/ljz24/projects/3DHuman/DPoser/checkpoints/dposer/wcpapre/BaseMLP/last.ckpt')
    parser.add_argument('--beta-ckpt-path', type=str,
                        default='/data4/ljz24/projects/3DHuman/DPoser/checkpoints/dposer/mica/TinyMLP2/last.ckpt')
    parser.add_argument('--exp-config-path', type=str,
                        default='configs.face.subvp.pose_timefc.get_config')
    parser.add_argument('--beta-config-path', type=str,
                        default='configs.face.subvp.shape_timefc_tiny.get_config')

    parser.add_argument('--data-root', type=str, default='/data4/ljz24/projects/3DHuman/data/human/Facedataset/MICA_FLAME')

    torch.manual_seed(42)
    args = parser.parse_args()
    main_demo(args)

"""
python photometric_fitting.py --prior DPoser --device cuda:0 --save-video True
"""
