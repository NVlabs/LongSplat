# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
import torch
from utils.camera_utils import camera_to_JSON
import json
from scipy.interpolate import UnivariateSpline
import cv2
from scipy.optimize import least_squares
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import warping
from utils.graphics_utils import unporject
from utils.mast3r_utils import Mast3rMatcher
from gaussian_renderer import render

def save_transforms(cameras, path):
    json_cams = []
    viewpoint_stack = cameras
    for id, view in enumerate(viewpoint_stack):
        json_cams.append(camera_to_JSON(None, view))

    with open(path, 'w') as file:
        json.dump(json_cams, file, indent=2)

def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm

def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )

def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V

def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R.t()
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3].t()
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def filter1d(vec, time, W):
    stepsize = 2 * W + 1
    filtered = np.median(strided_app(vec, stepsize, stepsize), axis=-1)
    pre_smoothed = np.interp(time, time[W:-W:stepsize], filtered)
    return pre_smoothed

def smooth_vec(vec, time, s, median_prefilter):
    if median_prefilter:
        vec = np.stack([
            filter1d(vec[..., 0], time, 5),
            filter1d(vec[..., 1], time, 5),
            filter1d(vec[..., 2], time, 5)
        ], axis=-1)
    smoothed = np.zeros_like(vec)
    for i in range(vec.shape[1]):
        spl = UnivariateSpline(time, vec[..., i])
        spl.set_smoothing_factor(s)
        smoothed[..., i] = spl(time)
    return smoothed

def smooth_poses_spline(poses, st=0.5, sr=4, median_prefilter=True):
    poses = np.asarray(poses)
    assert poses.shape[1:] == (4, 4), "Input must be (N, 4, 4) pose matrices."
    if len(poses) < 30:
        median_prefilter = False
    # Extract 3x4 for smoothing
    poses_3x4 = poses[:, :3, :4].copy()
    # For compatibility with old code, flip x axis before smoothing
    poses_3x4[:, 0] = -poses_3x4[:, 0]
    posesnp = poses_3x4
    scale = 2e-2 / np.median(np.linalg.norm(posesnp[1:, :3, 3] - posesnp[:-1, :3, 3], axis=-1))
    posesnp[:, :3, 3] *= scale
    time = np.linspace(0, 1, len(posesnp)) 
    t = smooth_vec(posesnp[..., 3], time, st, median_prefilter)
    z = smooth_vec(posesnp[..., 2], time, sr, median_prefilter)
    z /= np.linalg.norm(z, axis=-1)[:, None]
    y_ = smooth_vec(posesnp[..., 1], time, sr, median_prefilter)
    x = np.cross(z, y_)
    x /= np.linalg.norm(x, axis=-1)[:, None]
    y = np.cross(x, z)
    smooth_posesnp = np.stack([x, y, z, t], -1)
    poses_3x4[:, 0] = -poses_3x4[:, 0]
    smooth_posesnp[:, 0] = -smooth_posesnp[:, 0]
    smooth_posesnp[:, :3, 3] /= scale
    N = poses.shape[0]
    out = np.zeros((N, 4, 4), dtype=poses.dtype)
    out[:, :3, :4] = smooth_posesnp
    out[:, 3, 3] = 1.0
    return out

def reprojection_error(params, points_3d, points_2d, K):
    # Extract rotation and translation from params
    rvec = params[:3]
    tvec = params[3:]
    # Project 3D points to 2D using current R and t
    projected_points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
    projected_points_2d = projected_points_2d.squeeze()

    # Compute the residual (difference between observed and projected points)
    residuals = points_2d - projected_points_2d # N*1
    return residuals.flatten()

def visual_localization(viewpoint, ref_viewpoint, gaussians, pipeline, background, matcher):    
    # --- Part 1: Initialize Pose using Mast3r and PnP RANSAC ---
    
    # Get reference camera depth by rendering
    ref_render_pkg = render(ref_viewpoint, gaussians, pipeline, background, retain_grad=False)
    ref_voxel_visible_mask = ref_render_pkg["visible_mask"]
    ref_rendered_depth = ref_render_pkg["depth"][0]
    
    # Get intrinsic parameters
    intrinsic_np = viewpoint.intrinsic.detach().cpu().numpy()
    
    # Use mast3r for feature matching between reference and test images
    viewpoint.kp0, viewpoint.kp1, _, _, _, _, _, _, viewpoint.pre_depth_map, viewpoint.depth_map = matcher._forward(
        ref_viewpoint.original_image, viewpoint.original_image, intrinsic_np)
    
    # Set confidence and move to GPU
    viewpoint.conf = torch.ones(viewpoint.kp0.shape[0], device=viewpoint.kp0.device)
    viewpoint.kp0 = viewpoint.kp0.cuda()
    viewpoint.kp1 = viewpoint.kp1.cuda()
    viewpoint.depth_map = viewpoint.depth_map.cuda()
    viewpoint.pre_depth_map = viewpoint.pre_depth_map.cuda()
    
    # Store reference viewpoint information for correspondence loss calculation
    viewpoint.ref_viewpoint = ref_viewpoint
    
    # Unproject 3D points from reference depth using matched keypoints
    pre_pts = unporject(ref_rendered_depth, ref_viewpoint.view_world_transform, ref_viewpoint.intrinsic, viewpoint.kp0)
    
    # Convert keypoints to pixel coordinates
    kp1 = viewpoint.kp1 / 2 + .5
    kp1[:, 0] *= viewpoint.original_image.shape[2]
    kp1[:, 1] *= viewpoint.original_image.shape[1]
    pre_pts_np = pre_pts.detach().cpu().numpy()
    kp1_np = kp1.detach().cpu().numpy()
    
    # Solve PnP RANSAC with increased iterations for better accuracy
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        pre_pts_np, kp1_np, intrinsic_np, None, 
        iterationsCount=2000, reprojectionError=1.5, confidence=0.99, 
        flags=cv2.SOLVEPNP_ITERATIVE)
    
    # Refine pose using least squares optimization with tighter tolerance
    pre_pts_inliers = pre_pts_np[inliers].reshape(-1, 3)
    kp1_inliers = kp1_np[inliers].reshape(-1, 2)
    
    initial_params = np.hstack((rotation_vector.flatten(), translation_vector.flatten()))
    # First pass: coarse optimization
    result = least_squares(reprojection_error, initial_params, 
                        args=(pre_pts_inliers, kp1_inliers, intrinsic_np),
                        verbose=0, ftol=1e-6, xtol=1e-6, max_nfev=200)
    # Second pass: fine optimization from first result
    result = least_squares(reprojection_error, result.x, 
                        args=(pre_pts_inliers, kp1_inliers, intrinsic_np),
                        verbose=0, ftol=1e-8, xtol=1e-8)
    
    rotation_vector = result.x[:3]
    translation_vector = result.x[3:]
    viewpoint.is_registered = True
    
    # Convert rotation vector to rotation matrix and update viewpoint
    rotation_matrix, _ = cv2.Rodrigues(-rotation_vector)
    translation_vector = translation_vector.reshape(3)
    rotation_matrix = torch.from_numpy(rotation_matrix).float().cuda()
    translation_vector = torch.from_numpy(translation_vector).float().cuda()
    
    viewpoint.update_RT(rotation_matrix, translation_vector)

    # --- Part 2: Pose Estimation Test (Refinement) ---
    
    pose_iteration = 800  # Increased iterations for better convergence

    # Use smaller learning rate for rotation (more sensitive) and two-stage optimization
    pose_optimizer = torch.optim.Adam([
        {"params": [viewpoint.cam_trans_delta], "lr": 0.01}, 
        {"params": [viewpoint.cam_rot_delta], "lr": 0.005}  # Smaller LR for rotation
    ])
    # Use step decay for better fine-tuning
    scheduler = torch.optim.lr_scheduler.MultiStepLR(pose_optimizer, milestones=[400, 600], gamma=0.5)
    gt_image = viewpoint.original_image.cuda()

    for iteration in range(pose_iteration):
        render_pkg = render(viewpoint, gaussians, pipeline, background, retain_grad=True)
        voxel_visible_mask = render_pkg["visible_mask"]
        image = render_pkg["render"]
        rendered_depth = render_pkg["depth"][0]
        
        # L1 loss
        Ll1 = l1_loss(image, gt_image)
        
        # SSIM loss
        Lssim = 1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        

        loss = Ll1 * 0.8 + 0.2 * Lssim
        
        loss.backward()

        with torch.no_grad():
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([viewpoint.cam_trans_delta, viewpoint.cam_rot_delta], max_norm=1.0)
            
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_pose(viewpoint)