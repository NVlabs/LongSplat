# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, depth_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.pose_utils import save_transforms, update_pose
from utils.graphics_utils import get_occlusion_mask, unporject, warping, compute_scale
from utils.mast3r_utils import Mast3rMatcher
import cv2
from scipy.optimize import least_squares
from torch.optim.lr_scheduler import ExponentialLR

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
    print("fused ssim available")
except:
    FUSED_SSIM_AVAILABLE = False
    print("fused ssim not available")
    
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


def training(dataset, opt, pipe, dataset_name, debug_from, logger=None):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians)
    num_views = len(scene.getTrainCameras())
    start_view_id = 0
    end_view_id = 1

    init_iteraion = opt.init_iteraion
    pose_iteration = opt.pose_iteration
    local_iter = opt.local_iter
    global_iter = opt.global_iter
    post_iter = opt.post_iter
    matcher = Mast3rMatcher()

    ## initialize ##
    end_view_id = scene.init_frame_num
    opt.iterations = init_iteraion
    opt.update_until = init_iteraion
    gaussians.training_setup(opt)
    gaussians.training_pose_setup(scene.getTrainCameras()[0:end_view_id], opt)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Init Optimization")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras()[0:end_view_id]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, retain_grad=True)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        rendered_depth = render_pkg["depth"][0]
        voxel_visible_mask = render_pkg["visible_mask"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        if FUSED_SSIM_AVAILABLE:
            ssim_loss = (1 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
        else:
            ssim_loss = (1 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        # Depth loss
        if opt.depth_loss_weight > 0:
            midas_depth = viewpoint_cam.depth_map.detach().cuda()
            Ldepth = depth_loss(midas_depth, rendered_depth)
            loss += Ldepth * opt.depth_loss_weight

        # 2D correspondence loss
        if opt.loss_2d_correspondence_weight > 0 and viewpoint_cam.uid > 0:
            view1 = scene.getTrainCameras()[viewpoint_cam.uid - 1]
            view2 = viewpoint_cam
            kp0, kp1, conf = view2.kp0.cuda(), view2.kp1.cuda(), view2.conf.cuda()
            xy0 = kp0 / 2 + .5
            xy1 = warping(rendered_depth, view2.view_world_transform, view1.world_view_transform.detach(), view2.intrinsic, kp1)
            xy1 = xy1 / 2 + .5
            mask = torch.logical_and(xy1 > 0., xy1 < 1.).all(dim=-1)
            xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
            loss_2d = ((xy0.detach() - xy1).abs() * conf[:, None]).mean()
            loss += loss_2d * opt.loss_2d_correspondence_weight
        
        if opt.loss_2d_correspondence_weight > 0 and viewpoint_cam.uid < end_view_id - 2:
            view1 = viewpoint_cam
            view2 = scene.getTrainCameras()[viewpoint_cam.uid + 1]
            kp0, kp1, conf = view2.kp0.cuda(), view2.kp1.cuda(), view2.conf.cuda()
            xy0 = kp0 / 2 + .5
            xy1 = warping(rendered_depth, view2.view_world_transform.detach(), view1.world_view_transform, view2.intrinsic, kp1)
            xy1 = xy1 / 2 + .5
            mask = torch.logical_and(xy1 > 0., xy1 < 1.).all(dim=-1)
            xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
            loss_2d_2 = ((xy0.detach() - xy1).abs() * conf[:, None]).mean()
            loss += loss_2d_2 * opt.loss_2d_correspondence_weight

        loss.backward()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, require_purning = True)
            if iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.pose_optimizer.step()
                gaussians.pose_optimizer.zero_grad(set_to_none = True)
                update_pose(viewpoint_cam)

    end_view_id += 1

    # PnP retry limit to avoid infinite loop
    pnp_retry_count = {}
    max_pnp_retries = 10

    while start_view_id < num_views:
        ## pose estimation ##
        opt.iterations = local_iter
        opt.update_until = local_iter
        gaussians.training_setup(opt)
        gaussians.training_pose_setup(scene.getTrainCameras()[start_view_id:end_view_id] ,opt)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        
        if scene.getTrainCameras()[end_view_id-1].is_registered == False:
            with torch.no_grad():
                pre_viewpoint_cam1 = scene.getTrainCameras()[end_view_id-2]
                viewpoint_cam = scene.getTrainCameras()[end_view_id-1]

                pre_render_pkg = render(pre_viewpoint_cam1, gaussians, pipe, background, retain_grad=False)
                pre_rendered_depth = pre_render_pkg["depth"][0]

                intrinsic_np = viewpoint_cam.intrinsic.detach().cpu().numpy()
                viewpoint_cam.kp0, viewpoint_cam.kp1, _, _, _, _, _, _, viewpoint_cam.pre_depth_map, viewpoint_cam.depth_map = matcher._forward(pre_viewpoint_cam1.original_image, viewpoint_cam.original_image, intrinsic_np)
                viewpoint_cam.conf = torch.ones(viewpoint_cam.kp0.shape[0], device=viewpoint_cam.kp0.device)
                viewpoint_cam.kp0 = viewpoint_cam.kp0.cuda()
                viewpoint_cam.kp1 = viewpoint_cam.kp1.cuda()
                viewpoint_cam.depth_map = viewpoint_cam.depth_map.cuda()
                viewpoint_cam.pre_depth_map = viewpoint_cam.pre_depth_map.cuda()

                pre_pts = unporject(pre_rendered_depth, pre_viewpoint_cam1.view_world_transform, pre_viewpoint_cam1.intrinsic, viewpoint_cam.kp0)
                
                kp1 = viewpoint_cam.kp1 / 2 + .5
                kp1[:, 0] *= viewpoint_cam.original_image.shape[2]
                kp1[:, 1] *= viewpoint_cam.original_image.shape[1]
                pre_pts_np = pre_pts.detach().cpu().numpy()
                kp1_np = kp1.detach().cpu().numpy()

                success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(pre_pts_np, kp1_np, intrinsic_np, None, iterationsCount=200,reprojectionError=5.0,confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)

                if success and len(inliers) >= 4:
                    pre_pts_inliers = pre_pts_np[inliers].reshape(-1, 3)
                    kp1_inliers = kp1_np[inliers].reshape(-1, 2)

                    initial_params = np.hstack((rotation_vector.flatten(), translation_vector.flatten()))
                    result = least_squares(reprojection_error, initial_params, 
                                        args=(pre_pts_inliers, kp1_inliers, intrinsic_np),
                                        verbose=0,  # Verbose mode to print loss
                                        ftol=1e-8, xtol=1e-8)
                    rotation_vector = result.x[:3]
                    translation_vector = result.x[3:]
                    viewpoint_cam.is_registered = True
                else:
                    print("Failed to solve PnP")
                    viewpoint_cam.is_registered = False
                    
                    frame_id = end_view_id - 1
                    pnp_retry_count[frame_id] = pnp_retry_count.get(frame_id, 0) + 1
                    
                    if pnp_retry_count[frame_id] >= max_pnp_retries:
                        print(f"Frame {frame_id} failed PnP after {max_pnp_retries} retries, skipping...")
                        viewpoint_cam.is_registered = True
                    else:
                        end_view_id -= 1

                rotation_matrix, _ = cv2.Rodrigues(-rotation_vector)
                translation_vector = translation_vector.reshape(3)
                rotation_matrix = torch.from_numpy(rotation_matrix).float().cuda()
                translation_vector = torch.from_numpy(translation_vector).float().cuda()
                
                viewpoint_cam.update_RT(rotation_matrix, translation_vector)

                scale, offset = compute_scale(pre_rendered_depth, viewpoint_cam.pre_depth_map, viewpoint_cam.kp0)
                viewpoint_cam.depth_map = viewpoint_cam.depth_map * scale + offset
                
            pose_optimizer = torch.optim.Adam([{"params": [viewpoint_cam.cam_trans_delta], "lr": opt.translation_lr_init}, {"params": [viewpoint_cam.cam_rot_delta], "lr": opt.rotation_lr_init}])
            gt_image = viewpoint_cam.original_image.cuda()
            
            progress_bar = tqdm(range(0, pose_iteration), desc="Pose estiamtion progress")
            for iteration in range(pose_iteration):
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, retain_grad=True)
                image = render_pkg["render"]
                rendered_depth = render_pkg["depth"][0]
                voxel_visible_mask = render_pkg["visible_mask"]
                occ_mask = get_occlusion_mask(viewpoint_cam=pre_viewpoint_cam1, viewpoint_cam2=viewpoint_cam, depth=pre_rendered_depth, device=pre_rendered_depth.device, thresh=0.001).detach()

                Ll1 = l1_loss(image[:,occ_mask], gt_image[:,occ_mask])
                loss = Ll1

                # 2D correspondence loss
                if opt.loss_2d_correspondence_weight > 0 and viewpoint_cam.uid > 0:
                    view1 = scene.getTrainCameras()[viewpoint_cam.uid - 1]
                    view2 = viewpoint_cam
                    kp0, kp1, conf = view2.kp0.cuda(), view2.kp1.cuda(), view2.conf.cuda()
                    xy0 = kp0 / 2 + .5
                    xy1 = warping(rendered_depth, view2.view_world_transform, view1.world_view_transform.detach(), view2.intrinsic, kp1)
                    xy1 = xy1 / 2 + .5
                    mask = torch.logical_and(xy1 > 0., xy1 < 1.).all(dim=-1)
                    xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
                    loss_2d = ((xy0.detach() - xy1).abs() * conf[:, None]).mean()
                    loss += loss_2d * opt.loss_2d_correspondence_weight

                loss.backward()

                with torch.no_grad():
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    gaussians.pose_optimizer.zero_grad(set_to_none=True)
                    update_pose(viewpoint_cam)

                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                        progress_bar.update(10)
            
            progress_bar.close()

            with torch.no_grad():
                occ_mask = get_occlusion_mask(viewpoint_cam=pre_viewpoint_cam1, viewpoint_cam2=viewpoint_cam, depth=pre_rendered_depth, device=pre_rendered_depth.device, thresh=0.001).detach()
                densify_mask = occ_mask.view(-1) == 0
                if densify_mask.sum() > 0:
                    depth_ref =viewpoint_cam.depth_map.cuda()
                    gaussians.densify_occlusion(viewpoint_cam, depth_ref, densify_mask)
        
        ## local optimization ##
        first_iter = 0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Local Optimization " + str(end_view_id) + "/" + str(num_views) + "(w=" + str(end_view_id-start_view_id) + ")")
        first_iter += 1
        for iteration in range(first_iter, opt.iterations + 1):

            gaussians.update_learning_rate(iteration)

            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras()[start_view_id:end_view_id]
            
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, retain_grad=retain_grad)
            voxel_visible_mask = render_pkg["visible_mask"]
            
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
            rendered_depth = render_pkg["depth"][0]

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)

            if FUSED_SSIM_AVAILABLE:
                ssim_loss = (1 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
            else:
                ssim_loss = (1 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

            # Depth loss
            if opt.depth_loss_weight > 0:
                midas_depth = viewpoint_cam.depth_map.detach().cuda()
                Ldepth = depth_loss(midas_depth, rendered_depth)
                loss += Ldepth * opt.depth_loss_weight

            # 2D correspondence loss
            if opt.loss_2d_correspondence_weight > 0 and viewpoint_cam.uid > 0:
                view1 = scene.getTrainCameras()[viewpoint_cam.uid - 1]
                view2 = viewpoint_cam
                kp0, kp1, conf = view2.kp0.cuda(), view2.kp1.cuda(), view2.conf.cuda()
                xy0 = kp0 / 2 + .5
                xy1 = warping(rendered_depth, view2.view_world_transform, view1.world_view_transform.detach(), view2.intrinsic, kp1)
                xy1 = xy1 / 2 + .5
                mask = (xy1 > 0.) & (xy1 < 1.)
                mask = mask.all(dim=-1)
                xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
                loss_2d = ((xy0.detach() - xy1).abs() * conf[:, None]).mean()
                loss += loss_2d * opt.loss_2d_correspondence_weight
            
            if opt.loss_2d_correspondence_weight > 0 and viewpoint_cam.uid < end_view_id - 2:
                view1 = viewpoint_cam
                view2 = scene.getTrainCameras()[viewpoint_cam.uid + 1]
                kp0, kp1, conf = view2.kp0.cuda(), view2.kp1.cuda(), view2.conf.cuda()
                xy0 = kp0 / 2 + .5
                xy1 = warping(rendered_depth, view2.view_world_transform.detach(), view1.world_view_transform, view2.intrinsic, kp1)
                xy1 = xy1 / 2 + .5
                mask = (xy1 > 0.) & (xy1 < 1.)
                mask = mask.all(dim=-1)
                xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
                loss_2d_2 = ((xy0.detach() - xy1).abs() * conf[:, None]).mean()
                loss += loss_2d_2 * opt.loss_2d_correspondence_weight

            loss.backward()

            with torch.no_grad():
                # Progress bar
                if iteration % 10 == 0:
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # densification
                if iteration < opt.update_until and iteration > opt.start_stat:
                    # add statis
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, require_purning = False)
                if iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()
                    
                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    gaussians.pose_optimizer.step()
                    gaussians.pose_optimizer.zero_grad(set_to_none = True)
                    update_pose(viewpoint_cam)
        
        opt.iterations = global_iter
        opt.update_until = global_iter
        gaussians.training_setup(opt)
        gaussians.training_pose_setup(scene.getTrainCameras()[0:end_view_id] ,opt)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        first_iter = 0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Global Optimization " + str(end_view_id) + "/" + str(num_views))
        first_iter += 1
        
        for iteration in range(first_iter, opt.iterations + 1):

            gaussians.update_learning_rate(iteration)

            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras()[0:end_view_id]
            
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, retain_grad=retain_grad)
            voxel_visible_mask = render_pkg["visible_mask"]
            
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
            rendered_depth = render_pkg["depth"][0]
            
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)

            if FUSED_SSIM_AVAILABLE:
                ssim_loss = (1 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
            else:
                ssim_loss = (1 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

            # Depth loss
            if opt.depth_loss_weight > 0:
                midas_depth = viewpoint_cam.depth_map.detach().cuda()
                Ldepth = depth_loss(midas_depth, rendered_depth)
                loss += Ldepth * opt.depth_loss_weight

            # # 2D correspondence loss
            if opt.loss_2d_correspondence_weight > 0 and viewpoint_cam.uid > 0:
                view1 = scene.getTrainCameras()[viewpoint_cam.uid - 1]
                view2 = viewpoint_cam
                kp0, kp1, conf = view2.kp0.cuda(), view2.kp1.cuda(), view2.conf.cuda()
                xy0 = kp0 / 2 + .5
                xy1 = warping(rendered_depth, view2.view_world_transform, view1.world_view_transform.detach(), view2.intrinsic, kp1)
                xy1 = xy1 / 2 + .5
                mask = (xy1 > 0.) & (xy1 < 1.)
                mask = mask.all(dim=-1)
                xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
                loss_2d = ((xy0.detach() - xy1).abs() * conf[:, None]).mean()
                loss += loss_2d * opt.loss_2d_correspondence_weight
            
            if opt.loss_2d_correspondence_weight > 0 and viewpoint_cam.uid < end_view_id - 2:
                view1 = viewpoint_cam
                view2 = scene.getTrainCameras()[viewpoint_cam.uid + 1]
                kp0, kp1, conf = view2.kp0.cuda(), view2.kp1.cuda(), view2.conf.cuda()
                xy0 = kp0 / 2 + .5
                xy1 = warping(rendered_depth, view2.view_world_transform.detach(), view1.world_view_transform, view2.intrinsic, kp1)
                xy1 = xy1 / 2 + .5
                mask = (xy1 > 0.) & (xy1 < 1.)
                mask = mask.all(dim=-1)
                xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
                loss_2d_2 = ((xy0.detach() - xy1).abs() * conf[:, None]).mean()
                loss += loss_2d_2 * opt.loss_2d_correspondence_weight
            
            loss.backward()

            with torch.no_grad():
                # Progress bar
                if iteration % 10 == 0:
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()  
                
                # densification
                if iteration < opt.update_until and iteration > opt.start_stat:
                    # add statis
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                if iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()
                    
                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    gaussians.pose_optimizer.step()
                    gaussians.pose_optimizer.zero_grad(set_to_none = True)
                    update_pose(viewpoint_cam)

        with torch.no_grad():
            if end_view_id < num_views:
                end_viewpoint_cam = scene.getTrainCameras()[end_view_id - 1]
                render_pkg_end = render(end_viewpoint_cam, gaussians, pipe, background)
                end_visible_mask = render_pkg_end["visible_mask"]
                offset_selection_mask = render_pkg_end["selection_mask"]
                end_n_touched = (render_pkg_end["n_touched"] > 0)
                
                mask_size = gaussians.get_anchor.shape[0] * gaussians.n_offsets
                end_mask = torch.empty(mask_size, dtype=torch.bool, device="cuda")
                end_mask.zero_()
                
                end_visible_mask_expand = end_visible_mask.unsqueeze(0).expand(gaussians.n_offsets, -1).reshape(-1)
                visible_indices = torch.nonzero(end_visible_mask_expand).squeeze(1)
                final_indices = visible_indices[offset_selection_mask]
                end_mask[final_indices] = end_n_touched
                start_mask = torch.empty(mask_size, dtype=torch.bool, device="cuda")
                
                for start_viewpoint_cam in scene.getTrainCameras()[start_view_id:end_view_id - 1]:
                    start_mask.zero_()
                    render_pkg_start = render(start_viewpoint_cam, gaussians, pipe, background)
                    start_visible_mask = render_pkg_start["visible_mask"]
                    offset_selection_mask = render_pkg_start["selection_mask"]
                    start_n_touched = (render_pkg_start["n_touched"] > 0)
                    
                    start_visible_mask_expand = start_visible_mask.unsqueeze(0).expand(gaussians.n_offsets, -1).reshape(-1)
                    visible_indices = torch.nonzero(start_visible_mask_expand).squeeze(1)
                    final_indices = visible_indices[offset_selection_mask]
                    start_mask[final_indices] = start_n_touched

                    # Calculate the overlap ratio
                    count_start = start_mask.count_nonzero()
                    count_end = end_mask.count_nonzero()
                    denom = min(count_start, count_end)

                    if denom == 0:
                        visibility_ratio = 0.0
                    else:
                        intersection = torch.logical_and(start_mask, end_mask).count_nonzero()
                        visibility_ratio = intersection / denom
                    
                    if visibility_ratio < 0.2 and end_view_id - start_view_id > 5:
                        start_view_id += 1
                    else:
                        break
                end_view_id += 1
            else:
                start_view_id += 1


    
    if opt.pruning_ratio > 0:
        with torch.no_grad():
            anchor_touched_list = torch.zeros(gaussians.get_anchor.shape[0] * gaussians.n_offsets, device="cuda")
            for view in scene.getTrainCameras()[0:end_view_id]:
                render_pkg = render(view, gaussians, pipe, background, retain_grad=False)
                n_touched = render_pkg["n_touched"]
                offset_selection_mask = render_pkg["selection_mask"]
                voxel_visible_mask = render_pkg["visible_mask"]
                visible_mask_expand = voxel_visible_mask.unsqueeze(0).expand(gaussians.n_offsets, -1).reshape(-1)
                visible_indices = torch.nonzero(visible_mask_expand).squeeze(1)
                final_indices = visible_indices[offset_selection_mask]
                anchor_touched_list[final_indices] += n_touched

            anchor_touched = anchor_touched_list.reshape(gaussians.get_anchor.shape[0], gaussians.n_offsets).sum(dim=1)
                
            purning_mask = torch.zeros_like(anchor_touched, dtype=torch.bool)
            _, indices = torch.sort(anchor_touched)
            purning_mask[indices[:int(len(anchor_touched)*opt.pruning_ratio)]] = True
            gaussians.prune_anchor(purning_mask)
    
    ## refinement ##
    opt.iterations = 30000 + post_iter
    opt.update_until = 30000 + post_iter // 2
    gaussians.training_setup(opt)
    gaussians.training_pose_setup(scene.getTrainCameras(), opt)
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)
    gaussians.pose_scheduler = ExponentialLR(gaussians.pose_optimizer, gamma=0.95)

    # change to final resolution
    for view in scene.getAllCameras():
        view.to_final()
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    first_iter = 30000
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Refinement")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        gaussians.update_learning_rate(iteration)

        if iteration % 400 == 0:
            gaussians.scheduler.step()
            gaussians.pose_scheduler.step()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras()[0:end_view_id]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, retain_grad=retain_grad)
        voxel_visible_mask = render_pkg["visible_mask"]
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        rendered_depth = render_pkg["depth"][0]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        if FUSED_SSIM_AVAILABLE:
            ssim_loss = (1 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
        else:
            ssim_loss = (1 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        # Depth loss
        if opt.depth_loss_weight > 0:
            midas_depth = viewpoint_cam.depth_map.detach().cuda()
            Ldepth = depth_loss(midas_depth, rendered_depth)
            loss += Ldepth * opt.depth_loss_weight

        # # 2D correspondence loss
        if opt.loss_2d_correspondence_weight > 0 and viewpoint_cam.uid > 0:
            view1 = scene.getTrainCameras()[viewpoint_cam.uid - 1]
            view2 = viewpoint_cam
            kp0, kp1, conf = view2.kp0.cuda(), view2.kp1.cuda(), view2.conf.cuda()
            xy0 = kp0 / 2 + .5
            xy1 = warping(rendered_depth, view2.view_world_transform, view1.world_view_transform.detach(), view2.intrinsic, kp1)
            xy1 = xy1 / 2 + .5
            mask = torch.logical_and(xy1 > 0., xy1 < 1.).all(dim=-1)
            xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
            loss_2d = ((xy0.detach() - xy1).abs() * conf[:, None]).mean()
            loss += loss_2d * opt.loss_2d_correspondence_weight
        
        if opt.loss_2d_correspondence_weight > 0 and viewpoint_cam.uid < end_view_id - 2:
            view1 = viewpoint_cam
            view2 = scene.getTrainCameras()[viewpoint_cam.uid + 1]
            kp0, kp1, conf = view2.kp0.cuda(), view2.kp1.cuda(), view2.conf.cuda()
            xy0 = kp0 / 2 + .5
            xy1 = warping(rendered_depth, view2.view_world_transform.detach(), view1.world_view_transform, view2.intrinsic, kp1)
            xy1 = xy1 / 2 + .5
            mask = torch.logical_and(xy1 > 0., xy1 < 1.).all(dim=-1)
            xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
            loss_2d_2 = ((xy0.detach() - xy1).abs() * conf[:, None]).mean()
            loss += loss_2d_2 * opt.loss_2d_correspondence_weight

        loss.backward()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            if iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.pose_optimizer.step()
                gaussians.pose_optimizer.zero_grad(set_to_none = True)
                update_pose(viewpoint_cam)


    logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
    scene.save(iteration)
    save_transforms(scene.getTrainCameras().copy(), os.path.join(scene.model_path, "cameras_all_train.json"))
    save_transforms(scene.getTestCameras().copy(), os.path.join(scene.model_path, "cameras_all_test.json"))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, Ll1, loss, l1_loss, scene : Scene, renderFunc, renderArgs, end_view_id, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), end_view_id)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), end_view_id)
    
    # Report test and samples of training set
    scene.gaussians.eval()
    torch.cuda.empty_cache()
    validation_configs = [{'name': 'train', 'cameras' : scene.getTrainCameras()[0:end_view_id]}]

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0

            for idx, viewpoint in enumerate(config['cameras']):
                render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                voxel_visible_mask = render_pkg["visible_mask"]
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if tb_writer and (idx < 30):
                    tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=end_view_id)
                    tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=end_view_id)
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            
            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])          
            logger.info("\n[{} th] Evaluating {}: L1 {} PSNR {}".format(end_view_id, config['name'], l1_test, psnr_test))

            
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, end_view_id)
                tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, end_view_id)

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], end_view_id)
        torch.cuda.empty_cache()

        scene.gaussians.train()
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')
        
    dataset = args.source_path.split('/')[-1]
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset, args.debug_from, logger)
    
    # All done
    logger.info("\nTraining complete.")