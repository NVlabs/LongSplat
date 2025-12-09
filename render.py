# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.visualize_utils import vis_depth, vis_pose, eval_pose_metrics
from utils.pose_utils import smooth_poses_spline, save_transforms, vis_loc
from scene.cameras import Camera
import cv2
import imageio
from utils.colmap_utils import save_imagestxt, save_cameras

def render_nvs(model_path, name, iteration, views, gaussians, pipeline, background):
    nvs_path = os.path.join(model_path, name, "ours_{}".format(iteration), "nvs")
    videos_path = os.path.join(model_path, name, "ours_{}".format(iteration), "videos")
    
    if not os.path.exists(nvs_path):
        os.makedirs(nvs_path)
    if not os.path.exists(videos_path):
        os.makedirs(videos_path)

    poses_list = []
    for view in views:
        poses_list.append(view.view_world_transform.transpose(0, 1).detach().cpu().numpy())
    poses_list = np.array(poses_list)
    nvs_num = len(poses_list)

    poses_list = np.array(poses_list)
    nvs_pose_list = smooth_poses_spline(poses_list)
    nvs_pose_list = torch.from_numpy(nvs_pose_list).cuda()
    nvs_pose_list = nvs_pose_list.inverse()
    FoVx = views[0].FoVx
    FoVy = views[0].FoVy
    nvs_image_list = []
    nvs_depth_list = []
    gt_list = []
    nvs_views = []
    name_list = []
    for i in tqdm(range(nvs_num), desc="Rendering NVS progress"):
        nvs_view = Camera(colmap_id=i, R=None, T=None, R_gt=None, T_gt=None, FoVx=FoVx, FoVy=FoVy, 
                         image=views[0].original_image, gt_alpha_mask=None, image_name=None, uid=None)
        nvs_view.update_RT(nvs_pose_list[i, :3, :3].transpose(0, 1), nvs_pose_list[i, :3, 3])
        nvs_view.to_final()
        rendering = render(nvs_view, gaussians, pipeline, background, retain_grad=False)
        voxel_visible_mask = rendering["visible_mask"]
        torchvision.utils.save_image(rendering["render"], os.path.join(nvs_path, '{0:05d}'.format(i) + ".png"))
        render_img = torch.clamp(rendering["render"], min=0., max=1.)
        render_img = (render_img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        gt = nvs_view.original_image[0:3, :, :]
        gt = (gt.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        gt_list.append(gt)
        depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)
        nvs_depth_list.append(depth_map)
        render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
        nvs_image_list.append(render_img)
        nvs_views.append(nvs_view)
        name_list.append('{0:05d}'.format(i))
    imageio.mimwrite(os.path.join(videos_path, 'gt.mp4'), np.stack(gt_list), fps=30, quality=6, output_params=["-f", "mp4"])
    imageio.mimwrite(os.path.join(videos_path, 'nvs_rgb.mp4'), np.stack(nvs_image_list), fps=30, quality=6, output_params=["-f", "mp4"])
    imageio.mimwrite(os.path.join(videos_path, 'nvs_depth.mp4'), np.stack(nvs_depth_list), fps=30, quality=6, output_params=["-f", "mp4"])

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    poses_path = os.path.join(model_path, name, "ours_{}".format(iteration), "poses")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    videos_path = os.path.join(model_path, name, "ours_{}".format(iteration), "videos")
    nvs_path = os.path.join(model_path, name, "ours_{}".format(iteration), "nvs")

    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)
    if not os.path.exists(poses_path):
        os.makedirs(poses_path)
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    if not os.path.exists(videos_path):
        os.makedirs(videos_path)
    if not os.path.exists(nvs_path):
        os.makedirs(nvs_path)

    name_list = []
    per_view_dict = {}
    t_list = []
    poses_list = []
    pose_imgs_list = []
    render_imgs_list = []
    render_depth_list = []
    gt_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize(); t0 = time.time()
        rendering = render(view, gaussians, pipeline, background)
        voxel_visible_mask = rendering["visible_mask"]
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        poses_list.append(view.view_world_transform.transpose(0, 1).detach().cpu().numpy())
        gt = view.original_image[0:3, :, :]
        name_list.append('{0:05d}'.format(idx))

        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
        np.save(os.path.join(depth_path, view.image_name + '.npy'), rendering['depth'][0].detach().cpu().numpy())
        cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), depth_map)

        if view.T_gt is not None and idx > 1:
            pose_img = vis_pose(views[0:idx+1])
            pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
            pose_imgs_list.append(pose_img)
        
        render_img = torch.clamp(rendering["render"], min=0., max=1.)
        render_img = (render_img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        gt = (gt.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        gt_list.append(gt)
        render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
        render_imgs_list.append(render_img)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)
        render_depth_list.append(depth_map)
    
    # Only evaluate pose metrics if GT poses are available
    has_gt_poses = any(view.T_gt is not None for view in views)
    if has_gt_poses:
        eval_pose_metrics(views, poses_path)
    else:
        print("No GT poses available, skipping pose metrics evaluation")

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    if len(render_imgs_list) > 0:
        if len(pose_imgs_list) > 0:
            imageio.mimwrite(os.path.join(videos_path, 'poses.mp4'), np.stack(pose_imgs_list), fps=30, quality=6)
        imageio.mimwrite(os.path.join(videos_path, 'render.mp4'), np.stack(render_imgs_list), fps=30, quality=6)
        imageio.mimwrite(os.path.join(videos_path, 'depth.mp4'), np.stack(render_depth_list), fps=30, quality=6)

    if name == "train":        
        colmap_path = os.path.join(model_path, name)
        focals = [views[0].intrinsic[0, 0].detach().cpu().numpy()] * len(views)
        focals = np.array(focals)[..., None]
        principal_points = [views[0].intrinsic[:2, 2].detach().cpu().numpy()] * len(views)
        principal_points = np.array(principal_points)
        image_shape = views[0].original_image.shape
        world2cam_np = []
        for cam in views:
            Rt = np.eye(4)
            Rt[:3, :3] = cam.R.t().cpu().numpy()
            Rt[:3, 3] = cam.T.cpu().numpy()
            world2cam_np.append(Rt)
        world2cam_np = np.array(world2cam_np)
        name_list = np.array(name_list)
        save_cameras(focals, principal_points, colmap_path, imgs_shape=image_shape)
        save_imagestxt(world2cam_np, colmap_path, name_list)        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        dataset.load_pose = True
        scene = Scene(dataset, gaussians, load_iteration=iteration)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
    if not skip_train:
        with torch.no_grad():
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            render_nvs(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
    if not skip_test:
        from utils.mast3r_utils import Mast3rMatcher
        matcher = Mast3rMatcher()
        
        for idx, viewpoint in enumerate(tqdm(scene.getTestCameras(), desc="Processing test cameras")):
            if "hike_dataset" in dataset.model_path:
                test_frame_every = 10
            elif "Tanks" in dataset.model_path:
                test_frame_every = 2 if "Family" in dataset.model_path else 8
            else:
                test_frame_every = 8
            next_train_idx = viewpoint.uid * test_frame_every - idx
            if next_train_idx > len(scene.getTrainCameras()) - 1:
                next_train_idx = len(scene.getTrainCameras()) - 1
            ref_viewpoint = scene.getTrainCameras()[next_train_idx]

            vis_loc(viewpoint, ref_viewpoint, gaussians, pipeline, background, matcher)
            save_transforms(scene.getTestCameras().copy(), os.path.join(scene.model_path, "cameras_all_test.json"))
        with torch.no_grad():
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
