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
from gaussian_renderer import render3dgs, render
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.loss_utils import l1_loss, ssim
from random import randint
from utils.image_utils import psnr
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

def render_sets(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipe : PipelineParams, pruning_ratio : float = 0.6):
    """
    Convert and train 3D Gaussian Splatting (3DGS) model.
    
    The workflow is:
    1. Initialize 3DGS with anchor + offset points as initial positions
    2. Apply pruning to remove less important gaussians
    3. Convert to 3DGS format and fix xyz while training other parameters
    4. Train only other parameters (scaling, rotation, opacity, features) while keeping xyz fixed
    """
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                            dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    opt.iterations = iteration
    dataset.load_pose = True
    first_iter = 0
    scene = Scene(dataset, gaussians, load_iteration=-1)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    converted_3dgs_dir = os.path.join(dataset.model_path, "converted_3dgs")
    os.makedirs(converted_3dgs_dir, exist_ok=True)
    
    # Step 1: Pruning - Remove less important gaussians based on visibility and contribution
    # This helps reduce computational cost and improve quality by keeping only significant gaussians
    with torch.no_grad():
        gaussians.train()
        anchor_touched_list = torch.zeros(gaussians.get_anchor.shape[0] * gaussians.n_offsets, device="cuda")
        for view in scene.getTrainCameras():
            render_pkg = render(view, gaussians, pipe, background, retain_grad=False)
            n_touched = render_pkg["n_touched"]
            offset_selection_mask = render_pkg["selection_mask"]
            voxel_visible_mask = render_pkg["visible_mask"]
            visible_mask_expand = voxel_visible_mask.unsqueeze(0).expand(gaussians.n_offsets, -1).reshape(-1)
            visible_indices = torch.nonzero(visible_mask_expand).squeeze(1)
            final_indices = visible_indices[offset_selection_mask]
            anchor_touched_list[final_indices] += n_touched
    
    purning_mask = torch.zeros_like(anchor_touched_list, dtype=torch.bool)
    _, indices = torch.sort(anchor_touched_list)
    purning_mask[indices[int(len(anchor_touched_list)*pruning_ratio):]] = True
    
    # Step 2: Convert to 3DGS format
    # This converts the anchor+offset structure to standard 3DGS format
    # and applies the pruning mask to remove less important gaussians
    gaussians.convert_to_3dgs(purning_mask)
    
    # Step 3: Setup training for 3DGS
    gaussians.training_setup_3dgs(opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        image = render3dgs(viewpoint_cam, gaussians, pipe, background)["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        scaling_reg = gaussians.get_scaling.prod(dim=1).mean()
        loss += scaling_reg * 0.01

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()
                gaussians.save_ply_3dgs(os.path.join(converted_3dgs_dir, "point_cloud.ply"))

                if dataset.eval:
                    psnr_test = 0.0
                    for view in scene.getTestCameras():
                        image = render3dgs(view, gaussians, pipe, background)["render"]
                        gt_image = view.original_image.cuda()
                        psnr_test += psnr(image, gt_image).mean().double()
                    psnr_test /= len(scene.getTestCameras())
                    print(f"Test PSNR: {psnr_test}")
            
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=10000, type=int, help="Number of training iterations (use more iterations for longer video sequences)")
    parser.add_argument("--prune_ratio", default=0.6, type=float, help="Ratio of gaussians to prune (default: 0.6)")
    args = get_combined_args(parser)
    print("Converting " + args.model_path)

    render_sets(model.extract(args), op.extract(args), args.iteration, pipeline.extract(args), args.prune_ratio)
