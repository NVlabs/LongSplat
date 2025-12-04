# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import math
import numpy as np
from typing import NamedTuple
import torch.nn.functional as F
import cv2
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points: torch.Tensor, transf_matrix: torch.Tensor) -> torch.Tensor:
    points_hom = torch.nn.functional.pad(points, (0, 1), value=1.0)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))
    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt.float()

def getWorld2View_np(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.t().cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_depth_edge_mask(depth):
    depth_np = depth.cpu().numpy()
    grad_x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
    edge_mask = np.sqrt(grad_x**2 + grad_y**2) > 1.0
    return torch.tensor(edge_mask, device=depth.device).reshape(-1)

def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge

def depth2pointcloud(depth, extrinsic, intrinsic):
    H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing='ij')
    z = depth
    edge_mask = depth_edge(depth).reshape(-1)
    x = (u - W * 0.5) * z / intrinsic[0, 0]
    y = (v - H * 0.5) * z / intrinsic[1, 1]
    xyz = torch.stack([x, y, z], dim=0).reshape(3, -1).T
    xyz = geom_transform_points(xyz, extrinsic)[~edge_mask]
    return xyz.float()

def warping(depth: torch.Tensor, extrinsic1: torch.Tensor, extrinsic2: torch.Tensor, 
            intrinsic: torch.Tensor, keypoint: torch.Tensor) -> torch.Tensor:
    H, W = depth.shape
    W_half = W * 0.5
    H_half = H * 0.5
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    
    keypoint_px = (keypoint * 0.5 + 0.5) * torch.tensor([W, H], dtype=keypoint.dtype, device=keypoint.device)
    keypoints_norm = keypoint.unsqueeze(0)
    depth_values = F.grid_sample(depth.unsqueeze(0).unsqueeze(0), keypoints_norm.unsqueeze(0), mode='bilinear', align_corners=True).squeeze()
    
    xy_normalized = (keypoint_px - torch.tensor([W_half, H_half], dtype=keypoint.dtype, device=keypoint.device)) / torch.tensor([fx, fy], dtype=keypoint.dtype, device=keypoint.device)
    xyz = torch.cat([xy_normalized, torch.ones_like(depth_values).unsqueeze(-1)], dim=-1) * depth_values.unsqueeze(-1)
    
    xyz_world = geom_transform_points(xyz, extrinsic1)
    xyz_camera2 = geom_transform_points(xyz_world, extrinsic2)
    
    z_cam2 = xyz_camera2[:, 2:3]
    xy_proj_px = xyz_camera2[:, :2] * torch.tensor([fx, fy], dtype=keypoint.dtype, device=keypoint.device) / z_cam2 + torch.tensor([W_half, H_half], dtype=keypoint.dtype, device=keypoint.device)
    uv_proj = xy_proj_px / torch.tensor([W, H], dtype=keypoint.dtype, device=keypoint.device) * 2 - 1
    return uv_proj

def unporject(depth, extrinsic, intrinsic, keypoint=None):
    H, W = depth.shape
    if keypoint is not None:
        keypoint_x = (keypoint[:, 0] / 2 + 0.5) * W
        keypoint_y = (keypoint[:, 1] / 2 + 0.5) * H
        keypoints_norm = torch.stack([keypoint[:, 0], keypoint[:, 1]], dim=1).unsqueeze(0)
        depth_values = F.grid_sample(depth.unsqueeze(0).unsqueeze(0), keypoints_norm.unsqueeze(0), mode='bilinear', align_corners=True)
        depth_values = depth_values.squeeze()
    else:
        keypoint_x, keypoint_y = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing='ij')
        keypoint_x = keypoint_x.reshape(H*W)
        keypoint_y = keypoint_y.reshape(H*W)
        depth_values = depth.reshape(H*W)
    z = depth_values
    x = (keypoint_x - W * 0.5) * z / intrinsic[0, 0]
    y = (keypoint_y - H * 0.5) * z / intrinsic[1, 1]
    xyz = torch.stack([x, y, z], dim=0).T
    xyz_world = geom_transform_points(xyz, extrinsic)
    return xyz_world  

def get_occlusion_mask(viewpoint_cam, viewpoint_cam2, depth, device, thresh=0.01):
    # Get resolution and intrinsic parameters for view1 and view2
    H1, W1 = viewpoint_cam.image_height, viewpoint_cam.image_width
    H2, W2 = viewpoint_cam2.image_height, viewpoint_cam2.image_width

    # Create (row, col) coordinate grid for each pixel in view1
    v, u = torch.meshgrid(
        torch.arange(H1, device=device),
        torch.arange(W1, device=device),
        indexing='ij'
    )
    
    # Lift view1 pixels to camera coordinates based on depth
    # (Note: assumes depth has same dimensions as view1 resolution)
    z = depth
    x = (u - W1 * 0.5) * z / viewpoint_cam.Focalx
    y = (v - H1 * 0.5) * z / viewpoint_cam.Focaly
    # Convert to point set of shape (N,3), N = H1 * W1
    xyz = torch.stack([x, y, z], dim=0).reshape(3, -1).T

    # Transform from view1 camera coordinates to world coordinates, then to view2 camera coordinates
    xyz_world = geom_transform_points(xyz, viewpoint_cam.view_world_transform.detach())
    xyz_view2 = geom_transform_points(xyz_world, viewpoint_cam2.world_view_transform).T
    xyz_view2 = xyz_view2.reshape(3, H1, W1)

    # Project view2 camera coordinates onto view2's pixel plane
    # Using view2's intrinsic parameters (Focalx, Focaly) and resolution (W2, H2)
    x_proj = xyz_view2[0, :, :] / xyz_view2[2, :, :] * viewpoint_cam2.Focalx + W2 * 0.5
    y_proj = xyz_view2[1, :, :] / xyz_view2[2, :, :] * viewpoint_cam2.Focaly + H2 * 0.5

    # Create occupancy map for view2 (accumulate warp contributions)
    occ_map = torch.zeros((H2, W2), device=device)
    
    # Only consider points warped within view2's bounds (Note: can also check z>0)
    valid = (x_proj >= 0) & (x_proj <= W2 - 1) & (y_proj >= 0) & (y_proj <= H2 - 1)
    x_valid = x_proj[valid]
    y_valid = y_proj[valid]
    
    # Get the four integer pixels around each floating point position
    x0 = torch.floor(x_valid).long()
    y0 = torch.floor(y_valid).long()
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Clamp indices to avoid out of bounds
    x0 = x0.clamp(0, W2 - 1)
    x1 = x1.clamp(0, W2 - 1)
    y0 = y0.clamp(0, H2 - 1)
    y1 = y1.clamp(0, H2 - 1)
    
    # Calculate bilinear splatting weights
    wa = (x1.float() - x_valid) * (y1.float() - y_valid)
    wb = (x_valid - x0.float()) * (y1.float() - y_valid)
    wc = (x1.float() - x_valid) * (y_valid - y0.float())
    wd = (x_valid - x0.float()) * (y_valid - y0.float())
    
    # Accumulate contribution of each valid view1 pixel to view2
    occ_map.index_put_((y0, x0), wa, accumulate=True)
    occ_map.index_put_((y0, x1), wb, accumulate=True)
    occ_map.index_put_((y1, x0), wc, accumulate=True)
    occ_map.index_put_((y1, x1), wd, accumulate=True)

    # Consider positions with accumulated weight greater than threshold as warped
    occlusion_mask = occ_map > thresh

    return occlusion_mask

def compute_scale(depth1, depth2, keypoint):
    H, W = depth1.shape
    keypoints_norm = torch.stack([keypoint[:, 0], keypoint[:, 1]], dim=1).unsqueeze(0)
    kp_depth1 = F.grid_sample(depth1.unsqueeze(0).unsqueeze(0), keypoints_norm.unsqueeze(0), mode='bilinear', align_corners=True).squeeze()
    kp_depth2 = F.grid_sample(depth2.unsqueeze(0).unsqueeze(0), keypoints_norm.unsqueeze(0), mode='bilinear', align_corners=True).squeeze()

    t_depth1 = torch.median(kp_depth1)
    s_depth1 = torch.mean(torch.abs(kp_depth1 - t_depth1))

    t_depth2 = torch.median(kp_depth2)
    s_depth2 = torch.mean(torch.abs(kp_depth2 - t_depth2))

    scale = s_depth1 / s_depth2
    offset = t_depth1 - t_depth2 * scale

    return scale, offset

def get_depth_from_3d_points(pts_3d, w2c_pose):
    pts_camera_h = geom_transform_points(pts_3d, w2c_pose)
    depth = pts_camera_h[:, 2]  # The z-coordinate gives the depth
    
    return depth