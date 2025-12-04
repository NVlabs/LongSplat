# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View, getProjectionMatrix
from utils.graphics_utils import fov2focal, focal2fov

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 R_gt = None, T_gt = None, disable_resize=False):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.image_name = image_name
        self.is_registered = False

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if R_gt is not None and T_gt is not None:
            self.R_gt = torch.tensor(R_gt)
            self.T_gt = torch.tensor(T_gt)
        else:
            self.R_gt = None
            self.T_gt = None

        if R is not None and T is not None:
            self.R_pred = torch.tensor(R)
            self.T_pred = torch.tensor(T)
        else:
            self.R_pred = None
            self.T_pred = None

        t = torch.eye(4, device=data_device)
        self.R = t[:3, :3]
        self.T = t[:3, 3]

        with torch.no_grad():
            max_side = max(image.shape[1], image.shape[2])
            if not disable_resize and max_side > 512:
                resize_factor = 512.0 / max_side
                image_resize = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=resize_factor, mode='bilinear', align_corners=True).squeeze(0)
            else:
                image_resize = image

        self.original_image = image_resize.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.original_image_final = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width_final = self.original_image_final.shape[2]
        self.image_height_final = self.original_image_final.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.kp0 = None
        self.kp1 = None
        self.conf = None

        self.depth_map = None
        self.pre_depth_map = None
        self.pts3d = None
        
        if FoVx is None or FoVy is None:
            self.FoVx = None
            self.FoVy = None
            self.Focalx = None
            self.Focaly = None
            self.intrinsic = None
            self.projection_matrix = None
        else:
            self.FoVx = FoVx
            self.FoVy = FoVy
            self.Focalx = fov2focal(FoVx, self.image_width)
            self.Focaly = fov2focal(FoVy, self.image_height)
            self._tanfovx = None
            self._tanfovy = None
            self.intrinsic = torch.tensor([[self.Focalx, 0, self.image_width / 2], [0, self.Focaly, self.image_height / 2], [0, 0, 1]]).cuda()
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        self.cam_rot_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=data_device))
        self.cam_trans_delta = nn.Parameter(torch.zeros(3, requires_grad=True, device=data_device))

    @property
    def world_view_transform(self):
        return getWorld2View(self.R, self.T).transpose(0, 1)
    
    @property
    def view_world_transform(self):
        return self.world_view_transform.inverse()

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    @property
    def tanfovx(self):
        if self._tanfovx is None:
            import math
            self._tanfovx = math.tan(self.FoVx * 0.5)
        return self._tanfovx
    
    @property
    def tanfovy(self):
        if self._tanfovy is None:
            import math
            self._tanfovy = math.tan(self.FoVy * 0.5)
        return self._tanfovy
    
    def to_final(self):
        self.original_image = self.original_image_final
        self.image_width = self.image_width_final
        self.image_height = self.image_height_final
        self.Focalx = fov2focal(self.FoVx, self.image_width_final)
        self.Focaly = fov2focal(self.FoVy, self.image_height_final)
        self.intrinsic = torch.tensor([[self.Focalx, 0, self.image_width_final / 2], [0, self.Focaly, self.image_height_final / 2], [0, 0, 1]]).cuda()
        if self.depth_map is not None:
            self.depth_map = torch.nn.functional.interpolate(self.depth_map.unsqueeze(0).unsqueeze(0), size=(self.image_height_final, self.image_width_final), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)

    def update_RT(self, R, t):
        self.R = R.to(device=self.data_device)
        self.T = t.to(device=self.data_device)

    def update_focal(self, focal_length):
        self.FoVx = focal2fov(focal_length, self.image_width)
        self.FoVy = focal2fov(focal_length, self.image_height)
        self.Focalx = focal_length
        self.Focaly = focal_length
        self.intrinsic = torch.tensor([[self.Focalx, 0, self.image_width / 2], [0, self.Focaly, self.image_height / 2], [0, 0, 1]]).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self._tanfovx = None
        self._tanfovy = None

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

