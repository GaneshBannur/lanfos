#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
import os
import random
from lanfos.language_field.utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, lf_dir, lf_feature_level,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.setup_lf(lf_dir, lf_feature_level)

    @torch.no_grad()
    def setup_lf(self, lf_dir, lf_feature_level):
        lf_path = os.path.join(lf_dir, self.image_name+f"_f_{lf_feature_level}.npy")
        lf_seg_path = os.path.join(lf_dir, self.image_name+f"_s_{lf_feature_level}.npy")
        lf_list = np.load(lf_path)
        lf_seg = np.load(lf_seg_path)
        self.lf_list = torch.from_numpy(lf_list).to(self.data_device)
        self.lf_seg = torch.from_numpy(lf_seg).to(self.data_device)
        # Having shape [H, W, 1] will allow broadcasting with language features of shape [H, W, D]
        self.lf_bg_mask = (self.lf_seg!=-1).unsqueeze(-1)
        self.lf_num_considered = torch.count_nonzero(self.lf_bg_mask) * self.lf_list.shape[-1]
        self.lf_non_bg_flat_indices = torch.argwhere(self.lf_bg_mask.flatten()).squeeze().tolist()
        random.shuffle(self.lf_non_bg_flat_indices)
        self.lf_cur_non_bg_flat_indices = torch.tensor(self.lf_non_bg_flat_indices, device=self.data_device)

    @torch.no_grad()
    def get_lf_decoder_batch(self, lf_decoder_batch_size):
        if self.lf_cur_non_bg_flat_indices.numel()==0:
            random.shuffle(self.lf_non_bg_flat_indices)
            self.lf_cur_non_bg_flat_indices = torch.tensor(self.lf_non_bg_flat_indices, device=self.data_device)
        
        batch_indices = self.lf_cur_non_bg_flat_indices[:lf_decoder_batch_size]
        self.lf_cur_non_bg_flat_indices = self.lf_cur_non_bg_flat_indices[lf_decoder_batch_size:]
        batch_lf_list_indices = torch.index_select(input=self.lf_seg.flatten().to("cuda"), dim=0,
                                                   index=batch_indices.to("cuda"))
        batch_features = torch.index_select(input=self.lf_list.to("cuda"), dim=0, index=batch_lf_list_indices)
        batch_indices.requires_grad_(False)
        batch_features.requires_grad_(False)
        return batch_features, batch_indices

    @property
    @torch.no_grad()
    def gt_lf(self):
        per_pixel_lf = self.lf_list.to("cuda")[self.lf_seg.to("cuda")]
        per_pixel_lf.requires_grad_(False)
        return per_pixel_lf

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

