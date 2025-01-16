import torch
import torch.nn as nn
import torch.nn.functional as F
from HGPose.model.backbone import CustomPoseHead
from HGPose.utils.geometry import (apply_forward_flow, apply_backward_flow, apply_imagespace_prev, get_2d_coord, apply_imagespace_relative, render_geometry, get_separate_medoid)
from collections import defaultdict
from kornia.utils import create_meshgrid

class RefineModel(nn.Module):
	def __init__(self, cfg):
		super(RefineModel, self).__init__()
		self.refine_step = cfg.refine_step
		self.img_size = cfg.img_size
		self.N_z = cfg.N_z
		self.N_freq = cfg.N_freq
		self.represent_mode = cfg.represent_mode
		if self.represent_mode == 'positional':
			self.input_dim = 2 + 12 * self.N_freq
		else:
			self.input_dim = 8
		self.model = CustomPoseHead(f_dim=self.input_dim, size=[self.img_size, self.img_size])
		self.register_buffer('uv', 
			create_meshgrid(self.img_size, self.img_size, False).permute(0, 3, 1, 2).unsqueeze(1))

	def forward(self, source, initial_RT, pred_geo, que_K_img, que_mask):
		bsz = pred_geo.shape[0]
		pred = defaultdict()
		pred['RT_1'] = initial_RT.clone()
		pred_geo = (pred_geo + 1) * que_mask - 1
		for i in range(1, self.refine_step):
			refine_geo = render_geometry(source, que_K_img, 
				pred[f'RT_{i}'], self.img_size, self.img_size, self.N_z, self.N_freq, self.represent_mode)
			refine_geo = (refine_geo + 1) * que_mask - 1
			stack = torch.cat([self.uv.repeat(bsz, 1, 1, 1, 1),	pred_geo, refine_geo], dim=-3)
			relative = self.model(stack.flatten(0, 1)).unflatten(0, (bsz, -1))
			pred[f'RT_{i+1}'] = apply_imagespace_relative(
				que_K_img, que_K_img, pred[f'RT_{i}'].detach(), relative, [self.img_size, self.img_size])
		return pred
