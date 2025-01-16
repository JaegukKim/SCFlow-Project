import torch
import torch.nn as nn
import torch.nn.functional as F
from HGPose.model.backbone import CustomPoseHead
from HGPose.utils.geometry import (apply_forward_flow, apply_backward_flow, apply_imagespace_prev,
	get_2d_coord, apply_imagespace_relative, render_geometry, get_separate_medoid)
from collections import defaultdict
from kornia.utils import create_meshgrid

class RelativePoseEstimator(nn.Module):
	def __init__(self, cfg):
		super(RelativePoseEstimator, self).__init__()
		self.refine_step = cfg.refine_step
		self.N_freq = cfg.N_freq
		self.use_flow = cfg.use_flow
		self.use_recon_geo = cfg.use_recon_geo
		self.represent_mode = cfg.represent_mode
		if self.represent_mode == 'positional':
			self.input_dim = 2 + 12 * self.N_freq 
		else:
			self.input_dim = 2 + 6
		if self.use_flow:
			self.input_dim = self.input_dim + 2

		self.size = [cfg.img_size, cfg.img_size]
		self.model = CustomPoseHead(f_dim=self.input_dim, size=self.size)
		self.register_buffer('uv', 
			create_meshgrid(self.size[0], self.size[1], False).permute(0, 3, 1, 2).unsqueeze(1))

	def forward(self, ref_geo, ref_RT, ref_K_img, 
			pred_geo, pred_recon_geo, pred_flow, que_mask, que_K_img):
		bsz, N = ref_geo.shape[:2]
		pred = defaultdict()
		pred_geo = (pred_geo + 1) * que_mask - 1
		stack = [self.uv.repeat(bsz, N, 1, 1, 1), ref_geo]
		if self.use_flow:
			stack.append(pred_flow[-1])

		if self.use_recon_geo:
			stack.append(pred_recon_geo)
		else:
			stack.append(pred_geo.expand_as(ref_geo))
		stack = torch.cat(stack, dim=-3)
		relative = self.model(stack.flatten(0, 1)).unflatten(0, (bsz, N))
		pred['RT_candidate'] = apply_imagespace_relative(
			ref_K_img, que_K_img.expand_as(ref_K_img), ref_RT.detach(), relative, self.size)
		pred['RT_1'] = get_separate_medoid(pred['RT_candidate'])

		return pred
