import torch
import torch.nn as nn
import torch.nn.functional as F
from HGPose.model.backbone import RAFT_small, RAFT_large, RAFT_custom, PoseHead, Aggregator, HeavyPoseHead
from HGPose.utils.geometry import (apply_forward_flow, apply_backward_flow, apply_imagespace_prev,
	get_2d_coord, apply_imagespace_relative, render_geometry, get_separate_medoid)
from collections import defaultdict
from kornia.utils import create_meshgrid

class OpticalFlowCustom(nn.Module):
	def __init__(self, size):
		super(OpticalFlowCustom, self).__init__()
		self.model = RAFT_custom(size)
	def forward(self, ref_img, que_img, ref_K_img, que_K_img, ref_RT, ref_geo, ref_mask):
		N = max(ref_img.shape[1], que_img.shape[1])
		bsz, _, c, H, W = ref_img.shape
		ref_img = ref_img.expand(bsz, N, c, H, W).flatten(0, 1)
		que_img = que_img.expand(bsz, N, c, H, W).flatten(0, 1)
		ref_K_img = ref_K_img.expand(bsz, N, 3, 3).flatten(0, 1)
		que_K_img = que_K_img.expand(bsz, N, 3, 3).flatten(0, 1)
		ref_RT = ref_RT.expand(bsz, N, 4, 4).flatten(0, 1)
		ref_geo = ref_geo.expand(bsz, N, c, H, W).flatten(0, 1)
		ref_mask = ref_mask.expand(bsz, N, 1, H, W).flatten(0, 1)
		flow, pred_RT = self.model(ref_img, que_img, ref_K_img, que_K_img, ref_RT, ref_geo, ref_mask)
		for i in range(len(flow)):
			flow[i] = flow[i].reshape(bsz, N, 2, H, W)
			pred_RT[i] = pred_RT[i].reshape(bsz, N, 4, 4)
		return flow, pred_RT

class OpticalFlow(nn.Module):
	def __init__(self, size):
		super(OpticalFlow, self).__init__()
		self.model = RAFT_large()
	def forward(self, ref_img, que_img):
		N = max(ref_img.shape[1], que_img.shape[1])
		bsz, _, c, H, W = ref_img.shape
		ref_img = ref_img.expand(bsz, N, c, H, W).flatten(0, 1)
		que_img = que_img.expand(bsz, N, c, H, W).flatten(0, 1)
		flow = self.model(ref_img, que_img)
		for i in range(len(flow)):
			flow[i] = flow[i].reshape(bsz, N, 2, H, W)
		return flow

class PoseEstimator(nn.Module):
	def __init__(self, input_dim, size):
		super(PoseEstimator, self).__init__()
		self.input_dim = input_dim
		self.size = size
		# self.model = PoseHead(f_dim=self.input_dim, size=self.size)
		self.model = HeavyPoseHead(input_dim=self.input_dim, size=self.size)
		self.register_buffer('uv', create_meshgrid(self.size[0], self.size[1], False).permute(0, 3, 1, 2).unsqueeze(1))
	def forward(self, K_q, K_i, RT_i, mask_p, xyz_p, xyz_i):
		bsz, N = xyz_i.shape[:2]
		xyz_p = torch.cat([xyz_p.expand_as(xyz_i), self.uv.repeat(bsz, N, 1, 1, 1)], dim=-3)
		xyz_i = torch.cat([xyz_i, self.uv.repeat(bsz, N, 1, 1, 1)], dim=-3)
		stack = torch.cat([mask_p, xyz_p, xyz_i], dim=-3).flatten(0, 1)
		relative = self.model(stack).unflatten(0, (bsz, N))
		RT_q = apply_imagespace_prev(K_q, RT_i, relative, self.size)
		return RT_q

class GeometryAggregator(nn.Module):
	def __init__(self, g_dim):
		super(GeometryAggregator, self).__init__()
		self.model = Aggregator(g_dim)
	def forward(self, recon_que_geo, recon_que_mask, que_img):
		pred_que_geo, pred_que_mask = self.model(recon_que_geo, recon_que_mask, que_img)
		return pred_que_geo, pred_que_mask



class StructureFreeTransformer(nn.Module):
	def __init__(self, cfg):
		super(StructureFreeTransformer, self).__init__()
		self.img_size = cfg.img_size
		self.geo_size = cfg.geo_size
		self.N_que = cfg.N_que
		self.N_ref = cfg.N_ref
		self.ray_mode = cfg.ray_mode
		self.represent_mode = cfg.represent_mode
		self.n_freqs = cfg.n_freqs
		self.refine_step = cfg.refine_step
		self.g_dim = 3
		if self.represent_mode == 'positional':
			self.g_dim = 2 * self.g_dim * self.n_freqs
		self.opticalflow = OpticalFlow([self.img_size, self.img_size])
		self.aggregator = GeometryAggregator(self.g_dim)

		self.pose_estimator_coarse = PoseEstimator((self.g_dim+2) * 2 + 1, [self.img_size, self.img_size])  #  xyz_dim g_dim*2
		self.pose_estimator_fine = PoseEstimator((self.g_dim+2) * 2 + 1, [self.img_size, self.img_size])  #  xyz_dim g_dim*2

	def forward(self, xyz_mesh, ref_img, ref_mask, ref_geo, que_RT_coarse, que_img, que_K_img, mode):
		pred = defaultdict()
		pred['flow_r2q'] = self.opticalflow(ref_img, que_img)
		pred['recon_geo'] = apply_forward_flow(pred['flow_r2q'][-1], ref_geo, invalid_val=-1.0)
		pred['recon_mask'] = apply_forward_flow(pred['flow_r2q'][-1], ref_mask, invalid_val=0.0)
		pred['geo'], pred['mask'] = self.aggregator(pred['recon_geo'], pred['recon_mask'], que_img)

		pred['RT_0'] = que_RT_coarse.clone().detach()
		if mode != 'flow':
			coarse_geo = render_geometry(
				xyz_mesh, 
				que_K_img, 
				pred[f'RT_0'].detach(), 
				self.img_size, 
				self.n_freqs, 
				self.represent_mode, 
				self.ray_mode)
			pred[f'RT_1'] = self.pose_estimator_coarse(
				que_K_img, 
				que_K_img, 
				pred[f'RT_0'].detach(), 
				pred['mask'].detach(),
				pred['geo'].detach(),
				coarse_geo)

			for i in range(1, self.refine_step):
				refine_geo = render_geometry(
					xyz_mesh, 
					que_K_img, 
					pred[f'RT_{i}'], 
					self.img_size, 
					self.n_freqs, 
					self.represent_mode, 
					self.ray_mode)
				pred[f'RT_{i+1}'] = self.pose_estimator_fine(
					que_K_img, 
					que_K_img, 
					pred[f'RT_{i}'], 
					pred['mask'].detach(),
					pred['geo'].detach(),
					refine_geo)
		else:
			pred[f'RT_1'] = pred['RT_0'].clone()
			for i in range(1, self.refine_step):
				pred[f'RT_{i+1}'] = pred[f'RT_{i}'].clone()
		return pred
