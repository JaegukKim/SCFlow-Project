import torch.nn as nn
from HGPose.model.backbone import RAFT_small, RAFT_large, PoseHead, Aggregator, HeavyPoseHead
from HGPose.utils.geometry import (apply_forward_flow, apply_backward_flow, apply_imagespace_prev,
	get_2d_coord, apply_imagespace_relative, render_geometry, get_separate_medoid)
from collections import defaultdict

class OpticalFlow(nn.Module):
	def __init__(self, flow_refine_step):
		super(OpticalFlow, self).__init__()
		self.model = RAFT_large(flow_refine_step)
	def forward(self, ref_img, que_img):
		N = max(ref_img.shape[1], que_img.shape[1])
		bsz, _, c, H, W = ref_img.shape
		ref_img = ref_img.expand(bsz, N, c, H, W).flatten(0, 1)
		que_img = que_img.expand(bsz, N, c, H, W).flatten(0, 1)
		flow = self.model(ref_img, que_img)
		flow = [f.unflatten(0, (bsz, N)) for f in flow]
		return flow

class GeometryAggregator(nn.Module):
	def __init__(self, g_dim):
		super(GeometryAggregator, self).__init__()
		self.model = Aggregator(g_dim)
	def forward(self, recon_que_geo):
		pred_que_geo = self.model(recon_que_geo)
		return pred_que_geo

class FlowModel(nn.Module):
	def __init__(self, cfg):
		super(FlowModel, self).__init__()
		self.img_size = cfg.img_size
		self.represent_mode = cfg.represent_mode
		self.N_freq = cfg.N_freq
		self.flow_refine_step = cfg.flow_refine_step
		if self.represent_mode == 'positional':
			g_dim = 3 * 2 * self.N_freq
		else:
			g_dim = 3
		self.opticalflow = OpticalFlow(self.flow_refine_step)
		self.aggregator = GeometryAggregator(g_dim=g_dim)

	def forward(self, ref_img, ref_geo, que_img, que_mask):
		pred = defaultdict()
		que_img = que_img * que_mask
		pred['flow_r2q'] = self.opticalflow(ref_img, que_img)
		pred['recon_geo'] = apply_forward_flow(pred['flow_r2q'][-1], ref_geo, invalid_val=-1.0)
		pred['recon_geo'] = pred['recon_geo'] * que_mask
		pred['geo'] = self.aggregator(pred['recon_geo'])
		return pred