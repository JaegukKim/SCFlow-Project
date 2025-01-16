import torch
import torch.nn as nn
import torch.nn.functional as F
from HGPose.model.backbone import CoarsePoseEstimator
from HGPose.utils.geometry import (apply_forward_flow, apply_backward_flow, apply_imagespace_prev,
	get_2d_coord, apply_imagespace_relative, render_geometry, get_separate_medoid)
from collections import defaultdict
from kornia.utils import create_meshgrid
import numpy as np

class CoarseModel(nn.Module):
	def __init__(self, cfg):
		super(CoarseModel, self).__init__()
		self.tau = 0.1
		self.size = cfg.coarse_img_size
		self.cos = nn.CosineSimilarity(dim=-3, eps=1e-6)
		self.model = CoarsePoseEstimator()

	def forward(self, que_img, ref_img):
		bsz, N_ref = ref_img.shape[:2]
		ref_img = F.interpolate(ref_img.flatten(0, 1), [self.size, self.size])
		que_img = F.interpolate(que_img.flatten(0, 1), [self.size, self.size])
		stack = torch.cat([ref_img, que_img], dim=0)
		feat = self.model(stack)
		ref_feat = feat[:bsz*N_ref].unflatten(0, (bsz, -1))
		que_feat = feat[bsz*N_ref:].unflatten(0, (bsz, -1))
		ref_feat = F.normalize(ref_feat, dim=-3)
		que_feat = F.normalize(que_feat, dim=-3)
		sim = self.cos(que_feat, ref_feat)
		sim = sim.mean([-2, -1]) / self.tau
		return sim