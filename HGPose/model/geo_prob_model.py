import torch
import torch.nn as nn
import torch.nn.functional as F
from HGPose.model.backbone_scflow import ResNet18, ResNet34, ConvNextTiny, UpsampleHeadProb, PoseHead
from HGPose.utils.geometry import (
	PositionEmbeddingSine3D, get_distmap, get_2d_coord, apply_imagespace_relative, render_geometry, default_RT, get_separate_medoid, get_medoid,
	get_best_candidate, get_average)
from pytorch3d.transforms import rotation_6d_to_matrix
from collections import OrderedDict

class Backbone(nn.Module):
	def __init__(self, input_dim=3, model=None):
		super(Backbone, self).__init__()
		if model == 'res34':
			self.model = ResNet34(input_dim)
		elif model == 'convnext_tiny':
			self.model = ConvNextTiny(input_dim)
	def forward(self, query_views, input_views=None):
		bsz, n_que = query_views.shape[0:2]
		query_feat = OrderedDict() 
		if input_views is not None:
			input_feat = OrderedDict()
			n_inp = input_views.shape[1]
			stack = torch.cat([query_views, input_views], dim=1)        # (bsz, n_que+n_inp, 3, h, w)
			stack = stack.flatten(0, 1)                                 # (bsz * (n_que+n_inp), 3, h, w)
			f = self.model(stack)                                       # (bsz * (n_que+n_inp), feat_dim, h, w)
			for level, feature in f.items():
				feature = feature.unflatten(0, (bsz, (n_que+n_inp)))                  # (b, n_inp, feat_dim, h, w)
				query_feat[level] = feature[:, :n_que] 
				input_feat[level] = feature[:, n_que:]
			return query_feat, input_feat
		else:
			stack = query_views.flatten(0, 1)
			f = self.model(stack)             
			for level, feature in f.items():
				feature = feature.unflatten(0, (bsz, n_que))                  # (b, n_inp, feat_dim, h, w)
				query_feat[level] = feature[:, :n_que] 
			return query_feat

class Upsampler(nn.Module):
	def __init__(self, connect_level, feature_dim, geometry_dim):
		super(Upsampler, self).__init__()
		self.connect_level = connect_level
		self.feature_dim = feature_dim
		self.geometry_dim = geometry_dim
		self.model = UpsampleHeadProb(feature_dim=self.feature_dim, coord_dim=self.geometry_dim)

	def forward(self, query_backbone_feat, index_feat=None):                            # (b, n_que, feature_dim, h, w) (b, n_que*h*w, feature_dim)
		f = OrderedDict()
		for level in ['feat3', 'feat2', 'feat1']:
			bsz, n_que, _, h, w = query_backbone_feat[level].shape
			f[level] = query_backbone_feat[level]                               # (b, n_que, feature_dim, h, w)
			f[level] = f[level].flatten(0, 1)                                   # (b * n_que, feature_dim, h, w)
		coord, masks, mask_visibs = self.model(f, index_feat)                       # (b * n_que, 3, H, W), (b * n_que, 2, H, W), (b * n_que, 65, H, W)
		coord = coord.unflatten(0, (bsz, n_que))
		masks = masks.unflatten(0, (bsz, n_que))
		mask_visibs = mask_visibs.unflatten(0, (bsz, n_que))
		return coord, masks, mask_visibs


class GeoProbModel(nn.Module):
	def __init__(self, cfg):
		super(GeoProbModel, self).__init__()
		self.connect_level = cfg.connect_level
		self.image_size = cfg.image_size
		self.feature_size = cfg.feature_size
		# self.ref_size = cfg.ref_size
		# self.temperature = cfg.temperature
		# self.num_query_views = cfg.num_query_views
		# self.num_input_views = cfg.num_input_views
		# self.num_fw_layers = cfg.fw_decoder.num_layers
		self.represent_mode = cfg.represent_mode
		self.ray_mode = cfg.ray_mode
		self.num_class = cfg.num_class
		# self.n_freqs = cfg.n_freqs
		# self.is_adain = cfg.is_adain
		# self.additional_step = 4
		if self.ray_mode and self.represent_mode == 'positional':
			self.geometry_dim = 3 * 2 * self.n_freqs + 3 * 2 * int(self.n_freqs*2/5)
		elif self.represent_mode == 'positional':
			self.geometry_dim = 3 * 2 * self.n_freqs
		elif self.ray_mode:
			self.geometry_dim = 6
		else:
			self.geometry_dim = 3
		self.backbone_model = cfg.backbone_model
		self.backbone = Backbone(input_dim=3, model=self.backbone_model)
		self.upsampler = Upsampler(self.connect_level, self.backbone.model.block_feature_dim, self.geometry_dim)
		# self.pose_head = GeoPoseHead(self.num_class)
		# self.pose_estimator_1 = PoseEstimator(self.geometry_dim * 3 + 10, [self.ref_size, self.ref_size])  #  xyz_dim geometry_dim*2 + mask 1*2 + mask_visib 1*2 + error geometry_dim + view 3*2 
		# self.pose_estimator_2 = PoseEstimator(self.geometry_dim * 3 + 4,  [self.ref_size, self.ref_size])  #  xyz_dim geometry_dim*2 + mask 1*2 + mask_visib 1*2 + error geometry_dim 

	def forward(self, query_view, obj_index):
		query_backbone_feat = self.backbone(query_view)
		pred_coord, pred_mask, pred_mask_visib = self.upsampler(query_backbone_feat)   
		pred_coord, pred_mask, pred_mask_visib = pred_coord.squeeze(1), pred_mask.squeeze(1), pred_mask_visib.squeeze(1)
		# pred_rotation, pred_translation = self.pose_head(pred_coord, pred_mask, pred_mask_visib, obj_index)
		# rotation = rotation_6d_to_matrix(pred_rotation)

		# pred_RT = torch.cat([pred_rotation, pred_translation], -1)
		

		return pred_coord, pred_mask, pred_mask_visib
