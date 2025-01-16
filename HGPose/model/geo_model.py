import torch
import torch.nn as nn
import torch.nn.functional as F
from HGPose.model.backbone_scflow import ResNet18, ResNet34, ConvNextTiny, UpsampleHead, PoseHead
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
		self.model = UpsampleHead(feature_dim=self.feature_dim, coord_dim=self.geometry_dim)

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
		# error = error.unflatten(0, (bsz, n_que))
		return coord, masks, mask_visibs

# class PoseEstimator(nn.Module):
# 	def __init__(self, input_dim, size):
# 		super(PoseEstimator, self).__init__()
# 		self.input_dim = input_dim
# 		self.size = size
# 		self.model = PoseHead(feature_dim=self.input_dim, size=self.size)
# 		# self.register_buffer('uv', get_2d_coord(self.size))
# 	def forward(self, 
# 				K_q, K_i, RT_i, 
# 				xyz_p, xyz_i, 
# 				mask_visib_p, mask_visib_i, 
# 				mask_p, mask_i, 
# 				error_p,
# 				view_q=None, view_i=None):
# 		xyz = torch.cat([xyz_p.expand(xyz_i.shape), xyz_i], dim=-3).flatten(0, 1)
# 		mask_visib = torch.cat([mask_visib_p.expand(mask_visib_i.shape), mask_visib_i], dim=-3).flatten(0, 1)
# 		mask = torch.cat([mask_p.expand(mask_i.shape), mask_i], dim=-3).flatten(0, 1)
# 		error = error_p.expand(xyz_i.shape).flatten(0, 1)
# 		feature = torch.cat([xyz, mask_visib, mask, error], dim=-3)
# 		if view_i is not None and view_q is not None:
# 			view = torch.cat([view_q.expand(view_i.shape), view_i], dim=-3).flatten(0, 1)
# 			view = F.interpolate(view, xyz.shape[-2:])
# 			feature = torch.cat([view, feature], dim=-3)
# 		relative = self.model(feature)
# 		relative = relative.unflatten(0, (-1, K_i.shape[1]))
# 		RT_q = apply_imagespace_relative(K_i, K_q, RT_i, relative, self.size)
# 		return RT_q
	

# class GeoPoseHead(nn.Module):
# 	def __init__(self, num_class):
# 		super(GeoPoseHead, self).__init__()
# 		self.num_class = num_class
# 		self.coord_encoder = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1), 
# 			nn.ReLU(),
# 			nn.Conv2d(64, 128, 3, 1, 1),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 64, 3, 2, 1),
# 			nn.ReLU())
# 		self.mask_encoder = nn.Sequential(
# 			nn.Conv2d(1, 64, 3, 1, 1),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 32, 3, 2, 1),
# 			nn.ReLU())
# 		self.mask_visib_encoder = nn.Sequential(
# 			nn.Conv2d(1, 64, 3, 1, 1),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 32, 3, 2, 1),
# 			nn.ReLU())
# 		self.conv_layers = nn.Sequential(
# 			nn.Conv2d(128, 64, 3, 2, 1),
# 			nn.GroupNorm(16, 64, eps=1e-05, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 2, 1),
# 			nn.GroupNorm(16, 64, eps=1e-05, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 2, 1),
# 			nn.GroupNorm(16, 64, eps=1e-05, affine=True),
# 			nn.ReLU()
# 		)
# 		self.flatten_op = nn.Flatten(start_dim=1, end_dim=-1)
# 		self.fc_layers = nn.Sequential(
# 			nn.Linear(1024, 512, bias=True),
# 			nn.ReLU(),
# 			nn.Linear(512, 256, bias=True),
# 			nn.ReLU()
# 		)
# 		self.rotation_pred = nn.Linear(256, 6*num_class)
# 		self.translation_pred = nn.Linear(256, out_features=3*num_class)
# 		self.init_weights()

# 	def init_weights(self):
# 		 # zero translation
# 		nn.init.zeros_(self.translation_pred.weight)
# 		nn.init.zeros_(self.translation_pred.bias)
# 		nn.init.zeros_(self.rotation_pred.weight)
# 		# identity quarention
# 		with torch.no_grad():
# 			self.rotation_pred.bias.copy_(torch.Tensor([1., 0., 0., 0., 1., 0.]*self.num_class))

# 	def forward(self, pred_coord, pred_mask, pred_mask_visib, obj_id):
# 		#hidden_encode = self.hidden_state_model(hidden_state)
# 		coord_encode = self.coord_encoder(pred_coord)
# 		mask_encode = self.mask_encoder(pred_mask)
# 		mask_visib_encode = self.mask_visib_encoder(pred_mask_visib)
# 		encode = torch.cat([coord_encode, mask_encode, mask_visib_encode], dim=-3)
# 		encode = self.conv_layers(encode)
# 		encode = self.flatten_op(encode)
# 		encode = self.fc_layers(encode)
# 		pred_translation = self.translation_pred(encode)
# 		pred_rotation = self.rotation_pred(encode)
# 		pred_translation = pred_translation.view(-1, self.num_class, 3)
# 		pred_rotation = pred_rotation.view(-1, self.num_class, 6)
# 		pred_translation = torch.index_select(pred_translation, dim=1, index=obj_id-1)[:, 0, :]
# 		pred_rotation = torch.index_select(pred_rotation, dim=1, index=obj_id-1)[:, 0, :]

# 		return pred_rotation, pred_translation
	 

class GeoModel(nn.Module):
	def __init__(self, cfg):
		super(GeoModel, self).__init__()
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
