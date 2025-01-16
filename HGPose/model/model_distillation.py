import torch
import torch.nn as nn
import torch.nn.functional as F
from HGPose.model.backbone import ResNet18, ResNet34, ResNet50, ConvNextTiny, UpsampleHead, DINOv2_ViTs14, PoseHead
from HGPose.utils.geometry import (
	PositionEmbeddingSine3D, get_distmap, get_2d_coord, apply_imagespace_relative, render_geometry, 
	default_RT, get_separate_medoid, get_medoid, get_best_candidate, get_average)
from HGPose.model.HMMN import KeyValue, get_affinity, find_topk, attention, attention_topk, get_affinity_topk
from collections import OrderedDict

class Backbone(nn.Module):
	def __init__(self, model=None, input_dim=3):
		super(Backbone, self).__init__()
		if model == 'res34':
			self.model = ResNet34(input_dim)
		elif model == 'res50':
			self.model = ResNet50(input_dim)
		elif model == 'convnext_tiny':
			self.model = ConvNextTiny(input_dim)
		elif model == 'dinov2_Vits14':
			self.model = DINOv2_ViTs14()

	def forward(self, x):
		bsz, n_que = x.shape[0:2]
		stack = x.flatten(0, 1)
		feature = self.model(stack)             
		feat = {l: f.unflatten(0, (bsz, n_que)) for l, f in feature.items()}
		return feat
	

class Upsampler(nn.Module):
	def __init__(self, level, f_dim, g_dim):
		super(Upsampler, self).__init__()
		self.level = level
		self.f_dim = f_dim
		self.g_dim = g_dim
		self.model = UpsampleHead(f_dim=self.f_dim, g_dim=self.g_dim)
	def forward(self, que_val):
		bsz, n_que, _, _, _ = list(que_val.values())[0].shape
		feat = {l: que_val[l].flatten(0, 1) for l in self.level}
		geo, mask_visibs, error = self.model(feat)
		geo = geo.unflatten(0, (bsz, n_que))
		mask_visibs = mask_visibs.unflatten(0, (bsz, n_que))
		error = error.unflatten(0, (bsz, n_que))
		return geo, mask_visibs, error


class PoseEstimator(nn.Module):
	def __init__(self, input_dim, size):
		super(PoseEstimator, self).__init__()
		self.input_dim = input_dim
		self.size = size
		self.model = PoseHead(f_dim=self.input_dim, size=self.size)
		# self.register_buffer('uv', get_2d_coord(self.size))
	def forward(self, 
				K_q, K_i, RT_i, 
				xyz_p, xyz_i, 
				mask_visib_p, mask_visib_i,
				error_p,
				img_q=None, img_i=None):
		xyz = torch.cat([xyz_p.expand(xyz_i.shape), xyz_i], dim=-3).flatten(0, 1)
		mask_visib = torch.cat([mask_visib_p.expand(mask_visib_i.shape), mask_visib_i], dim=-3).flatten(0, 1)
		error = error_p.expand(xyz_i.shape).flatten(0, 1)
		feature = torch.cat([xyz, mask_visib, error], dim=-3)
		if img_i is not None and img_q is not None:
			img_q = F.interpolate(img_q.flatten(0, 1), xyz.shape[-2:]).unflatten(0, (-1, img_q.shape[1]))
			img = torch.cat([img_q.expand(img_i.shape), img_i], dim=-3).flatten(0, 1)
			feature = torch.cat([img, feature], dim=-3)
		relative = self.model(feature)
		relative = relative.unflatten(0, (-1, K_i.shape[1]))
		RT_q = apply_imagespace_relative(K_i, K_q, RT_i, relative, self.size)
		return RT_q


class StructureFreeTransformer(nn.Module):
	def __init__(self, cfg):
		super(StructureFreeTransformer, self).__init__()
		self.level = cfg.level
		self.img_size = cfg.img_size
		self.geo_size = cfg.geo_size
		self.N_que = cfg.N_que
		self.N_ref = cfg.N_ref
		self.topk = cfg.topk
		self.ray_mode = cfg.ray_mode
		self.key_dim = cfg.key_dim
		self.represent_mode = cfg.represent_mode
		self.n_freqs = cfg.n_freqs
		self.additional_step = cfg.additional_step
		self.temperature = cfg.temperature
		self.g_dim = 3
		if self.represent_mode == 'positional':
			self.g_dim = 2 * self.g_dim * self.n_freqs
		
		self.img_backbone = Backbone(model=cfg.img_backbone, input_dim=3)
		self.geo_backbone = Backbone(model=cfg.geo_backbone, input_dim=self.g_dim)
		self.img_f_dim = {l: self.img_backbone.model.f_dim[l] for l in self.level}
		self.geo_f_dim = {l: self.geo_backbone.model.f_dim[l] for l in self.level}

		self.KV_geo_ref4 = KeyValue(self.geo_f_dim[4], self.key_dim[4], self.geo_f_dim[4], only_key=False)
		self.KV_geo_ref3 = KeyValue(self.geo_f_dim[3], self.key_dim[3], self.geo_f_dim[3], only_key=False)
		self.KV_geo_ref2 = KeyValue(self.geo_f_dim[2], self.key_dim[2], self.geo_f_dim[2], only_key=False)
		self.KV_geo_que4 = KeyValue(self.geo_f_dim[4], self.key_dim[4], self.geo_f_dim[4], only_key=True)
		self.KV_geo_que3 = KeyValue(self.geo_f_dim[3], self.key_dim[3], self.geo_f_dim[3], only_key=True)
		self.KV_geo_que2 = KeyValue(self.geo_f_dim[2], self.key_dim[2], self.geo_f_dim[2], only_key=True)
		self.KV_img_ref4 = KeyValue(self.img_f_dim[4], self.key_dim[4], self.img_f_dim[4], only_key=False)
		self.KV_img_ref3 = KeyValue(self.img_f_dim[3], self.key_dim[3], self.img_f_dim[3], only_key=False)
		self.KV_img_ref2 = KeyValue(self.img_f_dim[2], self.key_dim[2], self.img_f_dim[2], only_key=False)
		self.KV_img_que4 = KeyValue(self.img_f_dim[4], self.key_dim[4], self.img_f_dim[4], only_key=False)
		self.KV_img_que3 = KeyValue(self.img_f_dim[3], self.key_dim[3], self.img_f_dim[3], only_key=False)
		self.KV_img_que2 = KeyValue(self.img_f_dim[2], self.key_dim[2], self.img_f_dim[2], only_key=False)

		self.upsampler = Upsampler(self.level, self.geo_f_dim, self.g_dim)
		self.pose_estimator_1 = PoseEstimator(self.g_dim * 3 + 8,[self.geo_size, self.geo_size])  #  xyz_dim g_dim*2 + error g_dim + mask 2 + img 3*2 
		self.pose_estimator_2 = PoseEstimator(self.g_dim * 3 + 2, [self.geo_size, self.geo_size])  #  xyz_dim g_dim*2 + error g_dim + mask 2 

	def forward(self, structure, ref_K_geo, ref_RT, ref_geo, ref_mask_visib, ref_img, que_K_geo, que_img, que_geo=None, is_student=False, is_train=True):
		geo_aff = None
		img_aff = None

		ref_geo_feat = self.geo_backbone(ref_geo)
		ref_geo_k4, ref_geo_v4 = self.KV_geo_ref4(ref_geo_feat[4])
		ref_geo_k3, ref_geo_v3 = self.KV_geo_ref3(ref_geo_feat[3])
		ref_geo_k2, ref_geo_v2 = self.KV_geo_ref2(ref_geo_feat[2])

		if is_train or not is_student:
			que_geo = F.interpolate(que_geo.flatten(0, 1), ref_geo.shape[-2:]).unflatten(0, (-1, que_geo.shape[1]))
			que_geo_feat = self.geo_backbone(que_geo)
			que_geo_k4, _ = self.KV_geo_que4(que_geo_feat[4])
			que_geo_k3, _ = self.KV_geo_que3(que_geo_feat[3])
			que_geo_k2, _ = self.KV_geo_que2(que_geo_feat[2])
			geo_aff4 = get_affinity(ref_geo_k4, que_geo_k4, self.temperature)
			geo_aff3 = get_affinity(ref_geo_k3, que_geo_k3, self.temperature)
			geo_aff2 = get_affinity(ref_geo_k2, que_geo_k2, self.temperature)
			geo_aff = {4: geo_aff4, 3: geo_aff3, 2: geo_aff2}
			aff = geo_aff

		if is_student:
			ref_img_feat = self.img_backbone(ref_img)
			que_img_feat = self.img_backbone(que_img)
			ref_img_k4, _ = self.KV_img_ref4(ref_img_feat[4])
			ref_img_k3, _ = self.KV_img_ref3(ref_img_feat[3])
			ref_img_k2, _ = self.KV_img_ref2(ref_img_feat[2])
			que_img_k4, _ = self.KV_img_que4(que_img_feat[4])
			que_img_k3, _ = self.KV_img_que3(que_img_feat[3])
			que_img_k2, _ = self.KV_img_que2(que_img_feat[2])
			img_aff4 = get_affinity(ref_img_k4, que_img_k4, self.temperature)
			img_aff3 = get_affinity(ref_img_k3, que_img_k3, self.temperature)
			img_aff2 = get_affinity(ref_img_k2, que_img_k2, self.temperature)
			img_aff = {4: img_aff4, 3: img_aff3, 2: img_aff2}
			aff = {k: v.detach() for k, v in img_aff.items()}
			ref_geo_v4 = ref_geo_v4.detach()
			ref_geo_v3 = ref_geo_v3.detach()
			ref_geo_v2 = ref_geo_v2.detach()

		que_v4 = attention(ref_geo_v4, aff[4])
		que_v3 = attention(ref_geo_v3, aff[3])
		que_v2 = attention(ref_geo_v2, aff[2])
		que_val = {4: que_v4, 3: que_v3, 2: que_v2}

		pred_geo, pred_mask_visib, pred_error = self.upsampler(que_val)
		ref_geo = F.interpolate(ref_geo.flatten(0, 1), pred_geo.shape[-2:]).unflatten(0, (-1, ref_geo.shape[1]))
		ref_mask_visib = F.interpolate(ref_mask_visib.flatten(0, 1), pred_geo.shape[-2:]).unflatten(0, (-1, ref_mask_visib.shape[1]))
		ref_img = F.interpolate(ref_img.flatten(0, 1), pred_geo.shape[-2:]).unflatten(0, (-1, ref_img.shape[1]))
		RT_1_candidate = self.pose_estimator_1(
			que_K_geo, ref_K_geo, ref_RT,
			pred_geo, ref_geo,
			pred_mask_visib, ref_mask_visib,
			pred_error,
			que_img, ref_img)
		RT_1 = get_separate_medoid(RT_1_candidate)
		RT_2 = RT_1.clone()
		if self.additional_step > 0:
			for _ in range(self.additional_step):
				refine_step_geo = render_geometry(structure, que_K_geo, RT_2, self.geo_size, self.n_freqs, self.represent_mode, self.ray_mode)
				refine_step_mask_visib = (refine_step_geo.abs().mean(-3, keepdims=True) + 1).type(torch.bool).type(torch.float)
				RT_2 = self.pose_estimator_2(
					que_K_geo, que_K_geo, RT_2,
					pred_geo, refine_step_geo,
					pred_mask_visib, refine_step_mask_visib,
					pred_error)
		return {'geo': pred_geo, 'mask_visib': pred_mask_visib, 'error': pred_error, 
		  		'RT_1_candidate': RT_1_candidate, 'RT_1': RT_1, 'RT_2': RT_2, 'geo_aff': geo_aff, 'img_aff': img_aff}