import torch
import torch.nn as nn
import torch.nn.functional as F
from HGPose.model.backbone import ResNet18, ResNet34, ResNet50, ConvNextTiny, DINOv2_ViTs14, UpsampleHead, PoseHead, ResNet50_Upsample, UpsampleHeadGeo
from HGPose.utils.geometry import (
	PositionEmbeddingSine3D, get_ref_RT, get_distmap, get_2d_coord, apply_imagespace_relative, render_geometry, R_T_to_RT, interpolate_shapes,
	default_RT, allo_R_to_ego_R, ego_R_to_allo_R, get_separate_medoid, get_medoid, get_best_candidate, get_average)
from HGPose.model.HMMN import QueryKeyValue, get_dual_affinity, get_affinity, find_topk, attention, attention_topk, get_affinity_topk
from collections import OrderedDict, defaultdict

class BackboneQue(nn.Module):
	def __init__(self, model=None, input_dim=3):
		super(BackboneQue, self).__init__()
		if model == 'res34':
			self.model = ResNet34(input_dim)
		elif model == 'res50':
			self.model = ResNet50(input_dim)
		elif model == 'convnext_tiny':
			self.model = ConvNextTiny(input_dim)
		elif model == 'dinov2_Vits14':
			self.model = DINOv2_ViTs14()
		elif model == 'res50_upsample':
			self.model = ResNet50_Upsample(input_dim)

	def forward(self, que_img):
		bsz, n_que = que_img.shape[0:2]
		stack = que_img.flatten(0, 1)
		feature = self.model(stack)             
		que_feat = {l: f.unflatten(0, (bsz, n_que)) for l, f in feature.items()}
		return que_feat


class BackboneSel(nn.Module):
	def __init__(self, model=None, input_dim=3):
		super(BackboneSel, self).__init__()
		if model == 'res34':
			self.model = ResNet34(input_dim)
		elif model == 'res50':
			self.model = ResNet50(input_dim)
		elif model == 'convnext_tiny':
			self.model = ConvNextTiny(input_dim)
		elif model == 'dinov2_Vits14':
			self.model = DINOv2_ViTs14()
		elif model == 'res50_upsample':
			self.model = ResNet50_Upsample(input_dim)

	def forward(self, sel_img):
		bsz, n_sel = sel_img.shape[0:2]
		stack = sel_img.flatten(0, 1)
		feature = self.model(stack)
		sel_feat = {l: f.unflatten(0, (bsz, n_sel)) for l, f in feature.items()}
		return sel_feat
	

class Upsampler(nn.Module):
	def __init__(self, level, f_dim, g_dim):
		super(Upsampler, self).__init__()
		self.level = level
		self.f_dim = f_dim
		self.g_dim = g_dim
		self.model = UpsampleHead(f_dim=self.f_dim, g_dim=self.g_dim)
	def forward(self, que_feat, que_val):
		bsz, n_que, _, _, _ = list(que_val.values())[0].shape
		feat = {l: que_feat[l].flatten(0, 1) for l in self.level}
		val = {l: que_val[l].flatten(0, 1) for l in self.level}
		geo = self.model(feat, val)
		geo = geo.unflatten(0, (bsz, n_que))
		# mask_visibs = mask_visibs.unflatten(0, (bsz, n_que))
		return geo


class UpsamplerGeoOnly(nn.Module):
	def __init__(self, level, g_dim):
		super(UpsamplerGeoOnly, self).__init__()
		self.level = level
		self.g_dim = g_dim
		self.model = UpsampleHeadGeo(g_dim=self.g_dim)
	def forward(self, que_val):
		bsz, n_que, _, _, _ = list(que_val.values())[0].shape
		val = {l: que_val[l].flatten(0, 1) for l in self.level}
		geo = self.model(val)
		geo = geo.unflatten(0, (bsz, n_que))
		return geo


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
				mask_visib_p):
		bsz, n_sel = K_i.shape[0:2]
		xyz = torch.cat([xyz_p.expand(xyz_i.shape), xyz_i], dim=-3).flatten(0, 1)
		mask_visib = mask_visib_p.repeat(1, n_sel, 1, 1, 1).flatten(0, 1)
		feature = torch.cat([xyz, mask_visib], dim=-3)
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
		self.N_sel = cfg.N_sel
		self.N_ref = cfg.N_ref
		# self.topk = cfg.topk
		self.ray_mode = cfg.ray_mode
		self.qk_dim = cfg.qk_dim
		self.represent_mode = cfg.represent_mode
		self.n_freqs = cfg.n_freqs
		self.additional_step = cfg.additional_step
		self.temperature = cfg.temperature
		self.g_dim = 3
		if self.represent_mode == 'positional':
			self.g_dim = 2 * self.g_dim * self.n_freqs
		
		self.que_backbone = BackboneQue(model=cfg.backbone_model, input_dim=3)
		self.sel_backbone = BackboneSel(model=cfg.backbone_model, input_dim=3)
		self.f_dim = {l: self.que_backbone.model.f_dim[l] for l in self.level}
		# from kornia.feature import LoFTR
		# self.loftr = LoFTR('indoor').cuda().eval()
		
		self.K_sel4 = QueryKeyValue(self.f_dim[4], self.qk_dim[4], self.f_dim[4], is_que=False, is_key=True, is_val=False)
		self.K_sel3 = QueryKeyValue(self.f_dim[3], self.qk_dim[3], self.f_dim[3], is_que=False, is_key=True, is_val=False)
		self.K_sel2 = QueryKeyValue(self.f_dim[2], self.qk_dim[2], self.f_dim[2], is_que=False, is_key=True, is_val=False)
		self.Q_que4 = QueryKeyValue(self.f_dim[4], self.qk_dim[4], self.f_dim[4], is_que=True, is_key=False, is_val=False)
		self.Q_que3 = QueryKeyValue(self.f_dim[3], self.qk_dim[3], self.f_dim[3], is_que=True, is_key=False, is_val=False)
		self.Q_que2 = QueryKeyValue(self.f_dim[2], self.qk_dim[2], self.f_dim[2], is_que=True, is_key=False, is_val=False)

		self.upsampler = UpsamplerGeoOnly(self.level, self.g_dim)
		self.pose_estimator_1 = PoseEstimator(self.g_dim * 2 + 1, [self.geo_size, self.geo_size])  #  xyz_dim g_dim*2 + mask 1
		self.pose_estimator_2 = PoseEstimator(self.g_dim * 2 + 1, [self.geo_size, self.geo_size])  #  xyz_dim g_dim*2 + mask 1 

	def forward(self, xyz_mesh, ref_RT, ref_img, que_K_geo, que_img, que_bbox_RT, sel_idx, que_mask_visib, que_geo):
		sel_img = ref_img.gather(1, sel_idx.cuda().view(*sel_idx.shape, 1, 1, 1).repeat(1, 1, *ref_img.shape[2:]))
		sel_RT = get_ref_RT(ref_RT, sel_idx)
		sel_K_geo = que_K_geo.repeat(1, self.N_sel, 1, 1)
		sel_geo = render_geometry(xyz_mesh, sel_K_geo, sel_RT, self.geo_size, self.n_freqs, self.represent_mode, self.ray_mode)

		sel_g = interpolate_shapes(sel_geo, {4: [8, 8], 3: [16, 16], 2: [32, 32]})
		sel_geo_aff = defaultdict()
		sel_geo_aff[4] = get_affinity(sel_g[4], sel_g[4], self.temperature)
		sel_geo_aff[3] = get_affinity(sel_g[3], sel_g[3], self.temperature)
		sel_geo_aff[2] = get_affinity(sel_g[2], sel_g[2], self.temperature)

		sel_feat = self.sel_backbone(sel_img)
		sel_k = defaultdict()
		_, sel_k[4], _ = self.K_sel4(sel_feat[4])
		_, sel_k[3], _ = self.K_sel3(sel_feat[3])
		_, sel_k[2], _ = self.K_sel2(sel_feat[2])

		sel_multiview_k = defaultdict()
		sel_multiview_k[4] = attention(sel_k[4], sel_geo_aff[4])
		sel_multiview_k[3] = attention(sel_k[3], sel_geo_aff[3])
		sel_multiview_k[2] = attention(sel_k[2], sel_geo_aff[2])

		que_feat = self.que_backbone(que_img)
		que_q = defaultdict()
		que_q[4], _, _ = self.Q_que4(que_feat[4])
		que_q[3], _, _ = self.Q_que3(que_feat[3])
		que_q[2], _, _ = self.Q_que2(que_feat[2])

		cross_aff = defaultdict()
		cross_aff[4] = get_affinity(sel_multiview_k[4], que_q[4], self.temperature)
		cross_aff[3] = get_affinity(sel_multiview_k[3], que_q[3], self.temperature)
		cross_aff[2] = get_affinity(sel_multiview_k[2], que_q[2], self.temperature)

		que_v = defaultdict()
		que_v[4] = attention(sel_g[4], cross_aff[4])
		que_v[3] = attention(sel_g[3], cross_aff[3])
		que_v[2] = attention(sel_g[2], cross_aff[2])

		pred_geo = self.upsampler(que_v)
		pred_mask_visib = que_mask_visib.clone()
		RT_1_candidate = self.pose_estimator_1(que_K_geo, sel_K_geo, sel_RT, pred_geo, sel_geo,	pred_mask_visib)
		RT_1 = get_separate_medoid(RT_1_candidate)
		RT_2 = RT_1.clone()
		for _ in range(self.additional_step):
			refine_geo = render_geometry(xyz_mesh, que_K_geo, RT_2, self.geo_size, self.n_freqs, self.represent_mode, self.ray_mode)
			RT_2 = self.pose_estimator_2(que_K_geo, que_K_geo, RT_2, pred_geo, refine_geo, pred_mask_visib)
		pred = {'que_v': que_v,
		  		'que_q': que_q,
				'geo': pred_geo, 
				'mask_visib': pred_mask_visib, 
				'RT_1_candidate': RT_1_candidate, 
				'RT_1': RT_1, 'RT_2': RT_2, 'cross_aff': cross_aff}
		sel = {'image': sel_img, 
		 	   'geo': sel_geo, 
			   'RT': sel_RT, 
			   'multiview_k': sel_multiview_k}
		return pred, sel
			