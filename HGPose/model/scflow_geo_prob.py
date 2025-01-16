import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
from HGPose.model.convnext import convnext_tiny
from collections import OrderedDict
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights, RAFT
from torchvision.models.optical_flow.raft import ResidualBlock, RecurrentBlock, CorrBlock, FeatureEncoder, MotionEncoder, FlowHead, UpdateBlock, MaskPredictor
from torchvision.models.optical_flow._utils import make_coords_grid, upsample_flow
from torch.nn.modules.instancenorm import InstanceNorm2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torchvision.ops import Conv2dNormActivation
from torch.nn.functional import grid_sample
from HGPose.model.deformable_conv import DeformableConv2d, DeformableConv2dGuided
from HGPose.utils.geometry import (apply_forward_flow, apply_backward_flow, get_flow_from_delta_pose_and_xyz, get_flow_from_delta_pose_and_xyz_scflow,
	get_2d_coord, apply_imagespace_relative, render_geometry_scflow, get_separate_medoid, get_region_smoothness_and_variation, get_pose_from_delta_pose_scflow)
from HGPose.model.mfn import GaborNet
from HGPose.model.geo_prob_model import GeoProbModel
from HGPose.utils.flow import filter_flow_by_mask
from einops import rearrange, reduce, repeat
from torch.distributions import Normal
import math

class SCFlow_geo_prob(nn.Module):
	def __init__(self, cfg):
		super(SCFlow_geo_prob, self).__init__()
		feature_encoder_layers=(64, 64, 96, 128, 256)
		feature_encoder_block=ResidualBlock
		feature_encoder_norm_layer=InstanceNorm2d
		# Context encoder
		context_encoder_layers=(64, 64, 96, 128, 256)
		context_encoder_block=ResidualBlock
		context_encoder_norm_layer=BatchNorm2d
		# Correlation block
		corr_block_num_levels=4
		corr_block_radius=4
		# Motion encoder
		motion_encoder_corr_layers=(256, 192)
		motion_encoder_flow_layers=(128, 64)
		motion_encoder_out_channels=128
		# Recurrent block
		recurrent_block_hidden_state_size=128
		recurrent_block_kernel_size=((1, 5), (5, 1))
		recurrent_block_padding=((0, 2), (2, 0))
		# Flow Head
		flow_head_hidden_size=256
		self.size = [cfg.image_size, cfg.image_size]
		
		# self.num_region = cfg.N_region

		#self.clip_mixer = ClipMixer(128, 256)
		self.feature_encoder = FeatureEncoder(
			block=feature_encoder_block, layers=feature_encoder_layers, norm_layer=feature_encoder_norm_layer
		)
		self.context_encoder = FeatureEncoder(
			block=context_encoder_block, layers=context_encoder_layers, norm_layer=context_encoder_norm_layer
		)
		# self.class_encoder = GaborNet(in_size=1,
		# 						hidden_size=256,
		# 						n_layers=2,
		# 						alpha=4.5,
		# 						out_size=128)

		self.corr_block = CorrBlock(num_levels=corr_block_num_levels, radius=corr_block_radius)

		self.motion_encoder = MotionEncoder(
			in_channels_corr=self.corr_block.out_channels,
			corr_layers=motion_encoder_corr_layers,
			flow_layers=motion_encoder_flow_layers,
			out_channels=motion_encoder_out_channels,
		)

		# See comments in forward pass of RAFT class about why we split the output of the context encoder
		out_channels_context = context_encoder_layers[-1] - recurrent_block_hidden_state_size
		self.recurrent_block = RecurrentBlock(
			input_size=self.motion_encoder.out_channels + out_channels_context,
			hidden_size=recurrent_block_hidden_state_size,
			kernel_size=recurrent_block_kernel_size,
			padding=recurrent_block_padding,
		)
		self.flow_head = FlowHead(in_channels=recurrent_block_hidden_state_size, hidden_size=flow_head_hidden_size)
		self.mask_head = MaskHead_custom_scflow(in_channels=recurrent_block_hidden_state_size, hidden_size=flow_head_hidden_size)
		self.update_block = UpdateBlock_custom(motion_encoder=self.motion_encoder, recurrent_block=self.recurrent_block, flow_head=self.flow_head, mask_head=self.mask_head)
		
		# self.motion_encoder_refine = MotionEncoderAdaptiveRefine(motion_encoder_out_channels)
		self.recurrent_block_refine = RecurrentBlock(
			input_size=self.motion_encoder.out_channels + out_channels_context,
			hidden_size=recurrent_block_hidden_state_size,
			kernel_size=recurrent_block_kernel_size,
			padding=recurrent_block_padding,
		)
		self.flow_head_refine = FlowHead(in_channels=recurrent_block_hidden_state_size, hidden_size=flow_head_hidden_size)
		self.mask_head_refine = MaskHead_custom_scflow(in_channels=recurrent_block_hidden_state_size, hidden_size=flow_head_hidden_size)
		self.update_block_refine = UpdateBlock_refine(motion_encoder=self.motion_encoder, recurrent_block_refine=self.recurrent_block_refine, flow_head_refine=self.flow_head_refine, mask_head_refine=self.mask_head_refine)
		
		
		
		self.num_class = cfg.num_class
		self.pose_head = SCFlowPoseHead_scflow(self.num_class)
		self.register_buffer('fix', torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0]).to(torch.float).view(1, 9))
		
		####################custom_geo######################
		self.image_size = cfg.image_size
		self.geo_size = cfg.geo_size
		self.geo_model = GeoProbModel(cfg)
		####custom for flow prob####
		self.flow_var1_minus_plus = torch.as_tensor(cfg.flow_var1_minus_plus).float()
		self.flow_var2_minus = torch.as_tensor(cfg.flow_var2_minus).float()
		self.flow_var2_plus = torch.as_tensor(cfg.flow_var2_plus).float()
		self.flow_R = cfg.flow_R
		
		self.uncertainty_module = nn.Sequential(
			nn.Conv2d(4, 32, 3, 1, 0),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 1, 0),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 16, 3, 1, 0),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16, 16, 3, 1, 0),
			# nn.BatchNorm2d(16),
			# nn.ReLU()
		)
		self.uncertainty_predictor = nn.Sequential(
			nn.Conv2d(16+2, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 16, 3, 1, 1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16, 3, 3, 1, 1), #3
			#nn.ReLU() #no relu
		)


	def forward(self, image_ref, image_que, K_ref, K_que, RT_ref, geo_que, geo_ref, geo_ref_origin, mask_ref, gt_flow, xyz_mesh, obj_id,  num_flow_updates: int=4):
		batch_size, _, h, w = image_ref.shape
		fmaps = self.feature_encoder(torch.cat([image_ref, image_que], dim=0))
		fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)
		self.corr_block.build_pyramid(fmap1, fmap2)

		context_out = self.context_encoder(image_ref)
		
		hidden_state_size = self.update_block.hidden_state_size
		out_channels_context = context_out.shape[1] - hidden_state_size
		hidden_state, context = torch.split(context_out, [hidden_state_size, out_channels_context], dim=1)
		hidden_state = torch.tanh(hidden_state)
		context = F.relu(context)

		coords0 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)
		# coords1 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)
		init_flow = fmap1.new_zeros((batch_size, 2, h, w), dtype=torch.float32, device=fmap1.device)

		flow_predictions = []
		pose_predictions = []
		sc_flow_predictions = []
		mask_predictions = []
		flow_confidence_predictions = []
		flow_uncertainty_predictions = []
		mask_refined_predictions = []
		flow_refined_predictions = []

		# RT_update = RT_ref
		RT_ref_scaled = apply_imagespace_relative(K_ref, K_que, RT_ref, self.fix.expand(batch_size, 9), self.size)
		RT_que = RT_ref_scaled.clone()

		
		downsampled_gt_flow = F.interpolate(gt_flow, scale_factor=1/8, mode="bilinear") / 8
		
		####################custom_geo######################
		que_coord_pred, que_mask_pred, que_mask_visib_pred = self.geo_model(image_que.unsqueeze(1), obj_id)
		
		#gt coord
		# que_coord_pred = geo_que
		current_coord = geo_ref
		current_coords = []
		flow = init_flow
		for _ in range(num_flow_updates):
			# coords1 = coords1.detach()  # Don't backpropagate gradients through this branch, see paper
			flow = F.interpolate(flow, scale_factor=1/8, mode='bilinear') / 8
			coords1 = (coords0 + flow).detach()
			corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)
			# pred_flow = coords1 - coords0
			hidden_state, delta_flow, pred_mask = self.update_block(hidden_state, context, corr_features, flow)
			
			# # gt_delta_flow
			# delta_gt_flow = (downsampled_gt_flow - flow)
			# delta_flow = delta_gt_flow.detach().clone()
			
			#
			# coords1 = coords1 + delta_flow
			# pred_flow = coords1 - coords0
			pred_flow = flow + delta_flow
			#pred_flow = gt_flow.clone()
			pred_up_flow = upsample_flow(flow=pred_flow, factor=8)
			pred_up_mask = F.interpolate(pred_mask, scale_factor=8, mode='bilinear')
			mask_predictions.append(pred_up_mask)
			flow_predictions.append(pred_up_flow)
			
			####flow prob
			corr_rearrange = rearrange(corr_features, 'b (c2 h2 w2) h w -> (b h w) c2 h2 w2', c2=4, h2=9, w2=9)
			corr_uncertainty = self.uncertainty_module(corr_rearrange)
			corr_uncertainty = rearrange(corr_uncertainty, '(b h w) c 1 1 -> b c h w', b=batch_size, h=32, w=32)
			flow_uncertainty = self.uncertainty_predictor(torch.cat([corr_uncertainty, pred_flow], 1))
			flow_large_log_var_map = flow_uncertainty[:,0:1,:,:]
			flow_large_log_var_map = torch.log(self.flow_var2_minus + (self.flow_var2_plus - self.flow_var2_minus) * torch.sigmoid(flow_large_log_var_map - torch.log(self.flow_var2_plus)))
			flow_small_log_var_map = torch.ones_like(flow_large_log_var_map, requires_grad=False) * torch.log(self.flow_var1_minus_plus)
			flow_log_var_map = torch.cat([flow_small_log_var_map, flow_large_log_var_map], axis=1)
			flow_weight_map = flow_uncertainty[:,1:,:,:]
			flow_weight_map = torch.nn.functional.softmax(flow_weight_map, dim=1)
			flow_uncertainty_pred = torch.cat([flow_log_var_map,flow_weight_map], axis=1)

			flow_confidence_map = torch.sum(flow_weight_map * (1 - torch.exp(- math.sqrt(2)*self.flow_R/torch.sqrt(torch.exp(flow_log_var_map))))**2, dim=1, keepdim=True)
			#confidence_mask = (confidence_map > 0.5).float()
			# pred_up_flow_confidence = F.interpolate(flow_confidence_map, scale_factor=8, mode='bilinear')
			pred_up_flow_uncertainty = F.interpolate(flow_uncertainty_pred, scale_factor=8, mode='bilinear')
			flow_uncertainty_predictions.append(pred_up_flow_uncertainty)
			flow_confidence_predictions.append(flow_confidence_map)

			####delta flow refine
			pred_geo = torch.cat([current_coord, que_coord_pred, que_mask_pred, que_mask_visib_pred], 1)
			coords1 = (coords0 + pred_flow).detach()
			corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)
			hidden_state, delta_flow_refined, pred_mask_refined = self.update_block_refine(hidden_state, context, corr_features, pred_flow, flow_confidence_map, pred_geo)
			delta_flow_refined = delta_flow + delta_flow_refined
			pred_flow_refined = flow + delta_flow_refined	
			pred_up_flow_refined = upsample_flow(flow=pred_flow_refined, factor=8)
			pred_up_mask_refined = F.interpolate(pred_mask_refined, scale_factor=8, mode='bilinear')
			mask_refined_predictions.append(pred_up_mask_refined)
			flow_refined_predictions.append(pred_up_flow_refined)		


			####pose estimate

			delta_rotation, delta_translation = self.pose_head(hidden_state, delta_flow_refined, pred_mask_refined, obj_id)  

			delta_relative = torch.cat([delta_rotation, delta_translation], -1)
			RT_que = apply_imagespace_relative(K_que, K_que, RT_que.detach(), delta_relative, self.size)
			pose_predictions.append(RT_que)

			shape_constraint_flow = get_flow_from_delta_pose_and_xyz_scflow(
				RT_que.unsqueeze(1).detach(), K_que.unsqueeze(1), geo_ref_origin.unsqueeze(1)).squeeze(1)
			sc_flow_predictions.append(shape_constraint_flow)
			flow = shape_constraint_flow

			####################custom_geo######################
			current_coords.append(current_coord)
			# current_coord = render_geometry_scflow(xyz_mesh, K_que.unsqueeze(1), RT_que.unsqueeze(1).detach(), self.image_size, self.geo_size, represent_mode='xyz').squeeze(1)

			


		return flow_predictions, pose_predictions, sc_flow_predictions, mask_predictions, flow_uncertainty_predictions, flow_confidence_predictions, current_coords, que_coord_pred, que_mask_pred, que_mask_visib_pred, flow_refined_predictions, mask_refined_predictions

class SCFlowPoseHead_scflow(nn.Module):
	def __init__(self, num_class):
		super(SCFlowPoseHead_scflow, self).__init__()
		self.num_class = num_class
		self.delta_flow_model = nn.Sequential(
			nn.Conv2d(2, 128, 7, 1, 3), # 2+1
			nn.ReLU(),
			nn.Conv2d(128, 64, 3, 1, 1),
			nn.ReLU())
		self.mask_model = nn.Sequential(
			nn.Conv2d(1, 64, 3, 1, 1),
			nn.ReLU(),
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.ReLU())
		self.confidence_model = nn.Sequential(
			nn.Conv2d(1, 64, 3, 1, 1),
			nn.ReLU(),
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.ReLU())
		self.conv_layers = nn.Sequential(
			nn.Conv2d(224, 128, 3, 2, 1),    #256 or 224
			nn.GroupNorm(32, 128, eps=1e-05, affine=True),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 2, 1),
			nn.GroupNorm(32, 128, eps=1e-05, affine=True),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 2, 1),
			nn.GroupNorm(32, 128, eps=1e-05, affine=True),
			nn.ReLU()
		)
		self.flatten_op = nn.Flatten(start_dim=1, end_dim=-1)
		self.fc_layers = nn.Sequential(
			nn.Linear(2048, 1024, bias=True),
			nn.ReLU(),
			nn.Linear(1024, 256, bias=True),
			nn.ReLU()
		)
		self.rotation_pred = nn.Linear(256, 6*num_class)
		self.translation_pred = nn.Linear(256, out_features=3*num_class)
		self.init_weights()

	def init_weights(self):
		 # zero translation
		nn.init.zeros_(self.translation_pred.weight)
		nn.init.zeros_(self.translation_pred.bias)
		nn.init.zeros_(self.rotation_pred.weight)
		# identity quarention
		with torch.no_grad():
			self.rotation_pred.bias.copy_(torch.Tensor([1., 0., 0., 0., 1., 0.]*self.num_class))

	def forward(self, hidden_state, delta_flow, mask, obj_id):
		#hidden_encode = self.hidden_state_model(hidden_state)
		delta_flow_encode = self.delta_flow_model(delta_flow)
		mask_encode = self.mask_model(mask)
		# confidence_encode = self.confidence_model(confidence)

		# encode = torch.cat([hidden_state, delta_flow_encode, mask_encode], dim=-3)
		encode = torch.cat([hidden_state, delta_flow_encode, mask_encode], dim=-3)
		encode = self.conv_layers(encode)
		encode = self.flatten_op(encode)
		encode = self.fc_layers(encode)
		pred_translation_delta = self.translation_pred(encode)
		pred_rotation_delta = self.rotation_pred(encode)
		pred_translation_delta = pred_translation_delta.view(-1, self.num_class, 3)
		pred_rotation_delta = pred_rotation_delta.view(-1, self.num_class, 6)
		pred_translation_delta = torch.index_select(pred_translation_delta, dim=1, index=obj_id-1)[:, 0, :]
		pred_rotation_delta = torch.index_select(pred_rotation_delta, dim=1, index=obj_id-1)[:, 0, :]

		return pred_rotation_delta, pred_translation_delta


# class SCFlowPoseHead_scflow_geo(nn.Module):
# 	def __init__(self, num_class):
# 		super(SCFlowPoseHead_scflow_geo, self).__init__()
# 		self.num_class = num_class
# 		self.geo_concat_model = nn.Sequential(
# 			nn.Conv2d(3+1+1+1+3, 128, 3, 1, 1),
# 			nn.BatchNorm2d(128),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 64, 3, 2, 1),
# 			nn.BatchNorm2d(64),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.BatchNorm2d(64),
# 			nn.ReLU()
# 		)
# 		self.flow_concat_model = nn.Sequential(
# 			nn.Conv2d(2+1+1, 128, 7, 1, 3), # 2+1
# 			nn.BatchNorm2d(128),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 64, 3, 1, 1),
# 			nn.BatchNorm2d(64),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.BatchNorm2d(64),
# 			nn.ReLU()
# 			)
# 		# self.mask_model = nn.Sequential(
# 		# 	nn.Conv2d(1, 64, 3, 1, 1),
# 		# 	nn.ReLU(),
# 		# 	nn.Conv2d(64, 32, 3, 1, 1),
# 		# 	nn.ReLU())
# 		# self.confidence_model = nn.Sequential(
# 		# 	nn.Conv2d(1, 64, 3, 1, 1),
# 		# 	nn.ReLU(),
# 		# 	nn.Conv2d(64, 32, 3, 1, 1),
# 		# 	nn.ReLU())
		
# 		self.pooling = nn.AdaptiveAvgPool2d((1,1))
# 		self.agg_model = nn.Sequential(
# 			nn.Conv2d(64, 128, 3, 1, 1),
# 			nn.BatchNorm2d(128),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 128, 1, 1, 0),
# 			nn.BatchNorm2d(128),
# 			nn.ReLU())
		
# 		self.conv_layers_concat = nn.Sequential(
# 			nn.Conv2d(128+128, 256, 3, 2, 1),
# 			nn.GroupNorm(32, 256, eps=1e-05, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(256, 128, 3, 2, 1),
# 			nn.GroupNorm(32, 128, eps=1e-05, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 128, 3, 2, 1),
# 			nn.GroupNorm(32, 128, eps=1e-05, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 128, 3, 1, 1),
# 			nn.GroupNorm(32, 128, eps=1e-05, affine=True),
# 			nn.ReLU()
# 		)
# 		self.conv_layers_agg = nn.Sequential(
# 			nn.Conv2d(128+128+32, 128, 3, 2, 1),
# 			nn.GroupNorm(32, 128, eps=1e-05, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 128, 3, 2, 1),
# 			nn.GroupNorm(32, 128, eps=1e-05, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 128, 3, 2, 1),
# 			nn.GroupNorm(32, 128, eps=1e-05, affine=True),
# 			nn.ReLU()
# 		)
# 		self.flatten_op = nn.Flatten(start_dim=1, end_dim=-1)
# 		self.fc_layers = nn.Sequential(
# 			nn.Linear(2048, 1024, bias=True),
# 			nn.ReLU(),
# 			nn.Linear(1024, 256, bias=True),
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

# 	def forward(self, hidden_state, flow_concatenated, geo_concatenated, obj_id):
# 		#hidden_encode = self.hidden_state_model(hidden_state)
# 		geo_encode = self.geo_concat_model(geo_concatenated) #[b, 64, 32, 32]
# 		flow_encode = self.flow_concat_model(flow_concatenated) #[b, 64, 32, 32]

# 		geo_conf_masked = geo_concatenated[:,3:4,:,:] * (geo_concatenated[:,5:6,:,:] > 0.5)
# 		geo_conf_pooled = self.pooling(geo_conf_masked)
# 		flow_conf_masked = flow_concatenated[:,2:3,:,:] * (flow_concatenated[:,3:4,:,:] > 0.5)
# 		flow_conf_pooled = self.pooling(flow_conf_masked)
# 		pooled_concat = torch.concat([geo_conf_pooled, flow_conf_pooled], dim=1)
# 		softmax_pooled_concat = F.softmax(pooled_concat, dim=-3).detach()
# 		agg = softmax_pooled_concat[:,0:1,:,:] * geo_encode + softmax_pooled_concat[:,1:2,:,:] * flow_encode #[b, 64, 32, 32]
# 		agg_encode = self.agg_model(agg) #[b, 128, 32, 32]
# 		encode = torch.cat([hidden_state, agg_encode], dim=-3)


# 		# weight = self.pooling(confidence_map)
# 		# weight = torch.sigmoid(weight)
# 		# agg = geo_encode * (1-weight) + delta_flow_encode * weight
# 		# agg = geo_encode * (1 - confidence_map) + delta_flow_encode * confidence_map 
# 		# agg_encode = self.agg_model(agg)
# 		# encode = torch.cat([hidden_state, agg_encode, mask_encode], dim=-3)
# 		# encode = self.conv_layers_agg(encode)

# 		# #confidence just concated
# 		# confidence_encode = self.confidence_model(confidence_map)
# 		# encode = torch.cat([hidden_state, delta_flow_encode, mask_encode, geo_encode, confidence_encode], dim=-3)
# 		encode = self.conv_layers_concat(encode)
# 		encode = self.flatten_op(encode)
# 		encode = self.fc_layers(encode)
# 		pred_translation_delta = self.translation_pred(encode)
# 		pred_rotation_delta = self.rotation_pred(encode)
# 		pred_translation_delta = pred_translation_delta.view(-1, self.num_class, 3)
# 		pred_rotation_delta = pred_rotation_delta.view(-1, self.num_class, 6)
# 		pred_translation_delta = torch.index_select(pred_translation_delta, dim=1, index=obj_id-1)[:, 0, :]
# 		pred_rotation_delta = torch.index_select(pred_rotation_delta, dim=1, index=obj_id-1)[:, 0, :]

# 		# encode = encode.flatten(1, 3)
# 		# rotation = self.rotat_fc(encode)
# 		# translation = self.trans_fc(encode)
# 		#
# 		#rotation = torch.index_select(rotation, dim=1, index=obj_id-1)
# 		#translation = torch.index_select(translation, dim=1, index=obj_id-1)
# 		# rotation = rotation.view(-1, self.num_class, 6)
# 		# translation = translation.view(-1, self.num_class, 3)
		
# 		# result = torch.cat([rotation, translation], -1)
# 		#
# 		return pred_rotation_delta, pred_translation_delta


class UpdateBlock_custom(nn.Module):
	"""The update block which contains the motion encoder, the recurrent block, and the flow head.

	It must expose a ``hidden_state_size`` attribute which is the hidden state size of its recurrent block.
	"""

	def __init__(self, *, motion_encoder, recurrent_block, flow_head, mask_head):
		super().__init__()
		self.motion_encoder = motion_encoder
		self.recurrent_block = recurrent_block
		self.flow_head = flow_head
		self.mask_head = mask_head
		self.hidden_state_size = recurrent_block.hidden_size

	def forward(self, hidden_state, context, corr_features, flow):
		motion_features = self.motion_encoder(flow, corr_features)
		x = torch.cat([context, motion_features], dim=1)

		hidden_state = self.recurrent_block(hidden_state, x)
		delta_flow = self.flow_head(hidden_state)
		mask = self.mask_head(hidden_state)
		return hidden_state, delta_flow, mask
	
class UpdateBlock_refine(nn.Module):
	def __init__(self, *, motion_encoder, recurrent_block_refine, flow_head_refine, mask_head_refine):
		super().__init__()
		self.motion_encoder = motion_encoder
		self.recurrent_block_refine = recurrent_block_refine
		self.flow_head_refine = flow_head_refine
		self.mask_head_refine = mask_head_refine
		self.hidden_state_size = recurrent_block_refine.hidden_size
		self.geo_encoder = nn.Sequential(
			nn.Conv2d(8, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, self.hidden_state_size-2, 3, 1, 1)

		)
		self.conf_encoder = nn.Sequential(
			nn.Conv2d(1, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, self.hidden_state_size-2, 3, 1, 1),
			nn.Sigmoid()
		)
		self.motion_encoder_refine = nn.Sequential(
			nn.Conv2d(self.hidden_state_size-2, self.hidden_state_size-2, 1, 1, 0)
		)

	def forward(self, hidden_state, context, corr_features, pred_flow, flow_confidence, geo):
		motion_features = self.motion_encoder(pred_flow, corr_features)
		motion_features = motion_features[:,:-2,:,:]
		flow_orig = pred_flow
		geo_features = self.geo_encoder(geo)
		conf_features = self.conf_encoder(flow_confidence)
		motion_features_refine = (motion_features * conf_features) + (geo_features * (1 - conf_features))
		motion_features_refine = self.motion_encoder_refine(motion_features_refine)
		motion_features_refine = torch.cat([motion_features_refine, flow_orig], dim=1)
		x = torch.cat([context, motion_features_refine], dim=1)

		hidden_state = self.recurrent_block_refine(hidden_state, x)
		delta_flow_refined = self.flow_head_refine(hidden_state)
		mask_refined = self.mask_head_refine(hidden_state)
		return hidden_state, delta_flow_refined, mask_refined
	
# class MotionEncoderRefine(nn.Module):
# 	def __init__(self, motion_encoder_out_channels):
# 		super(MotionEncoderRefine, self).__init__()
# 		# 입력 채널의 수를 조정하기 위한 첫 번째 컨볼루션 레이어
# 		self.out_channels = motion_encoder_out_channels
# 		self.conv1 = nn.Conv2d(in_channels=11, out_channels=64, kernel_size=3, padding=1) #inchannels: (2+1+1)+(3+3+2), outchannels: 128
# 		self.conv2 = nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=3, padding=1)
		
# 		# 배치 정규화와 ReLU 활성화 함수
# 		self.batch_norm1 = nn.BatchNorm2d(64)
# 		self.batch_norm2 = nn.BatchNorm2d(self.out_channels)
# 		self.relu = nn.ReLU(inplace=True)

# 	def forward(self, flow, confidence, mask, geo):
# 		# 입력들을 하나의 텐서로 연결하기 전에 차원 수를 맞춰줍니다.
# 		# flow: [B, 2, 32, 32], confidence: [B, 1, 32, 32], mask: [B, 1, 32, 32], geometric_feature: [B, 8, 32, 32]
# 		x = torch.cat([flow, mask, geo], dim=1)  # 결과적으로 [B, 12, 32, 32]

# 		# 연결된 데이터를 컨볼루션 레이어를 통과시킵니다.
# 		x = self.conv1(x)
# 		x = self.batch_norm1(x)
# 		x = self.relu(x)

# 		x = self.conv2(x)
# 		x = self.batch_norm2(x)

# 		# 최종 출력은 [B, 128, 32, 32]의 크기를 가집니다.
# 		return x
	
# class MotionEncoderAdaptiveRefine(nn.Module):
# 	def __init__(self, motion_encoder_out_channels):
# 		super(MotionEncoderAdaptiveRefine, self).__init__()
# 		# 입력 채널의 수를 조정하기 위한 첫 번째 컨볼루션 레이어
# 		self.out_channels = motion_encoder_out_channels
# 		self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1) #inchannels: (2+1+1)+(3+3+2), outchannels: 128
# 		self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=3, padding=1)

# 		self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, padding=1) #inchannels: (2+1+1)+(3+3+2), outchannels: 128
# 		self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=3, padding=1)

# 		self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=1, padding=0) #inchannels: (2+1+1)+(3+3+2), outchannels: 128

# 		self.conv4 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, padding=0)
# 		# 배치 정규화와 ReLU 활성화 함수
# 		self.batch_norm1 = nn.BatchNorm2d(64)
# 		self.batch_norm2 = nn.BatchNorm2d(self.out_channels)
# 		self.relu = nn.ReLU(inplace=True)
# 		self.sigmoid = nn.Sigmoid()

# 	def forward(self, flow, confidence, mask, geo):
# 		# 입력들을 하나의 텐서로 연결하기 전에 차원 수를 맞춰줍니다.
# 		# flow: [B, 2, 32, 32], confidence: [B, 1, 32, 32], mask: [B, 1, 32, 32], geometric_feature: [B, 8, 32, 32]
# 		flow = torch.cat([flow, mask], dim=1) #[B, 3, 32, 32]
# 		# 연결된 데이터를 컨볼루션 레이어를 통과시킵니다.
# 		x1 = self.conv1_1(flow)
# 		x1 = self.batch_norm1(x1)
# 		x1 = self.relu(x1)

# 		x1 = self.conv1_2(x1)
# 		x1 = self.batch_norm2(x1)

# 		x2 = self.conv2_1(geo)
# 		x2 = self.batch_norm1(x2)
# 		x2 = self.relu(x2)

# 		x2 = self.conv2_2(x2)
# 		x2 = self.batch_norm2(x2)

# 		w = self.conv3(confidence)
# 		w = self.sigmoid(w)

# 		x = x1 * w + x2 * (1-w)
# 		x = self.conv4(x)
# 		x = self.batch_norm2(x)

# 		# 최종 출력은 [B, 128, 32, 32]의 크기를 가집니다.
# 		return x
	
class FlowHead_custom(nn.Module):
	"""Flow head, part of the update block.

	Takes the hidden state of the recurrent unit as input, and outputs the predicted "delta flow".
	"""
	def __init__(self, *, in_channels, hidden_size):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
		self.conv2 = nn.Conv2d(hidden_size, 2, 3, padding=1)
		self.relu = nn.ReLU(inplace=True)
	def forward(self, x):
		x = self.conv2(self.relu(self.conv1(x)))
		return x

class MaskHead_custom(nn.Module):
	"""Flow head, part of the update block.

	Takes the hidden state of the recurrent unit as input, and outputs the predicted "delta flow".
	"""
	def __init__(self, *, in_channels, hidden_size):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1) #padding, kernel_size
		self.conv2 = nn.Conv2d(hidden_size, 1, 3, padding=1)
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()
	def forward(self, x):
		x = self.sigmoid(self.conv2(self.relu(self.conv1(x))))
		return x
	
class MaskHead_custom_scflow(nn.Module):
	"""Flow head, part of the update block.

	Takes the hidden state of the recurrent unit as input, and outputs the predicted "delta flow".
	"""
	def __init__(self, *, in_channels, hidden_size):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1) #padding, kernel_size
		self.conv2 = nn.Conv2d(hidden_size, 1, 1) #1,3,padding=1
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()
	def forward(self, x):
		x = self.sigmoid(self.conv2(self.relu(self.conv1(x))))
		return x
	
# class ClipMixer(nn.Module):

# 	def __init__(self, clip_out_channels=128, out_channels=256):
# 		super().__init__()
# 		self.linear = nn.Linear(512, clip_out_channels)
# 		self.conv1x1 = nn.Conv2d(in_channels=clip_out_channels + out_channels, out_channels=out_channels, kernel_size=1)

# 	def forward(self, cnn_feature, clip_feature):
# 		clip_feature = self.linear(clip_feature)
# 		upsampled_clip = clip_feature.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,32,32)
# 		combined_feature = torch.cat([cnn_feature, upsampled_clip], dim=1) #?
# 		output_feature = self.conv1x1(combined_feature)

# 		return output_feature
