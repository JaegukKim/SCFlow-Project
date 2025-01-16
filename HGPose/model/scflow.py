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
from HGPose.utils.geometry import (apply_forward_flow, apply_backward_flow, get_flow_from_delta_pose_and_xyz,
	get_2d_coord, apply_imagespace_relative, render_geometry, get_separate_medoid)

class SCFlow(nn.Module):
	def __init__(self, size):
		super(SCFlow, self).__init__()
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
		self.size = [size, size]


		self.feature_encoder = FeatureEncoder(
			block=feature_encoder_block, layers=feature_encoder_layers, norm_layer=feature_encoder_norm_layer
		)
		self.context_encoder = FeatureEncoder(
			block=context_encoder_block, layers=context_encoder_layers, norm_layer=context_encoder_norm_layer
		)

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

		self.update_block = UpdateBlock(motion_encoder=self.motion_encoder, recurrent_block=self.recurrent_block, flow_head=self.flow_head)

		self.mask_predictor = MaskPredictor(
			in_channels=recurrent_block_hidden_state_size,
			hidden_size=256,
			multiplier=0.25,  # See comment in MaskPredictor about this
		)

		self.pose_head = SCFlowPoseHead(recurrent_block_hidden_state_size)
		self.register_buffer('fix', torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0]).to(torch.float).view(1, 9))

	def forward(self, image_ref, image_que, K_ref, K_que, RT_ref, geo_ref, mask_ref, num_flow_updates: int=4):
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
		coords1 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)
		coords0.requires_grad_()
		coords0.requires_grad_(False)

		flow_predictions = []
		pose_predictions = []
		sc_flow_predictions = []

		RT_ref_scaled = apply_imagespace_relative(K_ref, K_que, RT_ref, self.fix.expand(batch_size, 9), self.size)
		RT_que = RT_ref_scaled.clone()

		flow = coords1.detach() - coords0
		#requires_grad
		up_flow = upsample_flow(flow=flow, factor=8)
		#geo = apply_forward_flow(up_flow.unsqueeze(0), geo_ref.unsqueeze(0), -1.0)[0]
		mask = apply_forward_flow(up_flow.unsqueeze(0), mask_ref.unsqueeze(0), 0.0)[0]
		#geo = F.interpolate(geo, scale_factor=1/8)
		mask = F.interpolate(mask, scale_factor=1/8)

		for _ in range(num_flow_updates):
			coords1 = coords1.detach()  # Don't backpropagate gradients through this branch, see paper
			corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)
			hidden_state, delta_flow = self.update_block(hidden_state, context, corr_features, flow)
			# up_mask = self.mask_predictor(hidden_state)
			coords1 = coords1 + delta_flow
			pred_flow = coords1 - coords0
			pred_up_flow = upsample_flow(flow=pred_flow, factor=8)

			coords1 = coords1 - delta_flow
			delta_relative = self.pose_head(hidden_state, delta_flow)
			RT_que = apply_imagespace_relative(K_que, K_que, RT_que, delta_relative, self.size)
			# shape_constraint_delta_flow = get_flow_from_delta_pose_and_xyz(
			# 	RT_que.unsqueeze(0).detach(), K_que.unsqueeze(0), geo_ref.unsqueeze(0), mask_ref.unsqueeze(0))[0]
			shape_constraint_delta_flow = get_flow_from_delta_pose_and_xyz(
				RT_que.unsqueeze(0).detach(), K_que.unsqueeze(0), geo_ref.unsqueeze(0))[0]
			shape_constraint_delta_flow = F.interpolate(shape_constraint_delta_flow, scale_factor=1/8) / 8
			coords1 = coords1 + shape_constraint_delta_flow
			
			shape_constraint_flow = coords1 - coords0
			shape_constraint_up_flow = upsample_flow(flow=shape_constraint_flow, factor=8)
			# geo = apply_forward_flow(up_flow.unsqueeze(0), geo_ref.unsqueeze(0), -1.0)[0]
			# mask = apply_forward_flow(up_flow.unsqueeze(0), mask_ref.unsqueeze(0), 0.0)[0]
			# geo = F.interpolate(geo, scale_factor=1/8)
			# mask = F.interpolate(mask, scale_factor=1/8)

			flow_predictions.append(pred_up_flow)
			pose_predictions.append(RT_que)
			sc_flow_predictions.append(shape_constraint_up_flow)
		return flow_predictions, pose_predictions, sc_flow_predictions
	

class SCFlowPoseHead(nn.Module):
	def __init__(self, hidden_state_dim):
		super(SCFlowPoseHead, self).__init__()
		self.delta_flow_model = nn.Sequential(
			nn.Conv2d(2, 32, 3, 2, 1), # 2+1
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2, 1),
			nn.ReLU())
		self.hidden_state_model = nn.Sequential(
			nn.Conv2d(hidden_state_dim, 32, 3, 2, 1), #hideen_state_dim+1
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2, 1),
			nn.ReLU())
		# self.geo_model = nn.Sequential(
		# 	nn.Conv2d(3+1, 32, 3, 2, 1),
		# 	nn.ReLU(),
		# 	nn.Conv2d(32, 32, 3, 2, 1),
		# 	nn.ReLU())
		
		self.trans_fc = nn.Sequential(
			nn.Linear((32+32)*8*8, 1024), #32+32+32
			nn.ReLU(),
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Linear(256, 3))
		self.trans_fc[-1].weight.data = nn.Parameter(torch.zeros_like(self.trans_fc[-1].weight.data))
		self.trans_fc[-1].bias.data = nn.Parameter(torch.Tensor([0,0,0]))

		self.rotat_fc = nn.Sequential(
			nn.Linear((32+32)*8*8, 1024), #32+32+32
			nn.ReLU(),
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Linear(256, 6))
		self.rotat_fc[-1].weight.data = nn.Parameter(torch.zeros_like(self.rotat_fc[-1].weight.data))
		self.rotat_fc[-1].bias.data = nn.Parameter(torch.Tensor([1,0,0,0,1,0]))

	def forward(self, hidden_state, delta_flow):
		hidden_encode = self.hidden_state_model(hidden_state)
		delta_flow_encode = self.delta_flow_model(delta_flow)
		# hidden_encode = self.hidden_state_model(torch.cat([hidden_state,mask],dim=1))
		# delta_flow_encode = self.delta_flow_model(torch.cat([delta_flow,mask],dim=1))
		#geo_encode = self.geo_model(torch.cat([geo,mask],dim=1))
		#encode = torch.cat([hidden_encode, delta_flow_encode, geo_encode], dim=-3)
		encode = torch.cat([hidden_encode, delta_flow_encode], dim=-3)
		encode = encode.flatten(1, 3)
		rotation = self.rotat_fc(encode)
		translation = self.trans_fc(encode)
		result = torch.cat([rotation, translation], -1)
		return result
	