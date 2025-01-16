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
	get_2d_coord, apply_imagespace_relative, render_geometry, get_separate_medoid, get_region_smoothness_and_variation, get_pose_from_delta_pose_scflow)
from HGPose.model.mfn import GaborNet
from HGPose.utils.flow import filter_flow_by_mask

class SCFlow(nn.Module):
	def __init__(self, cfg):
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
		self.num_class = cfg.num_class
		self.pose_head = SCFlowPoseHead_scflow(self.num_class)
		self.register_buffer('fix', torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0]).to(torch.float).view(1, 9))

	def forward(self, image_ref, image_que, K_ref, K_que, RT_ref, geo_ref, mask_ref, gt_flow, obj_id, num_flow_updates: int=4):
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

		# RT_update = RT_ref
		RT_ref_scaled = apply_imagespace_relative(K_ref, K_que, RT_ref, self.fix.expand(batch_size, 9), self.size)
		RT_que = RT_ref_scaled.clone()

		
		downsampled_gt_flow = F.interpolate(gt_flow, scale_factor=1/8, mode="bilinear") / 8
		
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
			
			delta_rotation, delta_translation = self.pose_head(hidden_state, delta_flow, pred_mask, obj_id)   #delta_flow.detach(), pred_mask.detach()
			current_rotation = RT_que[:,:3,:3]
			current_translation = RT_que[:,:3,3]
			update_rotation,update_translation = get_pose_from_delta_pose_scflow(
				delta_rotation, delta_translation,
				current_rotation.detach(), current_translation.detach(), depth_transform = 'linear'
			)
			RT_que = torch.ones(batch_size, 4, 4).to(update_rotation)
			RT_que[:, :3, :3] = update_rotation
			RT_que[:, :3, 3] = update_translation
		

			# delta_relative = torch.cat([delta_rotation, delta_translation], -1)
			# RT_que = apply_imagespace_relative(K_que, K_que, RT_que.detach(), delta_relative, self.size)
			pose_predictions.append(RT_que)

			shape_constraint_flow = get_flow_from_delta_pose_and_xyz_scflow(
				RT_que.unsqueeze(1).detach(), K_que.unsqueeze(1), geo_ref.unsqueeze(1)).squeeze(1)
			sc_flow_predictions.append(shape_constraint_flow)
			flow = shape_constraint_flow
			


		return flow_predictions, pose_predictions, sc_flow_predictions, mask_predictions


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
		self.conv_layers = nn.Sequential(
			nn.Conv2d(224, 128, 3, 2, 1),
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

class SCFlowPoseHead(nn.Module):
	def __init__(self, hidden_state_dim, num_class):
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
		self.mask_model = nn.Sequential(
			nn.Conv2d(1, 32, 3, 2, 1),
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2, 1),
			nn.ReLU())

		self.trans_fc = nn.Sequential(
			nn.Linear((32+32+32)*8*8, 1024), #32+32+32
			nn.ReLU(),
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Linear(256, 3*num_class))
		self.trans_fc[-1].weight.data = nn.Parameter(torch.zeros_like(self.trans_fc[-1].weight.data))
		self.trans_fc[-1].bias.data = nn.Parameter(torch.zeros(3*num_class))

		self.rotat_fc = nn.Sequential(
			nn.Linear((32+32+32)*8*8, 1024), #32+32+32
			nn.ReLU(),
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Linear(256, 6*num_class))
		self.rotat_fc[-1].weight.data = nn.Parameter(torch.zeros_like(self.rotat_fc[-1].weight.data))
		self.rotat_fc[-1].bias.data = nn.Parameter(torch.cat([torch.Tensor([1,0,0,0,1,0])] * num_class))
		self.num_class = num_class

	def forward(self, hidden_state, delta_flow, mask, obj_id):
		hidden_encode = self.hidden_state_model(hidden_state)
		delta_flow_encode = self.delta_flow_model(delta_flow)
		mask_encode = self.mask_model(mask)
		encode = torch.cat([hidden_encode, delta_flow_encode, mask_encode], dim=-3)
		encode = encode.flatten(1, 3)
		rotation = self.rotat_fc(encode)
		translation = self.trans_fc(encode)
		#
		#rotation = torch.index_select(rotation, dim=1, index=obj_id-1)
		#translation = torch.index_select(translation, dim=1, index=obj_id-1)
		rotation = rotation.view(-1, self.num_class, 6)
		translation = translation.view(-1, self.num_class, 3)
		rotation = torch.index_select(rotation, dim=1, index=obj_id-1)[:,0,:]
		translation = torch.index_select(translation, dim=1, index=obj_id-1)[:,0,:]
		#rotation = torch.index_select(rotation, dim=1, index=obj_id-1)[:, 0, :]
		#translation = torch.index_select(translation, dim=1, index=obj_id-1)[:, 0, :]
		
		result = torch.cat([rotation, translation], -1)
		#
		return result
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
	
