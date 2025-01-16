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

class DINOv2_ViTs14(nn.Module):
	def __init__(self):
		super(DINOv2_ViTs14, self).__init__()
		self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
		self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
		self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
		self.f_dim = {2:384, 3:384, 4:384}
	def forward(self, img):
		f = OrderedDict()
		img = F.interpolate(img, [224, 224])
		img = (img - self.mean) / self.std
		x = self.backbone.forward_features(img)["x_norm_patchtokens"]
		x = x.permute(0, 2, 1)
		x = x.reshape(x.shape[0], x.shape[1], 224//14, 224//14)
		x = F.normalize(x, dim=-3, p=2)
		f[2] = F.interpolate(x, [32, 32])
		f[3] = F.interpolate(x, [16, 16])
		f[4] = F.interpolate(x, [8, 8])
		return f

class ResNet18(nn.Module):
	def __init__(self, input_dim=3):
		super(ResNet18, self).__init__()
		self.resnet = resnet18(pretrained=True)
		self.input_dim = input_dim
		self.resnet.conv1 = nn.Conv2d(self.input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		self.f_dim = {2:128, 3:256, 4:512}
		self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
		self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
	def forward(self, x):
		f = OrderedDict()
		x = (x - self.mean) / self.std
		x = self.resnet.conv1(x)
		x = self.resnet.bn1(x)
		x = self.resnet.relu(x)
		x = self.resnet.maxpool(x)
		x = self.resnet.layer1(x)  # (B, C, H/2, W/2)
		f[2] = self.resnet.layer2(x)
		f[3] = self.resnet.layer3(f[2])
		f[4] = self.resnet.layer4(f[3])
		return f

class ResNet34(nn.Module):
	def __init__(self, input_dim=3):
		super(ResNet34, self).__init__()
		self.model = nn.Sequential(*(list(resnet34(pretrained=True, ).children())[:-2]))
		self.model[0] = nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		self.f_dim = {2:128, 3:256, 4:512}
	def forward(self, x):
		f = OrderedDict()
		x = self.model[0](x)
		x = self.model[1](x)
		x = self.model[2](x)
		x = self.model[3](x)
		x = self.model[4](x)
		f[2] = self.model[5](x)
		f[3] = self.model[6](f[2])
		f[4] = self.model[7](f[3])
		return f

class ResNet50(nn.Module):
	def __init__(self, input_dim=3):
		super(ResNet50, self).__init__()
		self.model = nn.Sequential(*(list(resnet50(pretrained=True, ).children())[:-2]))
		self.model[0] = nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		self.f_dim = {2:512, 3:1024, 4:2048}
	def forward(self, x):
		f = OrderedDict()
		x = self.model[0](x)
		x = self.model[1](x)
		x = self.model[2](x)
		x = self.model[3](x)
		x = self.model[4](x)
		f[2] = self.model[5](x)
		f[3] = self.model[6](f[2])
		f[4] = self.model[7](f[3])
		return f

class ConvNextTiny(nn.Module):
	def __init__(self, input_dim=3):
		super(ConvNextTiny, self).__init__()
		self.model = convnext_tiny(pretrained=True, in_22k=True, num_classes=21841)
		delattr(self.model, 'norm')
		delattr(self.model, 'head')
		self.model.downsample_layers[0][0] = nn.Conv2d(input_dim, 96, kernel_size=(4, 4), stride=(4, 4))
		self.f_dim = {2:192, 3:384, 4:768}
	def forward(self, x):
		f = OrderedDict()
		x = self.model.downsample_layers[0](x)
		x = self.model.stages[0](x)
		x = self.model.downsample_layers[1](x)
		x = self.model.stages[1](x)
		f[2] = x
		x = self.model.downsample_layers[2](x)
		x = self.model.stages[2](x)
		f[3] = x
		x = self.model.downsample_layers[3](x)  # 768 x 8 x 8
		x = self.model.stages[3](x)
		f[4] = x
		return f

class CoarsePoseEstimator(nn.Module):
	def __init__(self):
		super(CoarsePoseEstimator, self).__init__()
		self.model = resnet34(pretrained=True)
		self.model.last_layer = nn.Conv2d(512, 16, 1, 1)
		self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
		self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
	def forward(self, x):
		x = (x - self.mean) / self.std
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)
		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		x = self.model.layer4(x)
		x = self.model.last_layer(x)
		return x


class PoseHead(nn.Module):
	def __init__(self, f_dim=None, size=None):
		super(PoseHead, self).__init__()
		self.f_dim = f_dim
		self.feature_size = [size[0] // 8, size[1] // 8]
		self.model = nn.Sequential(
			nn.Conv2d(self.f_dim, 128, 3, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Flatten(1, 3),
			nn.Linear(128 * self.feature_size[0] * self.feature_size[1], 1024),
			nn.LeakyReLU(0.1),
			nn.Linear(1024, 256),
			nn.LeakyReLU(0.1))
		self.trans_fc = nn.Linear(256, 3, bias=True)
		self.trans_fc.weight.data = nn.Parameter(torch.zeros_like(self.trans_fc.weight.data))
		self.trans_fc.bias.data = nn.Parameter(torch.Tensor([0,0,0]))
		self.rotat_fc = nn.Linear(256, 6, bias=True)
		self.rotat_fc.weight.data = nn.Parameter(torch.zeros_like(self.rotat_fc.weight.data))
		self.rotat_fc.bias.data = nn.Parameter(torch.Tensor([1,0,0,0,1,0]))

	def forward(self, f):
		encoded = self.model(f)
		rotation = self.rotat_fc(encoded)
		translation = self.trans_fc(encoded)
		result = torch.cat([rotation, translation], -1)
		return result

class FlowPoseHead(nn.Module):
	def __init__(self, hidden_state_dim):
		super(FlowPoseHead, self).__init__()
		self.delta_flow_model = nn.Sequential(
			nn.Conv2d(2+1, 32, 3, 2, 1),
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2, 1),
			nn.ReLU())
		self.hidden_state_model = nn.Sequential(
			nn.Conv2d(hidden_state_dim+1, 32, 3, 2, 1),
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2, 1),
			nn.ReLU())
		self.geo_model = nn.Sequential(
			nn.Conv2d(3+1, 32, 3, 2, 1),
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2, 1),
			nn.ReLU())
		
		self.trans_fc = nn.Sequential(
			nn.Linear((32+32+32)*8*8, 1024),
			nn.ReLU(),
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Linear(256, 3))
		self.trans_fc[-1].weight.data = nn.Parameter(torch.zeros_like(self.trans_fc[-1].weight.data))
		self.trans_fc[-1].bias.data = nn.Parameter(torch.Tensor([0,0,0]))

		self.rotat_fc = nn.Sequential(
			nn.Linear((32+32+32)*8*8, 1024),
			nn.ReLU(),
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Linear(256, 6))
		self.rotat_fc[-1].weight.data = nn.Parameter(torch.zeros_like(self.rotat_fc[-1].weight.data))
		self.rotat_fc[-1].bias.data = nn.Parameter(torch.Tensor([1,0,0,0,1,0]))

	def forward(self, hidden_state, delta_flow, geo, mask):
		hidden_encode = self.hidden_state_model(torch.cat([hidden_state,mask],dim=1))
		delta_flow_encode = self.delta_flow_model(torch.cat([delta_flow,mask],dim=1))
		geo_encode = self.geo_model(torch.cat([geo,mask],dim=1))
		encode = torch.cat([hidden_encode, delta_flow_encode, geo_encode], dim=-3)
		encode = encode.flatten(1, 3)
		rotation = self.rotat_fc(encode)
		translation = self.trans_fc(encode)
		result = torch.cat([rotation, translation], -1)
		return result



class HeavyPoseHead(nn.Module):
	def __init__(self, input_dim=3, size=None):
		super(HeavyPoseHead, self).__init__()
		self.resnet = resnet18(pretrained=True)
		self.input_dim = input_dim
		self.feature_size = [size[0] // 32, size[1] // 32]
		self.resnet.conv1 = nn.Conv2d(self.input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		self.f_dim = {2:128, 3:256, 4:512}
		
		self.trans_fc = nn.Sequential(
			nn.Linear(self.f_dim[4] * self.feature_size[0] * self.feature_size[1], 1024, bias=False),
			nn.LayerNorm(1024),
			nn.LeakyReLU(0.1),
			nn.Linear(1024, 256, bias=False),
			nn.LayerNorm(256),
			nn.LeakyReLU(0.1),
			nn.Linear(256, 3, bias=False))
		# self.trans_fc[-1].weight.data = nn.Parameter(torch.zeros_like(self.trans_fc[-1].weight.data))
		# self.trans_fc[-1].bias.data = nn.Parameter(torch.Tensor([0,0,0]))

		self.rotat_fc = nn.Sequential(
			nn.Linear(self.f_dim[4] * self.feature_size[0] * self.feature_size[1], 1024, bias=False),
			nn.LayerNorm(1024),
			nn.LeakyReLU(0.1),
			nn.Linear(1024, 256, bias=False),
			nn.LayerNorm(256),
			nn.LeakyReLU(0.1),
			nn.Linear(256, 128, bias=False),
			nn.LayerNorm(128),
			nn.LeakyReLU(0.1),
			nn.Linear(128, 6, bias=False))
		# self.rotat_fc[-1].weight.data = nn.Parameter(torch.zeros_like(self.rotat_fc[-1].weight.data))
		# self.rotat_fc[-1].bias.data = nn.Parameter(torch.Tensor([0.1,0,0,0,0.1,0]))
		# self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
		# self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
	def forward(self, x):
		# x = (x - self.mean) / self.std
		x = self.resnet.conv1(x)
		x = self.resnet.bn1(x)
		x = self.resnet.relu(x)
		x = self.resnet.maxpool(x)
		x = self.resnet.layer1(x)  # (B, C, H/2, W/2)
		x = self.resnet.layer2(x)
		x = self.resnet.layer3(x)
		x = self.resnet.layer4(x)
		x = x.flatten(1, 3)
		rotation = self.rotat_fc(x)
		rotation = F.tanh(rotation)
		translation = self.trans_fc(x)
		result = torch.cat([rotation, translation], -1)
		return result


class CustomPoseHead(nn.Module):
	def __init__(self, f_dim=None, size=None):
		super(CustomPoseHead, self).__init__()
		self.f_dim = f_dim
		self.feature_size = [size[0] // 8, size[1] // 8]
		self.model = nn.Sequential(
			nn.Conv2d(self.f_dim, 128, 3, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Flatten(1, 3),
			nn.Linear(128 * self.feature_size[0] * self.feature_size[1], 1024),
			nn.LeakyReLU(0.1),
			nn.Linear(1024, 256),
			nn.LeakyReLU(0.1))
		self.trans_fc = nn.Linear(256, 3, bias=True)
		self.trans_fc.weight.data = nn.Parameter(torch.zeros_like(self.trans_fc.weight.data))
		self.trans_fc.bias.data = nn.Parameter(torch.Tensor([0, 0, 0]))
		self.rotat_fc = nn.Linear(256, 6, bias=True)
		self.rotat_fc.weight.data = nn.Parameter(torch.zeros_like(self.rotat_fc.weight.data))
		self.rotat_fc.bias.data = nn.Parameter(torch.Tensor([0.01, 0, 0, 0, 0.01, 0]))

	def forward(self, f):
		encoded = self.model(f)
		rotation = self.rotat_fc(encoded)
		rotation = F.tanh(rotation)
		translation = self.trans_fc(encoded)
		result = torch.cat([rotation, translation], -1)
		return result


class OnlyConvAggregator(nn.Module):
	def __init__(self, geo_dim, channel=32):
		super(OnlyConvAggregator, self).__init__()
		self.geo_dim = geo_dim
		self.channel = channel
		self.each_model = nn.Sequential(
				nn.Conv2d(self.geo_dim, self.channel, 5, 1, 2),
				nn.BatchNorm2d(self.channel),
				nn.ReLU(),
				nn.Conv2d(self.channel, self.channel, 5, 1, 2),
				nn.BatchNorm2d(self.channel),
				nn.ReLU(),
				nn.Conv2d(self.channel, self.channel, 3, 1, 1))
		self.all_model = nn.Sequential(
				nn.Conv2d(self.channel, self.channel, 3, 1, 1),
				nn.BatchNorm2d(self.channel),
				nn.ReLU(),
				nn.Conv2d(self.channel, self.channel, 3, 1, 1),
				nn.BatchNorm2d(self.channel), 
				nn.ReLU(),
				nn.Conv2d(self.channel, self.geo_dim, 3, 1, 1))
	def forward(self, recon_geo):
		bsz, N = recon_geo.shape[:2]
		each_ref = self.each_model(recon_geo.flatten(0, 1)).unflatten(0, (bsz, N))
		geometry = self.all_model(each_ref.mean(1)).unsqueeze(1)
		geometry = F.tanh(geometry)
		return geometry
	

class OnlyDeformableConv(nn.Module):
	def __init__(self, geo_dim, channel=32):
		super(OnlyDeformableConv, self).__init__()
		self.geo_dim = geo_dim
		self.channel = channel
		self.each_model = nn.Sequential(
				nn.Conv2d(self.geo_dim, self.channel, 5, 1, 2),
				nn.BatchNorm2d(self.channel),
				nn.ReLU(),
				nn.Conv2d(self.channel, self.channel, 5, 1, 2),
				nn.BatchNorm2d(self.channel),
				nn.ReLU(),
				nn.Conv2d(self.channel, self.channel, 3, 1, 1))
		self.deformable = DeformableConv2d(self.channel, self.channel, 5, 1, 2)
		self.all_model = nn.Sequential(
				nn.Conv2d(self.channel, self.channel, 3, 1, 1),
				nn.BatchNorm2d(self.channel),
				nn.ReLU(),
				nn.Conv2d(self.channel, self.geo_dim, 3, 1, 1))
	def forward(self, recon_geo):
		bsz, N = recon_geo.shape[:2]
		each_ref = self.each_model(recon_geo.flatten(0, 1)).unflatten(0, (bsz, N))
		geometry = self.deformable(each_ref.mean(1))
		geometry = self.all_model(geometry).unsqueeze(1)
		geometry = F.tanh(geometry)
		return geometry


class DeformableAggregator(nn.Module):
	def __init__(self, geo_dim, N_ref, channel=32):
		super(DeformableAggregator, self).__init__()
		self.geo_dim = geo_dim
		self.channel = channel
		self.N_ref = N_ref
		self.each_model = nn.Sequential(
				nn.Conv2d(self.geo_dim, self.channel, 5, 1, 2),
				nn.BatchNorm2d(self.channel),
				nn.ReLU(),
				nn.Conv2d(self.channel, self.channel, 5, 1, 2),
				nn.BatchNorm2d(self.channel),
				nn.ReLU(),
				nn.Conv2d(self.channel, self.channel, 3, 1, 1))
		self.deformable = DeformableConv2dGuided(self.channel, self.channel, self.N_ref, 5, 1, 2)
		self.all_model = nn.Sequential(
				nn.Conv2d(self.channel, self.channel, 3, 1, 1),
				nn.BatchNorm2d(self.channel),
				nn.ReLU(),
				nn.Conv2d(self.channel, self.geo_dim, 3, 1, 1))
	def forward(self, recon_geo):
		bsz, N = recon_geo.shape[:2]
		each_ref = self.each_model(recon_geo.flatten(0, 1)).unflatten(0, (bsz, N))
		geometry, offset = self.deformable(each_ref)
		geometry = self.all_model(geometry).unsqueeze(1)
		geometry = F.tanh(geometry)
		return geometry


class RAFT_small(nn.Module):
	def __init__(self):
		super(RAFT_small, self).__init__()
		weights = Raft_Small_Weights.DEFAULT
		self.transforms = weights.transforms()
		self.model = raft_small(weights=weights, progress=False)
	def forward(self, img1, img2):
		img1, img2 = self.transforms(img1, img2)
		list_of_flows = self.model(img1, img2, num_flow_updates=2)
		return list_of_flows


class RAFT_large(nn.Module):
	def __init__(self, flow_refine_step=2):
		super(RAFT_large, self).__init__()
		weights = Raft_Large_Weights.DEFAULT
		self.transforms = weights.transforms()
		self.model = raft_large(weights=weights, progress=False)
		self.flow_refine_step = flow_refine_step
	def forward(self, img1, img2):
		img1, img2 = self.transforms(img1, img2)
		list_of_flows = self.model(img1, img2, num_flow_updates=self.flow_refine_step)
		return list_of_flows
	

class RAFT_custom(nn.Module):
	def __init__(self, size):
		super(RAFT_custom, self).__init__()
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
		self.size = size


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

		self.pose_head = FlowPoseHead(recurrent_block_hidden_state_size)
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

		flow_predictions = []
		pose_predictions = []

		RT_ref_scaled = apply_imagespace_relative(K_ref, K_que, RT_ref, self.fix.expand(batch_size, 9), self.size)
		RT_que = RT_ref_scaled.clone()

		flow = coords1.detach() - coords0
		up_flow = upsample_flow(flow=flow, factor=8)
		geo = apply_forward_flow(up_flow.unsqueeze(0), geo_ref.unsqueeze(0), -1.0)[0]
		mask = apply_forward_flow(up_flow.unsqueeze(0), mask_ref.unsqueeze(0), 0.0)[0]
		geo = F.interpolate(geo, scale_factor=1/8)
		mask = F.interpolate(mask, scale_factor=1/8)

		for _ in range(num_flow_updates):
			coords1 = coords1.detach()  # Don't backpropagate gradients through this branch, see paper
			corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)
			hidden_state, delta_flow = self.update_block(hidden_state, context, corr_features, flow)
			# up_mask = self.mask_predictor(hidden_state)
		
			delta_relative = self.pose_head(hidden_state, delta_flow, geo, mask)
			RT_que = apply_imagespace_relative(K_que, K_que, RT_que, delta_relative, self.size)
			shape_constraint_delta_flow = get_flow_from_delta_pose_and_xyz(
				RT_que.unsqueeze(0).detach(), K_que.unsqueeze(0), geo_ref.unsqueeze(0), mask_ref.unsqueeze(0))[0]
			shape_constraint_delta_flow = F.interpolate(shape_constraint_delta_flow, scale_factor=1/8) / 8
			coords1 = coords1 + shape_constraint_delta_flow
			
			flow = coords1 - coords0
			up_flow = upsample_flow(flow=flow, factor=8)
			geo = apply_forward_flow(up_flow.unsqueeze(0), geo_ref.unsqueeze(0), -1.0)[0]
			mask = apply_forward_flow(up_flow.unsqueeze(0), mask_ref.unsqueeze(0), 0.0)[0]
			geo = F.interpolate(geo, scale_factor=1/8)
			mask = F.interpolate(mask, scale_factor=1/8)

			flow_predictions.append(up_flow)
			pose_predictions.append(RT_que)
		return flow_predictions, pose_predictions
	

