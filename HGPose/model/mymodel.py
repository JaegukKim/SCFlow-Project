import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from HGPose.model.geo_model import GeoModel
from HGPose.model.scflow_base import SCFlow
from HGPose.utils.geometry import render_geometry, apply_forward_flow, apply_imagespace_relative
from HGPose.model.backbone_scflow import ResNet34

class AggregateModule(nn.Module):
	def __init__(self):
		super(AggregateModule, self).__init__()
		self.image_encoder = ResNet34(input_dim=3)
		self.geo_encoder = nn.Sequential(
			nn.Conv2d(5, 32, 3, 1 ,1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, 3, 1 ,1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)
		self.geo_encoder2 = nn.Sequential(
			nn.Conv2d(128, 128, 3, 2 ,1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, 3, 2 ,1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 512, 3, 2, 1),
			nn.BatchNorm2d(512),
			nn.ReLU()
		)
		self.weight_encoder = nn.Sequential(
			nn.Conv2d(512, 256, 3, 2, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 128, 3, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 64, 3, 2, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 1, 1),
			nn.Sigmoid()
		)
		self.coord_head = nn.Sequential(
			nn.Conv2d(128, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 3, 3, 1, 1) ,
			nn.Tanh())
		self.mask_head = nn.Sequential(
			nn.Conv2d(128, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 1, 3, 1, 1) ,
			nn.Sigmoid())
		self.mask_visib_head = nn.Sequential(
			nn.Conv2d(128, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 1, 3, 1, 1) ,
			nn.Sigmoid())



	def forward(self, image_que, geo_flow_pred, geo_pred):
		f_que = self.image_encoder(image_que)
		f_que = f_que['feat3'] #[b, 512, 8, 8]
		# weight = self.weight_encoder(f_que)
		# weight = torch.nn.Sigmoid(weight)

		f1 = self.geo_encoder(geo_flow_pred) #[b, 128. 64. 64]
		f1_2 = self.geo_encoder2(f1) #[b, 512, 8, 8]
		f2 = self.geo_encoder(geo_pred) #[b, 128, 64, 64]
		f2_2 = self.geo_encoder2(f2) #[b, 512, 8, 8]
		weight = self.weight_encoder(f_que + f1_2 + f2_2)
		
		f_aggregated = f1 * weight + f2 * (1-weight)
		coord_aggregated = self.coord_head(f_aggregated)
		mask_aggregated = self.mask_head(f_aggregated)
		mask_visib_aggregated = self.mask_visib_head(f_aggregated)

		return coord_aggregated, mask_aggregated, mask_visib_aggregated

class PoseHead(nn.Module):
	def __init__(self, num_class):
		super(PoseHead, self).__init__()
		self.num_class = num_class
		self.image_encoder = nn.Sequential(
			nn.Conv2d(3, 64, 3, 2, 1), 
			nn.ReLU(),
			nn.Conv2d(64, 128, 3, 2, 1),
			nn.ReLU(),
			nn.Conv2d(128, 64, 3, 2, 1),
			nn.ReLU())
		self.coord_encoder = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, 1),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 2, 1),
			nn.ReLU())
		self.mask_encoder = nn.Sequential(
			nn.Conv2d(1, 64, 3, 1, 1),
			nn.ReLU(),
			nn.Conv2d(64, 32, 3, 2, 1),
			nn.ReLU())
		self.mask_visib_encoder = nn.Sequential(
			nn.Conv2d(1, 64, 3, 1, 1),
			nn.ReLU(),
			nn.Conv2d(64, 32, 3, 2, 1),
			nn.ReLU())
		self.conv_layers = nn.Sequential(
			nn.Conv2d(320, 128, 3, 2, 1),
			nn.GroupNorm(32, 128, eps=1e-05, affine=True),
			nn.ReLU(),
			nn.Conv2d(128, 64, 3, 2, 1),
			nn.GroupNorm(32, 64, eps=1e-05, affine=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 2, 1),
			nn.GroupNorm(32, 64, eps=1e-05, affine=True),
			nn.ReLU()
		)
		self.flatten_op = nn.Flatten(start_dim=1, end_dim=-1)
		self.fc_layers = nn.Sequential(
			nn.Linear(1024, 512, bias=True),
			nn.ReLU(),
			nn.Linear(512, 256, bias=True),
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

	def forward(self, image_ref, image_que, coord_ref, coord_que, mask_que, mask_visib_que, obj_id):
		#hidden_encode = self.hidden_state_model(hidden_state)
		ref_encode = self.image_encoder(image_ref)
		que_encode = self.image_encoder(image_que)
		coord_ref_encode = self.coord_encoder(coord_ref)
		coord_que_encode = self.coord_encoder(coord_que)
		mask_que_encode = self.mask_encoder(mask_que)
		mask_visib_que_encode = self.mask_visib_encoder(mask_visib_que)
		encode = torch.cat([ref_encode, que_encode, coord_ref_encode, coord_que_encode, mask_que_encode, mask_visib_que_encode], dim=-3)
		encode = self.conv_layers(encode)
		encode = self.flatten_op(encode)
		encode = self.fc_layers(encode)
		pred_translation = self.translation_pred(encode)
		pred_rotation = self.rotation_pred(encode)
		pred_translation = pred_translation.view(-1, self.num_class, 3)
		pred_rotation = pred_rotation.view(-1, self.num_class, 6)
		pred_translation = torch.index_select(pred_translation, dim=1, index=obj_id-1)[:, 0, :]
		pred_rotation = torch.index_select(pred_rotation, dim=1, index=obj_id-1)[:, 0, :]

		return pred_rotation, pred_translation



class MyModel(nn.Module):
	def __init__(self, cfg):
		super(MyModel, self).__init__()
		self.scflow_model = SCFlow(cfg)
		self.geo_model = GeoModel(cfg)
		self.image_size = cfg.image_size
		self.geo_size = 64
		self.aggregator = AggregateModule()
		self.posehead = PoseHead(cfg.num_class)

	def forward(self, image_ref, image_que, K_ref, K_que, RT_ref, coord_que, coord_ref, mask_ref, gt_flow, obj_id, num_flow_updates: int=4):
		coord_gt = F.interpolate(coord_que, [self.geo_size, self.geo_size], mode='bilinear')
		flow_preds, flow_RT_preds, sc_flow_preds, ref_mask_visib_preds = self.scflow_model(image_ref, image_que, K_ref, K_que, RT_ref, coord_ref, mask_ref, gt_flow, obj_id, num_flow_updates)

		flow_pred = flow_preds[-1]
		ref_mask_visib_pred = ref_mask_visib_preds[-1]

		coord_flow_pred = apply_forward_flow(flow_pred.unsqueeze(1), coord_ref.unsqueeze(1), -1.).squeeze(1)
		masks_flow_pred = apply_forward_flow(flow_pred.unsqueeze(1), torch.cat([mask_ref, ref_mask_visib_pred], 1).unsqueeze(1), 0.).squeeze(1)
		geo_flow_pred = torch.cat([coord_flow_pred, masks_flow_pred], 1)
		geo_flow_pred = F.interpolate(geo_flow_pred, [self.geo_size, self.geo_size], mode='bilinear')  #[b, 5, 64, 64]

		coord_pred, que_mask_pred, que_mask_visib_pred = self.geo_model(image_que.unsqueeze(1), obj_id)
		#coord_pred = coord_gt
		geo_pred = torch.concat([coord_pred, que_mask_pred, que_mask_visib_pred],1) #[b, 5, 64, 64]

		coord_aggregated, mask_aggregated, mask_visib_aggregated = self.aggregator(image_que, geo_flow_pred.detach(), geo_pred.detach())
		#coord_aggregated = coord_gt

		geo_aggregated = torch.concat([coord_aggregated, mask_aggregated, mask_visib_aggregated], 1) #[b, 5, 64, 64]

		coord_ref = F.interpolate(coord_ref, [self.geo_size, self.geo_size], mode='nearest')
		delta_rotation, delta_translation = self.posehead(image_ref, image_que, coord_ref, coord_aggregated, mask_aggregated, mask_visib_aggregated, obj_id)
		delta_relative = torch.cat([delta_rotation, delta_translation], -1)
		RT_final = apply_imagespace_relative(K_que, K_que, RT_ref.detach(), delta_relative, [self.image_size, self.image_size])
		
		return flow_preds, flow_RT_preds, sc_flow_preds, ref_mask_visib_preds, geo_flow_pred, geo_pred, geo_aggregated, RT_final