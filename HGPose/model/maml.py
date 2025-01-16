from collections import defaultdict
import torch
import random
from HGPose.utils.geometry import (get_flow_from_delta_pose_and_xyz, bbox_add_noise, squaring_boxes, apply_forward_flow, jitter_RT,
								   get_K_crop_resize, image_cropping, render_geometry, apply_backward_flow)
from HGPose.utils.loss import sequence_loss, geometry_loss, mask_l1_loss, grid_matching_loss

def fast_adapt(task_ref, learner, cfg):
	randomlist = random.sample(range(cfg.N_ref), cfg.N_ref)
	spt_ref = {k: task_ref[k].transpose(0, 1).expand(cfg.N_ref, cfg.N_ref, *task_ref[k].shape[2:]) for k in ['image', 'mask', 'RT', 'K_img', 'geo']}
	spt_que = {k: task_ref[k].transpose(0, 1)[randomlist] for k in ['image', 'mask', 'RT', 'K_img', 'geo']}
	spt_ref['xyz_mesh'] = task_ref['xyz_mesh'] * cfg.N_ref

	img_box = torch.tensor([0, 0, cfg.img_size, cfg.img_size]).to(spt_que['image']).expand(cfg.N_ref, 1, 4)
	crop_box = bbox_add_noise(img_box, std_rate=0.2)
	crop_box = squaring_boxes(crop_box, lamb=1)
	spt_que['K_img'] = get_K_crop_resize(spt_que['K_img'], crop_box, [cfg.img_size, cfg.img_size])
	spt_que['image'] = image_cropping(spt_que['image'], crop_box, [cfg.img_size, cfg.img_size])
	spt_que['mask'] = image_cropping(spt_que['mask'], crop_box, [cfg.img_size, cfg.img_size]) 
	spt_que['geo'] = image_cropping((spt_que['geo'] + 1) / 2, crop_box, [cfg.img_size, cfg.img_size]) * 2 - 1
	spt_flow_r2q_gt = get_flow_from_delta_pose_and_xyz(spt_que['RT'], spt_que['K_img'], spt_ref['geo'], spt_ref['mask'], spt_que['geo'])
	spt_que['RT_coarse'] = jitter_RT(spt_que['RT'].flatten(0, 1)).unflatten(0, (spt_que['RT'].shape[:-2]))

	for k in range(0, cfg.update_step):
		pred = learner(
			spt_ref['xyz_mesh'], 
			spt_ref['image'], 
			spt_ref['mask'], 
			spt_ref['geo'], 
			spt_que['RT_coarse'], 
			spt_que['image'], 
			spt_que['K_img'], 
			mode='flow')

		grid_loss = 0
		# for k in range(1, cfg.refine_step+1):
		# 	# grid_loss += 10 * grid_matching_loss(pred[f'RT_{k}'], spt_que['RT'], 
		# 	# 	spt_que['K_img'], cfg.img_size, cfg.img_size) / cfg.refine_step
		flow_loss = 0.1 * sequence_loss(pred['flow_r2q'], spt_flow_r2q_gt)
		geo_loss = 10 * geometry_loss(pred['geo'], spt_que['geo'], spt_que['mask'])
		mask_loss = 10 * mask_l1_loss(pred['mask'], spt_que['mask'])
		loss = flow_loss + (geo_loss + mask_loss) + (grid_loss)
		learner.adapt(loss)
	return learner
