import numpy as np
import torch
import cv2
import torch.nn.functional as F
from HGPose.utils.geometry import get_3d_bbox_from_pts, render_geometry, apply_forward_flow, render_mesh, make_region, apply_backward_flow
from bop_toolkit.bop_toolkit_lib.misc import project_pts
from torchvision.utils import flow_to_image
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic
from skimage.color import label2rgb
import random

def region_to_img(region):
	n_class = region.shape[-3]
	img = torch.argmax(region, dim=-3, keepdim=True) / n_class
	return img
def convert_tensor_to_images_scflow_geo(que, ref, flow_pred, RT_refine, gt_flow_r2q, sc_flow_pred, ref_mask_visib_pred, ref_mask_visib_gt, 
			coord_flow_pred, mask_flow_pred, mask_visib_flow_pred, coord_pred, mask_pred, mask_visib_pred,coord_agg, mask_agg, mask_visib_agg, RT_final, cfg, invalid_flow=400.0, size=[128, 128]):
	ref_image= ref['image'][0]
	que_image = que['image'][0]

	gt_recon_query = apply_forward_flow(gt_flow_r2q, ref['image'], 0.0)
	gt_recon_query = gt_recon_query[0, :, -3:] 

	pred_recon_query = apply_forward_flow(flow_pred, ref['image'], 0.0)
	pred_recon_query = pred_recon_query[0, :, -3:] 

	
	flow_mask = (gt_flow_r2q[0]!=invalid_flow).to(gt_flow_r2q)

	gt_flow_r2q = flow_to_image(gt_flow_r2q[0] * flow_mask) / 255.0
	pred_flow_r2q = flow_to_image(flow_pred[0] * flow_mask) / 255.0


	#que_mask = que['mask'][0].repeat(1,3,1,1)
	ref_mask_visib_gt = ref_mask_visib_gt[0].repeat(1,3,1,1)
	ref_mask_visib_pred = ref_mask_visib_pred[0].repeat(1,3,1,1)


	gt_box = bbox_3d_visualize(que['RT'], que['K'], ref['source'], que['image'], color=(0, 0, 255))
	pred_refine_box = bbox_3d_visualize(RT_refine, que['K'], ref['source'], gt_box, color=(0, 255, 0))
	pred_final_box = bbox_3d_visualize(RT_final, ref['K_img'], ref['source'], pred_refine_box, color=(255, 0, 0))
	final_box = pred_final_box[0]

	coord_gt = (que['xyz'][0] + 1)/2
	coord_flow_pred = (coord_flow_pred[0] + 1)/2
	coord_pred = (coord_pred[0] + 1)/2
	coord_agg = (coord_agg[0] + 1)/2

	mask_gt = que['mask'][0].repeat(1,3,1,1)
	mask_flow_pred = mask_flow_pred[0].repeat(1,3,1,1)
	mask_pred = mask_pred[0].repeat(1,3,1,1)
	mask_agg = mask_agg[0].repeat(1,3,1,1)

	mask_visib_gt = que['mask_visib'][0].repeat(1,3,1,1)
	mask_visib_flow_pred = mask_visib_flow_pred[0].repeat(1,3,1,1)
	mask_visib_pred = mask_visib_pred[0].repeat(1,3,1,1)
	mask_visib_agg = mask_visib_agg[0].repeat(1,3,1,1)

	




	d = {
		'ref_i': ref_image, 
		'box_i': final_box,
		'gt_ref_visib_mask': ref_mask_visib_gt,
		'pred_ref_visib_mask': ref_mask_visib_pred,

		'gt_recon_q': gt_recon_query, 
		'pred_recon_q': pred_recon_query, 
		'gt_f': gt_flow_r2q,
		'pred_f': pred_flow_r2q, 

		'gt_coord': coord_gt,
		'flow_pred_coord': coord_flow_pred,
		'pred_coord': coord_pred,
		'final_coord': coord_agg,

		'gt_mask': mask_gt,
		'flow_pred_mask': mask_flow_pred,
		'pred_mask': mask_pred,
		'final_mask': mask_agg,

		'gt_mask_visib': mask_visib_gt,
		'flow_pred_mask_visib': mask_visib_flow_pred,
		'pred_mask_visib': mask_pred,
		'final_mask_visib': mask_agg,
		}
	
	for k, v in d.items():
		v = F.interpolate(v, size=size, mode='nearest')
		v = np.transpose(v.cpu().numpy(), (0, 2, 3, 1))
		v = np.concatenate(list(v), axis=1)
		d[k] = v

	image1 = np.concatenate([d['ref_i'], d['box_i'], d['gt_ref_visib_mask'], d['pred_ref_visib_mask']], axis=0)
	image2 = np.concatenate([d['gt_recon_q'], d['pred_recon_q'], d['gt_f'], d['pred_f']], axis=0)
	image3 = np.concatenate([d['gt_coord'], d['flow_pred_coord'], d['pred_coord'], d['final_coord']], axis=0)
	image4 = np.concatenate([d['gt_mask'], d['flow_pred_mask'], d['pred_mask'], d['final_mask']], axis=0)
	image5 = np.concatenate([d['gt_coord'], d['flow_pred_mask_visib'], d['pred_mask_visib'], d['final_mask_visib']], axis=0)



	stitched_image = np.concatenate([image1, image2, image3, image4, image5], axis=1)

	return stitched_image

def convert_tensor_to_images_scflow_geo2(que, ref, flow_pred, RT_refine, gt_flow_r2q, sc_flow_pred, mask_pred, mask_gt, 
										 current_coord, que_coord_pred, que_mask_pred, que_mask_visib_pred, cfg, invalid_flow=400.0, size=[128, 128]):
	ref_image= ref['image'][0]
	que_image = que['image'][0]
	que_image_masked = que_image * que['mask'][0]

	gt_recon_query = apply_forward_flow(gt_flow_r2q, ref['image'], 0.0)
	gt_recon_query = gt_recon_query[0, :, -3:] 
	pred_recon_query = apply_forward_flow(flow_pred, ref['image'], 0.0)
	pred_recon_query = pred_recon_query[0, :, -3:] 
	sc_pred_recon_query = apply_forward_flow(sc_flow_pred, ref['image'], 0.0)
	sc_pred_recon_query = sc_pred_recon_query[0, :, -3:] 
	
	flow_mask = (gt_flow_r2q[0]!=invalid_flow).to(gt_flow_r2q)
	gt_flow_r2q = flow_to_image(gt_flow_r2q[0] * flow_mask) / 255.0
	pred_flow_r2q = flow_to_image(flow_pred[0] * flow_mask) / 255.0
	sc_pred_flow_r2q = flow_to_image(sc_flow_pred[0] * flow_mask) / 255.0


	#que_mask = que['mask'][0].repeat(1,3,1,1)
	gt_mask = mask_gt[0].repeat(1,3,1,1)
	pred_mask = mask_pred[0].repeat(1,3,1,1)

	gt_que_mask = que['mask'][0].repeat(1,3,1,1)
	pred_que_mask = que_mask_pred[0].repeat(1,3,1,1)

	gt_que_mask_visib = que['mask_visib'][0].repeat(1,3,1,1)
	pred_que_mask_visib = que_mask_visib_pred[0].repeat(1,3,1,1)

	gt_que_coord = (que['xyz'][0] + 1)/2 * pred_que_mask_visib
	current_coord = (current_coord[0] + 1)/2
	pred_que_coord = ((que_coord_pred[0] + 1)/2) * pred_que_mask_visib


	gt_box = bbox_3d_visualize(que['RT'], que['K'], ref['source'], que['image'], color=(0, 255, 0))[0]
	pred_box = bbox_3d_visualize(RT_refine, que['K'], ref['source'], que['image'], color=(0, 0, 255))[0]
	ref_box = bbox_3d_visualize(ref['RT'], ref['K_img'], ref['source'], que['image'], color=(255, 0, 0))[0]


	d = {
		'ref_i': ref_image, 
		'que_i': que_image,
		'que_m': que_image_masked,
		'gt_recon_q': gt_recon_query, 
		'pred_recon_q': pred_recon_query, 
		'sc_pred_recon_q': sc_pred_recon_query,
		'gt_f': gt_flow_r2q,
		'pred_f': pred_flow_r2q, 
		'sc_pred_f': sc_pred_flow_r2q,
		'gt_box': gt_box,
		'pred_box': pred_box,
		'ref_box': ref_box,
		'gt_mask': gt_mask,
		'pred_mask': pred_mask,
		'gt_que_mask': gt_que_mask,
		'pred_que_mask': pred_que_mask,
		'gt_que_mask_visib': gt_que_mask_visib,
		'pred_que_mask_visib': pred_que_mask_visib,
		'gt_que_coord': gt_que_coord,
		'pred_que_coord': pred_que_coord,
		'current_coord': current_coord
		}
	
	for k, v in d.items():
		v = F.interpolate(v, size=size, mode='nearest')
		v = np.transpose(v.cpu().numpy(), (0, 2, 3, 1))
		v = np.concatenate(list(v), axis=1)
		d[k] = v

	
	image = np.concatenate([d['ref_i'], d['que_i'], d['que_m']],axis=0)
	recon = np.concatenate([d['gt_recon_q'], d['pred_recon_q'], d['sc_pred_recon_q']],axis=0)
	flow = np.concatenate([d['gt_f'], d['pred_f'], d['sc_pred_f']],axis=0)
	box = np.concatenate([d['gt_box'], d['pred_box'], d['ref_box']],axis=0)
	mask = np.concatenate([d['gt_mask'], d['pred_mask'], np.abs(d['gt_mask'] - d['pred_mask'])],axis=0)
	que_coord = np.concatenate([d['gt_que_coord'], d['pred_que_coord'], np.abs(d['current_coord'] - d['gt_que_coord'])])
	que_mask = np.concatenate([d['gt_que_mask'], d['pred_que_mask'], np.abs(d['gt_que_mask'] - d['pred_que_mask'])])
	que_mask_visib = np.concatenate([d['gt_que_mask_visib'], d['pred_que_mask_visib'], np.abs(d['gt_que_mask_visib'] - d['pred_que_mask_visib'])])
	


	stitched_image = np.concatenate([image, recon, flow, mask, que_coord, que_mask, que_mask_visib, box], axis=1)

	return stitched_image

def convert_tensor_to_images_scflow_geo_prob(que, ref, flow_pred, RT_refine, gt_flow_r2q, sc_flow_pred, mask_pred, mask_gt, 
										 current_coord, que_coord_pred, que_mask_pred, que_mask_visib_pred, flow_conf_pred, flow_refined_pred, mask_refined_pred, cfg, invalid_flow=400.0, size=[128, 128]):
	ref_image= ref['image'][0]
	que_image = que['image'][0]
	que_image_masked = que_image * que['mask'][0]

	gt_recon_query = apply_forward_flow(gt_flow_r2q, ref['image'], 0.0)
	gt_recon_query = gt_recon_query[0, :, -3:] 
	pred_recon_query = apply_forward_flow(flow_pred, ref['image'], 0.0)
	pred_recon_query = pred_recon_query[0, :, -3:]
	pred_refined_recon_query = apply_forward_flow(flow_refined_pred, ref['image'], 0.0)
	pred_refined_recon_query = pred_refined_recon_query[0, :, -3:] 
	# sc_pred_recon_query = apply_forward_flow(sc_flow_pred, ref['image'], 0.0)
	# sc_pred_recon_query = sc_pred_recon_query[0, :, -3:] 
	
	flow_mask = (gt_flow_r2q[0]!=invalid_flow).to(gt_flow_r2q)
	gt_flow_r2q = flow_to_image(gt_flow_r2q[0] * flow_mask) / 255.0
	pred_flow_r2q = flow_to_image(flow_pred[0] * flow_mask) / 255.0
	pred_refined_flow_r2q = flow_to_image(flow_refined_pred[0] * flow_mask) / 255.0
	# sc_pred_flow_r2q = flow_to_image(sc_flow_pred[0] * flow_mask) / 255.0

	flow_conf_pred = flow_conf_pred[0]
	flow_conf_mask = (flow_conf_pred > 0.5).float()
	flow_conf_pred = flow_conf_pred.repeat(1,3,1,1)
	flow_conf_mask = flow_conf_mask.repeat(1,3,1,1)





	#que_mask = que['mask'][0].repeat(1,3,1,1)
	gt_mask = mask_gt[0].repeat(1,3,1,1)
	pred_mask = mask_pred[0].repeat(1,3,1,1)
	pred_refined_mask = mask_refined_pred[0].repeat(1,3,1,1)

	down_gt_mask = F.interpolate(gt_mask, (32,32), mode='nearest')
	down_pred_refined_mask = F.interpolate(pred_refined_mask, (32, 32), mode='bilinear')
	flow_conf_pred_masked = flow_conf_pred * down_pred_refined_mask
	flow_conf_mask = flow_conf_mask * down_pred_refined_mask

	gt_que_mask = que['mask'][0].repeat(1,3,1,1)
	pred_que_mask = que_mask_pred[0].repeat(1,3,1,1)

	gt_que_mask_visib = que['mask_visib'][0].repeat(1,3,1,1)
	pred_que_mask_visib = que_mask_visib_pred[0].repeat(1,3,1,1)

	gt_que_coord = (que['xyz_origin'][0] + 1)/2 * gt_que_mask
	current_coord = (current_coord[0] + 1)/2
	pred_que_coord = ((que_coord_pred[0] + 1)/2) * pred_que_mask_visib


	gt_box = bbox_3d_visualize(que['RT'], que['K'], ref['source'], que['image'], color=(0, 255, 0))[0]
	pred_box = bbox_3d_visualize(RT_refine, que['K'], ref['source'], que['image'], color=(0, 0, 255))[0]
	ref_box = bbox_3d_visualize(ref['RT'], ref['K_img'], ref['source'], que['image'], color=(255, 0, 0))[0]


	d = {
		'ref_i': ref_image, 
		'que_i': que_image,
		'que_m': que_image_masked,
		'gt_recon_q': gt_recon_query, 
		'pred_recon_q': pred_recon_query, 
		'pred_refined_recon_q': pred_refined_recon_query,
		'gt_f': gt_flow_r2q,
		'pred_f': pred_flow_r2q, 
		'pred_refined_f': pred_refined_flow_r2q,
		'gt_box': gt_box,
		'pred_box': pred_box,
		'ref_box': ref_box,
		'gt_mask': gt_mask,
		'pred_mask': pred_mask,
		'pred_refined_mask': pred_refined_mask,
		'gt_que_mask': gt_que_mask,
		'pred_que_mask': pred_que_mask,
		'gt_que_mask_visib': gt_que_mask_visib,
		'pred_que_mask_visib': pred_que_mask_visib,
		'gt_que_coord': gt_que_coord,
		'pred_que_coord': pred_que_coord,
		'current_coord': current_coord,
		'pred_flow_conf': flow_conf_pred,
		'pred_flow_conf_mask': flow_conf_mask,
		'pred_flow_conf_masked': flow_conf_pred_masked
		}
	
	for k, v in d.items():
		v = F.interpolate(v, size=size, mode='nearest')
		v = np.transpose(v.cpu().numpy(), (0, 2, 3, 1))
		v = np.concatenate(list(v), axis=1)
		d[k] = v

	
	image = np.concatenate([d['ref_i'], d['que_i'], d['que_m']],axis=0)
	recon = np.concatenate([d['gt_recon_q'], d['pred_recon_q'], d['pred_refined_recon_q']],axis=0)
	flow = np.concatenate([d['gt_f'], d['pred_f'], d['pred_refined_f']],axis=0)
	box = np.concatenate([d['gt_box'], d['pred_box'], d['current_coord']],axis=0)
	# mask = np.concatenate([d['gt_mask'], d['pred_mask'], np.abs(d['gt_mask'] - d['pred_mask'])],axis=0)
	mask = np.concatenate([d['gt_mask'], d['pred_mask'], d['pred_refined_mask']],axis=0)
	que_coord = np.concatenate([d['gt_que_coord'], d['pred_que_coord'], np.abs(d['pred_que_coord'] - d['gt_que_coord'])])
	que_mask = np.concatenate([d['gt_que_mask'], d['pred_que_mask'], np.abs(d['gt_que_mask'] - d['pred_que_mask'])])
	que_mask_visib = np.concatenate([d['gt_que_mask_visib'], d['pred_que_mask_visib'], np.abs(d['gt_que_mask_visib'] - d['pred_que_mask_visib'])])
	flow_conf = np.concatenate([d['pred_flow_conf'], d['pred_flow_conf_masked'],d['pred_flow_conf_mask']])
	


	stitched_image = np.concatenate([image, recon, flow, flow_conf, mask, que_coord, que_mask, que_mask_visib, box], axis=1)

	return stitched_image
def convert_tensor_to_images_scflow_base(que, ref, flow_pred, RT_refine, gt_flow_r2q, sc_flow_pred, mask_pred, mask_gt,cfg, invalid_flow=400.0, size=[128, 128]):
	ref_image= ref['image'][0]
	que_image = que['image'][0]
	que_image_masked = que_image * que['mask'][0]

	gt_recon_query = apply_forward_flow(gt_flow_r2q, ref['image'], 0.0)
	gt_recon_query = gt_recon_query[0, :, -3:] 
	#gt_recon_query = (gt_recon_query[0, :, -3:] + 1) / 2 * que['mask'][0].expand_as(que_image)
	#image = gt_recon_query.squeeze().permute(1,2,0).cpu().numpy()
	#image = a[0].squeeze().cpu().numpy()
	#plt.imsave("a.jpg", image)
	pred_recon_query = apply_forward_flow(flow_pred, ref['image'], 0.0)
	pred_recon_query = pred_recon_query[0, :, -3:] 
	#pred_recon_query = (pred_recon_query[0, :, -3:] + 1) / 2 * que['mask'][0].expand_as(que_image)
	sc_pred_recon_query = apply_forward_flow(sc_flow_pred, ref['image'], 0.0)
	sc_pred_recon_query = sc_pred_recon_query[0, :, -3:] 
	
	flow_mask = (gt_flow_r2q[0]!=invalid_flow).to(gt_flow_r2q)
	# gt_flow_r2q = flow_to_image(gt_flow_r2q[0] * flow_mask) / 255.0
	# pred_flow_r2q = flow_to_image(flow_pred[0] * flow_mask) / 255.0
	# sc_pred_flow_r2q = flow_to_image(sc_flow_pred[0] * flow_mask) / 255.0
	gt_flow_r2q = flow_to_image(gt_flow_r2q[0] * flow_mask) / 255.0
	pred_flow_r2q = flow_to_image(flow_pred[0] * flow_mask) / 255.0
	sc_pred_flow_r2q = flow_to_image(sc_flow_pred[0] * flow_mask) / 255.0
	masked_pred_flow_r2q = flow_to_image(flow_pred[0] * mask_gt[0]) / 255.0
	masked_sc_pred_flow_r2q = flow_to_image(sc_flow_pred[0] * mask_gt[0]) / 255.0


	#que_mask = que['mask'][0].repeat(1,3,1,1)
	gt_mask = mask_gt[0].repeat(1,3,1,1)
	pred_mask = mask_pred[0].repeat(1,3,1,1)

	# que_mask_visib = que['mask_visib'][0].repeat(1,3,1,1)
	# pred_mask_visib = mask_visib_pred[0].unsqueeze(0).repeat(1,3,1,1)

	# query_region = make_region(que['xyz'], ref['region_centers'])[0]
	# query_region = region_to_img(query_region).repeat(1,3,1,1)
	# pred_region = region_pred[0].unsqueeze(0)
	# pred_region = region_to_img(pred_region).repeat(1,3,1,1)

	# gt_flowconf = (flowconf_gt[0] * ref['mask'][0]).repeat(1,3,1,1)
	# pred_flowconf = (flowconf_pred[0] * ref['mask'][0]).repeat(1,3,1,1)

	gt_box = bbox_3d_visualize(que['RT'], que['K'], ref['source'], que['image'], color=(0, 255, 0))[0]
	pred_box = bbox_3d_visualize(RT_refine, que['K'], ref['source'], que['image'], color=(0, 0, 255))[0]
	ref_box = bbox_3d_visualize(ref['RT'], ref['K_img'], ref['source'], que['image'], color=(255, 0, 0))[0]


	d = {
		'ref_i': ref_image, 
		'que_i': que_image,
		'que_m': que_image_masked,
		'gt_recon_q': gt_recon_query, 
		'pred_recon_q': pred_recon_query, 
		'sc_pred_recon_q': sc_pred_recon_query,
		'gt_f': gt_flow_r2q,
		'pred_f': pred_flow_r2q, 
		'sc_pred_f': sc_pred_flow_r2q,
		'masked_pred_f': masked_pred_flow_r2q, 
		'masked_sc_pred_f': masked_sc_pred_flow_r2q,
		'gt_box': gt_box,
		'pred_box': pred_box,
		'ref_box': ref_box,
		'gt_mask': gt_mask,
		'pred_mask': pred_mask
		# 'gt_mask_visib': que_mask_visib,
		# 'pred_mask_visib': pred_mask_visib,
		# 'gt_region': query_region,
		# 'pred_region': pred_region,
		# 'gt_flowconf': gt_flowconf,
		# 'pred_flowconf': pred_flowconf
		}
	
	for k, v in d.items():
		v = F.interpolate(v, size=size, mode='nearest')
		v = np.transpose(v.cpu().numpy(), (0, 2, 3, 1))
		v = np.concatenate(list(v), axis=1)
		d[k] = v

	
	image = np.concatenate([d['ref_i'], d['que_i'], d['que_m']],axis=0)
	recon = np.concatenate([d['gt_recon_q'], d['pred_recon_q'], d['sc_pred_recon_q']],axis=0)
	flow = np.concatenate([d['gt_f'], d['pred_f'], d['sc_pred_f']],axis=0)
	masked_flow = np.concatenate([d['gt_f'], d['masked_pred_f'], d['masked_sc_pred_f']],axis=0)
	box = np.concatenate([d['gt_box'], d['pred_box'], d['ref_box']],axis=0)
	mask = np.concatenate([d['gt_mask'], d['pred_mask'], np.abs(d['gt_mask'] - d['pred_mask'])],axis=0)
	


	stitched_image = np.concatenate([image, recon, flow, masked_flow, box, mask], axis=1)

	return stitched_image


def convert_tensor_to_images_scflow_all(que, ref, flow_pred, RT_refine, gt_flow_r2q, sc_flow_pred, mask_pred, mask_gt,cfg, invalid_flow=400.0, size=[128, 128]):
	N = len(ref['image'])
	stitched_images = []
	gt_recon_query = apply_forward_flow(gt_flow_r2q, ref['image'], 0.0)
	pred_recon_query = apply_forward_flow(flow_pred, ref['image'], 0.0)
	sc_pred_recon_query = apply_forward_flow(sc_flow_pred, ref['image'], 0.0)

	gt_box = bbox_3d_visualize(que['RT'], que['K'], ref['source'], que['image'], color=(0, 255, 0))
	pred_box = bbox_3d_visualize(RT_refine, que['K'], ref['source'], que['image'], color=(0, 0, 255))
	ref_box = bbox_3d_visualize(ref['RT'], ref['K_img'], ref['source'], que['image'], color=(255, 0, 0))

	for i in range(N):
		ref_image= ref['image'][i]
		que_image = que['image'][i]
		que_image_masked = que_image * que['mask'][i]

		
		gt_recon_query = gt_recon_query[i, :, -3:] 
		pred_recon_query = pred_recon_query[i, :, -3:] 	
		sc_pred_recon_query = sc_pred_recon_query[i, :, -3:] 
		
		flow_mask = (gt_flow_r2q[i]!=invalid_flow).to(gt_flow_r2q)

		gt_flow_r2q = flow_to_image(gt_flow_r2q[i] * flow_mask) / 255.0
		pred_flow_r2q = flow_to_image(flow_pred[i] * flow_mask) / 255.0
		sc_pred_flow_r2q = flow_to_image(sc_flow_pred[i] * flow_mask) / 255.0
		masked_pred_flow_r2q = flow_to_image(flow_pred[i] * mask_gt[0]) / 255.0
		masked_sc_pred_flow_r2q = flow_to_image(sc_flow_pred[i] * mask_gt[i]) / 255.0


		#que_mask = que['mask'][0].repeat(1,3,1,1)
		gt_mask = mask_gt[i].repeat(1,3,1,1)
		pred_mask = mask_pred[i].repeat(1,3,1,1)

		# que_mask_visib = que['mask_visib'][0].repeat(1,3,1,1)
		# pred_mask_visib = mask_visib_pred[0].unsqueeze(0).repeat(1,3,1,1)

		# query_region = make_region(que['xyz'], ref['region_centers'])[0]
		# query_region = region_to_img(query_region).repeat(1,3,1,1)
		# pred_region = region_pred[0].unsqueeze(0)
		# pred_region = region_to_img(pred_region).repeat(1,3,1,1)

		# gt_flowconf = (flowconf_gt[0] * ref['mask'][0]).repeat(1,3,1,1)
		# pred_flowconf = (flowconf_pred[0] * ref['mask'][0]).repeat(1,3,1,1)

		gt_box = gt_box[i]
		pred_box = pred_box[i]
		ref_box = ref_box[i]


		d = {
			'ref_i': ref_image, 
			'que_i': que_image,
			'que_m': que_image_masked,
			'gt_recon_q': gt_recon_query, 
			'pred_recon_q': pred_recon_query, 
			'sc_pred_recon_q': sc_pred_recon_query,
			'gt_f': gt_flow_r2q,
			'pred_f': pred_flow_r2q, 
			'sc_pred_f': sc_pred_flow_r2q,
			'masked_pred_f': masked_pred_flow_r2q, 
			'masked_sc_pred_f': masked_sc_pred_flow_r2q,
			'gt_box': gt_box,
			'pred_box': pred_box,
			'ref_box': ref_box,
			'gt_mask': gt_mask,
			'pred_mask': pred_mask
			# 'gt_mask_visib': que_mask_visib,
			# 'pred_mask_visib': pred_mask_visib,
			# 'gt_region': query_region,
			# 'pred_region': pred_region,
			# 'gt_flowconf': gt_flowconf,
			# 'pred_flowconf': pred_flowconf
			}
		
		for k, v in d.items():
			v = F.interpolate(v, size=size, mode='nearest')
			v = np.transpose(v.cpu().numpy(), (0, 2, 3, 1))
			v = np.concatenate(list(v), axis=1)
			d[k] = v

		
		image = np.concatenate([d['ref_i'], d['que_i'], d['que_m']],axis=0)
		recon = np.concatenate([d['gt_recon_q'], d['pred_recon_q'], d['sc_pred_recon_q']],axis=0)
		flow = np.concatenate([d['gt_f'], d['pred_f'], d['sc_pred_f']],axis=0)
		masked_flow = np.concatenate([d['gt_f'], d['masked_pred_f'], d['masked_sc_pred_f']],axis=0)
		box = np.concatenate([d['gt_box'], d['pred_box'], d['ref_box']],axis=0)
		mask = np.concatenate([d['gt_mask'], d['pred_mask'], np.abs(d['gt_mask'] - d['pred_mask'])],axis=0)
		


		stitched_image = np.concatenate([image, recon, flow, masked_flow, box, mask], axis=1)

		stitched_images.append(stitched_image)
	return stitched_images

def convert_tensor_to_images_scflow_with_conf(que, ref, flow_pred, RT_refine, gt_flow_r2q, sc_flow_pred, mask_pred, mask_visib_pred, region_pred, flowconf_gt, flowconf_pred, conf, cfg, invalid_flow=400.0, size=[128, 128]):
	d = defaultdict()
	conf_image = conf[0].repeat(3,1,1).unsqueeze(0)


	ref_image= ref['image'][0]
	que_image = que['image'][0]
	que_image_masked = que_image * que['mask'][0]

	# input = ref_image.squeeze().permute(1,2,0).cpu().numpy()
	# segments = slic(input, n_segments=32, compactness=10, sigma=1)
	# segmented_image = label2rgb(segments, input, kind='avg')
	# plt.imsave("segmented_image.jpg", segmented_image)


	gt_recon_query = apply_forward_flow(gt_flow_r2q, ref['image'], 0.0)
	gt_recon_query = gt_recon_query[0]

	#gt_recon_ref_back = apply_backward_flow(gt_flow_r2q, que['image'], 0.0)
	#gt_reon_ref_back = gt_recon_ref_back[0]

	#gt_recon_query = gt_recon_query[0, :, -3:] 
	#gt_recon_query = (gt_recon_query[0, :, -3:] + 1) / 2 * que['mask'][0].expand_as(que_image)
	#image = gt_recon_query.squeeze().permute(1,2,0).cpu().numpy()
	#image = a[0].squeeze().cpu().numpy()
	#plt.imsave("a.jpg", image)
	pred_recon_query = apply_forward_flow(flow_pred, ref['image'], 0.0)
	pred_recon_query = pred_recon_query[0]
	#pred_recon_query = pred_recon_query[0, :, -3:] 
	#pred_recon_query = (pred_recon_query[0, :, -3:] + 1) / 2 * que['mask'][0].expand_as(que_image)
	sc_pred_recon_query = apply_forward_flow(sc_flow_pred, ref['image'], 0.0)
	sc_pred_recon_query = sc_pred_recon_query[0, :, -3:] 

	que_mask = que['mask'][0].repeat(1,3,1,1)
	pred_mask = mask_pred[0].unsqueeze(0).repeat(1,3,1,1)

	que_mask_visib = que['mask_visib'][0].repeat(1,3,1,1)
	pred_mask_visib = mask_visib_pred[0].unsqueeze(0).repeat(1,3,1,1)

	query_region = make_region(que['xyz'], ref['region_centers'])[0]
	query_region = region_to_img(query_region).repeat(1,3,1,1)
	pred_region = region_pred[0].unsqueeze(0)
	pred_region = region_to_img(pred_region).repeat(1,3,1,1)

	gt_flowconf = (flowconf_gt[0] * ref['mask'][0]).repeat(1,3,1,1)
	pred_flowconf = (flowconf_pred[0] * ref['mask'][0]).repeat(1,3,1,1)

	gt_box = bbox_3d_visualize(que['RT'], que['K'], ref['source'], que['image'], color=(0, 255, 0))[0]
	pred_box = bbox_3d_visualize(RT_refine, que['K'], ref['source'], que['image'], color=(0, 0, 255))[0]
	ref_box = bbox_3d_visualize(ref['RT'], ref['K_img'], ref['source'], que['image'], color=(255, 0, 0))[0]

	# ref_region = make_region(ref['xyz'], ref['region_centers'])[0]
	# ref_region = region_to_img(ref_region).repeat(1,3,1,1)
	# recon_region = apply_forward_flow(gt_flow_r2q, ref_region.unsqueeze(0).expand(gt_flow_r2q.shape[0], -1, -1, -1, -1), 0.0)
	# recon_region = recon_region[0]

	flow_mask = (gt_flow_r2q[0]!=invalid_flow).to(gt_flow_r2q)
	gt_flow_r2q = flow_to_image(gt_flow_r2q[0] * flow_mask) / 255.0
	pred_flow_r2q = flow_to_image(flow_pred[0] * flow_mask) / 255.0
	sc_pred_flow_r2q = flow_to_image(sc_flow_pred[0] * flow_mask) / 255.0

	d = {
		'ref_i': ref_image, 
		'que_i': que_image,
		'que_m': que_image_masked,
		'gt_recon_q': gt_recon_query, 
		'pred_recon_q': pred_recon_query, 
		'sc_pred_recon_q': sc_pred_recon_query,
		'gt_f': gt_flow_r2q,
		'pred_f': pred_flow_r2q, 
		'sc_pred_f': sc_pred_flow_r2q,
		'gt_box': gt_box,
		'pred_box': pred_box,
		'ref_box': ref_box,
		'gt_mask': que_mask,
		'pred_mask': pred_mask,
		'gt_mask_visib': que_mask_visib,
		'pred_mask_visib': pred_mask_visib,
		'gt_region': query_region,
		'pred_region': pred_region,
		'gt_flowconf': gt_flowconf,
		'pred_flowconf': pred_flowconf,
		'conf_i': conf_image
		}
	
	for k, v in d.items():
		v = F.interpolate(v, size=size, mode='nearest')
		v = np.transpose(v.cpu().numpy(), (0, 2, 3, 1))
		v = np.concatenate(list(v), axis=1)
		d[k] = v

	
	image = np.concatenate([d['ref_i'], d['que_i'], d['que_m']],axis=0)
	recon = np.concatenate([d['gt_recon_q'], d['pred_recon_q'], d['sc_pred_recon_q']],axis=0)
	flow = np.concatenate([d['gt_f'], d['pred_f'], d['sc_pred_f']],axis=0)
	box = np.concatenate([d['gt_box'], d['pred_box'], d['ref_box']],axis=0)
	mask = np.concatenate([d['gt_mask'], d['pred_mask'], np.abs(d['gt_mask'] - d['pred_mask'])],axis=0)
	mask_visib = np.concatenate([d['gt_mask_visib'], d['pred_mask_visib'], np.abs(d['gt_mask_visib'] - d['pred_mask_visib'])],axis=0)
	region = np.concatenate([d['gt_region'], d['pred_region'], np.abs(d['gt_region'] - d['pred_region'])], axis=0)
	flowconf = np.concatenate([d['gt_flowconf'], d['pred_flowconf'], np.abs(d['gt_flowconf'] - d['pred_flowconf'])], axis=0)
	conf = np.concatenate([d['conf_i'], np.abs(d['gt_f'] - d['pred_f']), np.zeros_like(d['conf_i'])], axis=0)


	stitched_image = np.concatenate([image, recon, flow, box, mask, mask_visib, region, flowconf, conf], axis=1)

	return stitched_image

def convert_tensor_to_images_scflow(que, ref, flow_pred, RT_refine, gt_flow_r2q, sc_flow_pred, mask_pred, mask_visib_pred, region_pred, flowconf_gt, flowconf_pred, cfg, invalid_flow=400.0, size=[128, 128]):
	d = defaultdict()
	ref_image= ref['image'][0]
	que_image = que['image'][0]
	que_image_masked = que_image * que['mask'][0]

	gt_recon_query = apply_forward_flow(gt_flow_r2q, ref['image'], 0.0)
	gt_recon_query = gt_recon_query[0, :, -3:] 
	#gt_recon_query = (gt_recon_query[0, :, -3:] + 1) / 2 * que['mask'][0].expand_as(que_image)
	#image = gt_recon_query.squeeze().permute(1,2,0).cpu().numpy()
	#image = a[0].squeeze().cpu().numpy()
	#plt.imsave("a.jpg", image)
	pred_recon_query = apply_forward_flow(flow_pred, ref['image'], 0.0)
	pred_recon_query = pred_recon_query[0, :, -3:] 
	#pred_recon_query = (pred_recon_query[0, :, -3:] + 1) / 2 * que['mask'][0].expand_as(que_image)
	sc_pred_recon_query = apply_forward_flow(sc_flow_pred, ref['image'], 0.0)
	sc_pred_recon_query = sc_pred_recon_query[0, :, -3:] 
	
	flow_mask = (gt_flow_r2q[0]!=invalid_flow).to(gt_flow_r2q)
	gt_flow_r2q = flow_to_image(gt_flow_r2q[0] * flow_mask) / 255.0
	pred_flow_r2q = flow_to_image(flow_pred[0] * flow_mask) / 255.0
	sc_pred_flow_r2q = flow_to_image(sc_flow_pred[0] * flow_mask) / 255.0

	que_mask = que['mask'][0].repeat(1,3,1,1)
	pred_mask = mask_pred[0].unsqueeze(0).repeat(1,3,1,1)

	que_mask_visib = que['mask_visib'][0].repeat(1,3,1,1)
	pred_mask_visib = mask_visib_pred[0].unsqueeze(0).repeat(1,3,1,1)

	query_region = make_region(que['xyz'], ref['region_centers'])[0]
	query_region = region_to_img(query_region).repeat(1,3,1,1)
	pred_region = region_pred[0].unsqueeze(0)
	pred_region = region_to_img(pred_region).repeat(1,3,1,1)

	gt_flowconf = (flowconf_gt[0] * ref['mask'][0]).repeat(1,3,1,1)
	pred_flowconf = (flowconf_pred[0] * ref['mask'][0]).repeat(1,3,1,1)

	gt_box = bbox_3d_visualize(que['RT'], que['K'], ref['source'], que['image'], color=(0, 255, 0))[0]
	pred_box = bbox_3d_visualize(RT_refine, que['K'], ref['source'], que['image'], color=(0, 0, 255))[0]
	ref_box = bbox_3d_visualize(ref['RT'], ref['K_img'], ref['source'], que['image'], color=(255, 0, 0))[0]


	d = {
		'ref_i': ref_image, 
		'que_i': que_image,
		'que_m': que_image_masked,
		'gt_recon_q': gt_recon_query, 
		'pred_recon_q': pred_recon_query, 
		'sc_pred_recon_q': sc_pred_recon_query,
		'gt_f': gt_flow_r2q,
		'pred_f': pred_flow_r2q, 
		'sc_pred_f': sc_pred_flow_r2q,
		'gt_box': gt_box,
		'pred_box': pred_box,
		'ref_box': ref_box,
		'gt_mask': que_mask,
		'pred_mask': pred_mask,
		'gt_mask_visib': que_mask_visib,
		'pred_mask_visib': pred_mask_visib,
		'gt_region': query_region,
		'pred_region': pred_region,
		'gt_flowconf': gt_flowconf,
		'pred_flowconf': pred_flowconf
		}
	
	for k, v in d.items():
		v = F.interpolate(v, size=size, mode='nearest')
		v = np.transpose(v.cpu().numpy(), (0, 2, 3, 1))
		v = np.concatenate(list(v), axis=1)
		d[k] = v

	
	image = np.concatenate([d['ref_i'], d['que_i'], d['que_m']],axis=0)
	recon = np.concatenate([d['gt_recon_q'], d['pred_recon_q'], d['sc_pred_recon_q']],axis=0)
	flow = np.concatenate([d['gt_f'], d['pred_f'], d['sc_pred_f']],axis=0)
	box = np.concatenate([d['gt_box'], d['pred_box'], d['ref_box']],axis=0)
	mask = np.concatenate([d['gt_mask'], d['pred_mask'], np.abs(d['gt_mask'] - d['pred_mask'])],axis=0)
	mask_visib = np.concatenate([d['gt_mask_visib'], d['pred_mask_visib'], np.abs(d['gt_mask_visib'] - d['pred_mask_visib'])],axis=0)
	region = np.concatenate([d['gt_region'], d['pred_region'], np.abs(d['gt_region'] - d['pred_region'])], axis=0)
	flowconf = np.concatenate([d['gt_flowconf'], d['pred_flowconf'], np.abs(d['gt_flowconf'] - d['pred_flowconf'])], axis=0)


	stitched_image = np.concatenate([image, recon, flow, box, mask, mask_visib, region, flowconf], axis=1)

	return stitched_image

def convert_tensor_to_images(que, ref, flow_pred, RT_1, RT_refine, gt_flow_r2q, cfg, invalid_flow=400.0, size=[128, 128]):
	d = defaultdict()
	ref_image, ref_geometry = ref['image'][0], (ref['geo'][0, :, -3:] + 1) / 2
	que_image, que_geometry = que['image'][0], (que['geo'][0, :, -3:] + 1) / 2
	que_geometry_masked = que_geometry * que['mask'][0]

	gt_recon_geo = apply_forward_flow(gt_flow_r2q, ref['geo'], -1.0)

	gt_recon_geometry = (gt_recon_geo[0, :, -3:] + 1) / 2 * que['mask'][0].expand_as(que_image)
	pred_recon_geometry = (flow_pred['recon_geo'][0, :, -3:].detach() + 1) / 2 * que['mask'][0].expand_as(que_image)
	pred_geometry = (flow_pred['geo'][0, :, -3:].detach() + 1) / 2 * que['mask'][0].expand_as(que_image)

	flow_mask = (gt_flow_r2q[0]!=invalid_flow).to(gt_flow_r2q)
	gt_flow_r2q = flow_to_image(gt_flow_r2q[0] * flow_mask) / 255.0
	pred_flow_r2q = flow_to_image(flow_pred['flow_r2q'][-1][0] * flow_mask) / 255.0

	initial_geometry = (render_geometry(ref['source'], que['K_img'], 
		RT_1, cfg.img_size, cfg.img_size, cfg.N_z, cfg.N_freq, cfg.represent_mode)[0, :, -3:].detach() + 1) / 2
	refined_geometry = (render_geometry(ref['source'], que['K_img'], 
		RT_refine, cfg.img_size, cfg.img_size, cfg.N_z, cfg.N_freq, cfg.represent_mode)[0, :, -3:].detach() + 1) / 2

	d = {
		'ref_i': ref_image, 
		'ref_g': ref_geometry, 
		'que_i': que_image,
		'que_g': que_geometry, 
		'que_gm': que_geometry_masked,
		'gt_f': gt_flow_r2q,
		'gt_recon_g': gt_recon_geometry, 
		'pred_f': pred_flow_r2q, 
		'pred_recon_g': pred_recon_geometry, 
		'pred_g': pred_geometry, 
		'initial_g': initial_geometry,
		'refined_g': refined_geometry}
	
	for k, v in d.items():
		v = F.interpolate(v, size=size, mode='nearest')
		v = np.transpose(v.cpu().numpy(), (0, 2, 3, 1))
		v = np.concatenate(list(v), axis=1)
		d[k] = v

	reference = np.concatenate([d['ref_i'], d['ref_g'], np.zeros_like(d['ref_i'])], axis=0)
	query = np.concatenate([d['que_i'], d['que_g'], np.zeros_like(d['que_i'])], axis=0)
	flow = np.concatenate([np.abs(d['gt_f']-d['pred_f']), d['gt_f'], d['pred_f']], axis=0)
	recon_geo = np.concatenate([np.abs(d['gt_recon_g']-d['pred_recon_g']), d['gt_recon_g'], d['pred_recon_g']], axis=0)
	pred_geo = np.concatenate([np.abs(d['que_gm']-d['pred_g']), d['que_gm'], d['pred_g']], axis=0)
	initial_geo = np.concatenate([np.abs(d['que_g']-d['initial_g']), d['que_g'], d['initial_g']], axis=0)
	refined_geo = np.concatenate([np.abs(d['que_g']-d['refined_g']), d['que_g'], d['refined_g']], axis=0)

	stitched_image = np.concatenate([reference, query, flow, recon_geo, pred_geo, initial_geo, refined_geo], axis=1)

	return stitched_image


def convert_attn_to_images(ref_image, que_image, attn_l, attn_p, size=[128, 128]):
	attns = []
	attns.append(stitch_attn(ref_image, que_image, attn_l[0], size))
	for layer, attn in attn_p.items():
		attns.append(stitch_attn(ref_image, que_image, attn[0], size))
	stitched_attn = np.concatenate(attns, axis=0)
	return stitched_attn

def stitch_attn(ref_image, que_image, attn, size):
	n_inp = ref_image.shape[0]
	n_que = que_image.shape[0]
	n_patch_x, n_patch_y = int(np.sqrt(attn.shape[0] // n_que)), int(np.sqrt(attn.shape[0] // n_que))
	ref_image = F.interpolate(ref_image, size=size, mode='bilinear', align_corners=True)
	que_image = F.interpolate(que_image, size=size, mode='bilinear', align_corners=True)
	que_image = np.transpose(que_image.cpu().numpy(), (0, 2, 3, 1))[0]
	ref_image = np.transpose(ref_image.cpu().numpy(), (0, 2, 3, 1))
	que_image[size[0]//2-2:size[0]//2+2, size[1]//2-2:size[1]//2+2] = [1, 0, 0]   # attended point
	attn_quecenter = attn.reshape(n_que, n_patch_y, n_patch_x, n_inp, n_patch_y, n_patch_x)[0, n_patch_y//2, n_patch_x//2]
	attn_quecenter = attn_quecenter - attn_quecenter.min()
	attn_quecenter = attn_quecenter / attn_quecenter.max()
	vis = []
	for i in range(n_inp):
		attn_inp = attn_quecenter[i].detach().cpu().numpy()
		attn_inp = cv2.resize(attn_inp, size)
		vis.append(attn_inp[..., np.newaxis]*3/4 + ref_image[i]/4)
	vis += [que_image]
	stitched_image = np.concatenate(vis, axis=1)
	return stitched_image

def contour_image(geometry, image, size=[128, 128]):
	geometry = (geometry + 1).abs()[0]
	image = image[0]
	geometry = F.interpolate(geometry, size=size, mode='nearest')
	image = F.interpolate(image, size=size, mode='nearest')
	contour_image = contour(geometry.permute(0, 2, 3, 1).detach().cpu().numpy()[0], 
							image.permute(0, 2, 3, 1).detach().cpu().numpy()[0])
	return contour_image

def contour(render, img, color=(255, 255, 255)):
	img = ((img - np.min(img))/(np.max(img) - np.min(img)) * 255).astype(np.uint8).copy() # convert to contiguous array
	render = ((render - np.min(render))/(np.max(render) - np.min(render)) * 255).astype(np.uint8)
	render = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
	_, thr = cv2.threshold(render, 2, 255, 0)
	contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	result = cv2.drawContours(img, contours, -1, color, 2)
	return result


def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
	"""Draw 3d bounding box in image
	qs: (8,2), projected 3d points array of vertices for the 3d box in following order:
	  7 -------- 6
	 /|         /|
	4 -------- 5 .
	| |        | |
	. 3 -------- 2
	|/         |/
	0 -------- 1
	"""
	# Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py
	image = np.ascontiguousarray(image * 255, dtype=np.uint8)
	qs = qs.astype(np.int32)
	_bottom_color = _middle_color = _top_color = color
	for k in range(0, 4):
		# Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
		i, j = k + 4, (k + 1) % 4 + 4
		cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), _bottom_color, thickness, cv2.LINE_AA)
		i, j = k, k + 4
		cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), _middle_color, thickness, cv2.LINE_AA)
		i, j = k, (k + 1) % 4
		cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), _top_color, thickness, cv2.LINE_AA)
	return image / 255.0

def bbox_3d_visualize(RT, K, structure, image, color=(0, 255, 0)):
	device = image.device
	image = image[0,0].permute(1, 2, 0).detach().cpu().numpy()
	K = K[0,0].detach().cpu().numpy()
	RT = RT[0,0].detach().cpu().numpy()
	corners_3d = get_3d_bbox_from_pts(structure[0].verts_list()[0])
	corners_2d = project_pts(corners_3d, K, RT[:3, :3], RT[:3, [3]])
	image = draw_projected_box3d(image, corners_2d, color)
	image = torch.tensor(image, device=device).permute(2, 0, 1).unsqueeze(0).unsqueeze(1)
	return image