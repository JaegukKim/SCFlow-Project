import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys
sys.path.append('bop_toolkit')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
from HGPose.dataset.BOP_SPECIFIC_CHALLENGE_dataset_scflow_posecnn_bopdet import get_dataset as challenge_get_dataset
from HGPose.dataset.BOP_SPECIFIC_scflow import get_dataset, OcclusionAugmentation, ImageAugmentation, worker_init_fn, get_least_loaded_cores
# from HGPose.model.flowmodel import FlowModel
# from HGPose.model.relative_pose_model import RelativePoseEstimator
# from HGPose.model.coarse_model import CoarseModel
# from HGPose.model.refine_model import RefineModel
# from HGPose.utils.image import convert_tensor_to_images, convert_attn_to_images
# from HGPose.utils.geometry import render_geometry, get_flow_from_delta_pose_and_xyz, apply_forward_flow
# from HGPose.utils.metric import total_metric, challenge_format
# from HGPose.model.maml import fast_adapt
# import HGPose.utils.error as E
from HGPose.model.scflow_base import SCFlow
from HGPose.utils.model import load_model
from HGPose.utils.loss import grid_matching_loss, sequence_loss, sequence_loss2, flow_loss, flow_loss_with_mask
from HGPose.utils.flow import filter_flow_by_mask, get_mask_by_flow

from HGPose.utils.image import convert_tensor_to_images_scflow, convert_attn_to_images, region_to_img, convert_tensor_to_images_scflow_with_conf, convert_tensor_to_images_scflow_base
from HGPose.utils.geometry import xyz2pose, get_distmap, render_geometry, get_separate_medoid, make_region, get_region_smoothness_and_variation
from HGPose.utils.metric import total_metric
import HGPose.utils.error as E
from HGPose.utils.util import str2bool

from HGPose.utils.util import str2bool
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from time import time
import csv
import numpy as np
from HGPose.utils.dataset import load_json_fast
from collections import OrderedDict
import learn2learn as l2l
import subprocess
from HGPose.utils.geometry import xyz2pose


def infer_add(model, dataloader, visdir, obj_list):

	
	# coarse_model.load_state_dict(load_dict['coarse_model_state_dict'])
	# flow_model.load_state_dict(load_dict['flow_model_state_dict'])
	# pose_model.load_state_dict(load_dict['pose_model_state_dict'])
	# refine_model.load_state_dict(load_dict['refine_model_state_dict'])
	
	# coarse_model.eval()
	# flow_model.eval()
	# pose_model.eval()
	# refine_model.eval()
	model.eval()
	results = defaultdict(list)
	# challenge = []
	# pose_times = defaultdict(int)
	# detection_times = defaultdict(int)

	obj_idx = np.ones_like(obj_list)
	for i, (ids, que) in enumerate(tqdm(dataloader)):
		# if i > 10: break
		bsz = que['image'].shape[0]
		start = time()
		que = {k: v.cuda(non_blocking=True) for k, v in que.items()}
		obj = dataloader.dataset.batch_object(ids['obj_id'])
		
		# que['K'] = que['K_img']
		que, ref, flow_gt = dataloader.dataset.preprocess_batch(que, obj)
	
		# ref = dataloader.dataset.batch_coarse_reference(ids, que, device=que['image'].device)
		# que, ref, flow_gt = dataloader.dataset.preprocess_batch(que,ref)
		mask_occ, mask_self_occ, mask_external_occ, mask_invalid, mask_valid = get_mask_by_flow(flow_gt.squeeze(1), que['mask'].squeeze(1).squeeze(1), que['mask_visib'].squeeze(1).squeeze(1), ref['mask'].squeeze(1).squeeze(1))
		flow_gt1 = filter_flow_by_mask(flow_gt.squeeze(1), mask_valid.squeeze(1)).unsqueeze(1)
		#flow_gt2 = filter_flow_by_mask(flow_gt.squeeze(1), ref['mask'].squeeze(1).squeeze(1)).unsqueeze(1)
		#flow_gt, mask_gt = filter_flow_by_mask(flow_gt.squeeze(1), que['mask_visib'].squeeze(1).squeeze(1))
		mask_occ, mask_self_occ, mask_external_occ, mask_invalid, mask_valid = mask_occ.unsqueeze(1), mask_self_occ.unsqueeze(1), mask_external_occ.unsqueeze(1), mask_invalid.unsqueeze(1), mask_valid.unsqueeze(1)
		mask_gt1 = mask_valid
		#mask_gt2 = ref['mask']

		flow_gt = flow_gt1
		mask_gt = mask_gt1

		# ref['mask'] = (ref['image'] > 0.0).to(torch.float).mean(dim=-3, keepdims=True)
		obj_ids = torch.tensor(ids['obj_id']).to(flow_gt).to(torch.int)
		
		flow_preds, pose_preds, sc_flow_preds, mask_preds = model(ref['image'].squeeze(1), que['image'].squeeze(1), ref['K_img'].squeeze(1), que['K'].squeeze(1), 
						 ref['RT'].squeeze(1), ref['xyz'].squeeze(1), ref['mask'].squeeze(1), flow_gt.squeeze(1), mask_gt.squeeze(1), obj_ids, 8)
		pose_preds = [pp.unsqueeze(1) for pp in pose_preds]
		flow_preds = [fp.unsqueeze(1) for fp in flow_preds]
		sc_flow_preds = [scfp.unsqueeze(1) for scfp in sc_flow_preds]
		mask_preds = [mp.unsqueeze(1) for mp in mask_preds]
		pose_pred = pose_preds[-1]
		flow_pred = flow_preds[-1]
		sc_flow_pred = sc_flow_preds[-1]
		mask_pred = mask_preds[-1]

		results['adds_1'] += E.ADDS(ref['RT'][:, 0], que['RT'][:, 0], obj['pts'], obj['symmetry'], obj['scale'])
		results['adds_2'] += E.ADDS(pose_pred[:, 0], que['RT'][:, 0], obj['pts'], obj['symmetry'], obj['scale'])
		results['obj_id'] += ids['obj_id'].tolist()
		#results['obj_id'] += ids['obj_id']
		results['diameter'] += [s * d for s, d in zip(obj['scale'], obj['diameter'])]
		
		# results['adds_1'] += E.ADDS(pred['RT_1'][:, 0], que['RT'][:, 0], obj['pts'], obj['symmetry'], obj['scale'])
		# results['adds_2'] += E.ADDS(pred['RT_2'][:, 0], que['RT'][:, 0], obj['pts'], obj['symmetry'], obj['scale'])
		# results['obj_id'] += ids['obj_id'].tolist()
		# results['diameter'] += [s * d for s, d in zip(obj['scale'], obj['diameter'])]
		# if (i == random_seed):
		obj_id = ids['obj_id'][0].item()
		stitched_images = convert_tensor_to_images_scflow_base(que, ref, flow_pred, pose_pred, flow_gt, sc_flow_pred, mask_pred, mask_gt, cfg)
		clipped_stitched_images = np.clip(stitched_images, a_min=0.0, a_max=1.0)
		plt.imsave(os.path.join(visdir,str(ids['scene_id'].item()) + '_' + str(ids['im_id'].item())+'_'+str(ids['inst_id'].item()) + '_' + str(round(results['adds_2'][i] / results['diameter'][i], 3))+".jpg"), clipped_stitched_images)
		#plt.imsave(os.path.join(visdir,str(obj_id) + '_' + str(obj_idx[obj_id-1])+'_'+str(round(results['adds_2'][i] / results['diameter'][i], 3))+".jpg"), clipped_stitched_images)
		obj_idx[obj_id-1] += 1

		# 	writer.add_image("validation results", stitched_image, steps, dataformats='HWC')
		# if cfg.debug and i==10:
		# 	break
	metrics = total_metric(results['obj_id'], results['diameter'], results['adds_1'], results['adds_2'])
	adds_10 = metrics['adds_2_10']
	
	print(adds_10)



		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config-path', type=str, required=True)
	parser.add_argument('--debug', type=str2bool, default=False, required=False)
	args = parser.parse_args()
	cfg = OmegaConf.load(args.config_path)
	cfg.debug = args.debug

	test_dataset = challenge_get_dataset(cfg.dataset)
	
	least_loaded_cores = get_least_loaded_cores(0)
	custom_init_fn = partial(worker_init_fn, least_loaded_cores=least_loaded_cores)
	dataloader = DataLoader(
		dataset=test_dataset, 
		batch_size=1, 
		shuffle=False, 
		num_workers=8, 
		collate_fn=test_dataset.collate_fn,
		worker_init_fn=custom_init_fn)

	
	model = SCFlow(cfg.model).cuda()
	load_dict = torch.load(cfg.infer.load_path, map_location=torch.device('cuda'))
	model.load_state_dict(load_dict['model_state_dict'])

	visdir = os.path.join(*cfg.infer.load_path.split('/')[:-2], 'infer_vis_posecnn_bopdet')
	os.makedirs(visdir, exist_ok=True)

	with torch.no_grad():
		infer_add(model, dataloader, visdir, cfg.dataset.obj_list)
