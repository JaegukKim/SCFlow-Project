import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys
sys.path.append('bop_toolkit')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.modules.batchnorm import SyncBatchNorm
from torch.multiprocessing import spawn

from HGPose.dataset.BOP_SPECIFIC_scflow import get_dataset, OcclusionAugmentation, ImageAugmentation, worker_init_fn, get_least_loaded_cores
from HGPose.dataset.BOP_SPECIFIC_CHALLENGE_dataset_scflow import get_dataset as challenge_get_dataset
from HGPose.model.scflow_base import SCFlow
from HGPose.utils.model import load_model
from HGPose.utils.loss import grid_matching_loss, sequence_loss, sequence_loss2, flow_loss, flow_loss_with_mask, DisentanglePointMatchingLoss
from HGPose.utils.flow import filter_flow_by_mask, get_mask_by_flow

from HGPose.utils.image import convert_tensor_to_images_scflow, convert_attn_to_images, region_to_img, convert_tensor_to_images_scflow_with_conf, convert_tensor_to_images_scflow_base
from HGPose.utils.geometry import xyz2pose, get_distmap, render_geometry, get_separate_medoid, make_region, get_region_smoothness_and_variation
from HGPose.utils.metric import total_metric
import HGPose.utils.error as E
from HGPose.utils.util import str2bool

from functools import partial, partialmethod
import builtins
from collections import defaultdict
from time import time
import random

# from scripts.infer import infer
@torch.no_grad()
def validate_challenge_add(cfg, model, val_dataloader, steps, writer):
	model.eval()
	results = defaultdict(list)
	random_seed = random.randint(0, len(val_dataloader)-1)
	for i, (ids, que) in enumerate(tqdm(val_dataloader)):
		
		que = {k: v.cuda(non_blocking=True) for k, v in que.items()}
		ref = val_dataloader.dataset.batch_coarse_reference(ids, que, device=que['image'].device)   
		#ref = val_dataloader.dataset.batch_reference(ids, que) 
		que, ref, flow_gt = val_dataloader.dataset.preprocess_batch(que, ref)
		#mask_gt[:,0,0,:,:]
		mask_occ, mask_self_occ, mask_external_occ, mask_invalid, mask_valid = get_mask_by_flow(flow_gt.squeeze(1), que['mask'].squeeze(1).squeeze(1), que['mask_visib'].squeeze(1).squeeze(1), ref['mask'].squeeze(1).squeeze(1))
		flow_gt1 = filter_flow_by_mask(flow_gt.squeeze(1), mask_valid.squeeze(1)).unsqueeze(1)
		flow_gt2 = filter_flow_by_mask(flow_gt.squeeze(1), ref['mask'].squeeze(1).squeeze(1)).unsqueeze(1)
		#flow_gt, mask_gt = filter_flow_by_mask(flow_gt.squeeze(1), que['mask_visib'].squeeze(1).squeeze(1))
		mask_occ, mask_self_occ, mask_external_occ, mask_invalid, mask_valid = mask_occ.unsqueeze(1), mask_self_occ.unsqueeze(1), mask_external_occ.unsqueeze(1), mask_invalid.unsqueeze(1), mask_valid.unsqueeze(1)
		mask_gt1 = mask_valid
		mask_gt2 = ref['mask']

		flow_gt = flow_gt2
		mask_gt = mask_gt2

		obj_ids = torch.tensor(ids['obj_id']).to(flow_gt).to(torch.int)
		que['K'] = que['K_img']
		flow_preds, pose_preds, sc_flow_preds, mask_preds = model(ref['image'].squeeze(1), que['image'].squeeze(1), ref['K_img'].squeeze(1), que['K'].squeeze(1), 
						 ref['RT'].squeeze(1), ref['geo'].squeeze(1), ref['mask'].squeeze(1), flow_gt.squeeze(1), mask_gt.squeeze(1), obj_ids, 8)
		pose_preds = [pp.unsqueeze(1) for pp in pose_preds]
		flow_preds = [fp.unsqueeze(1) for fp in flow_preds]
		sc_flow_preds = [scfp.unsqueeze(1) for scfp in sc_flow_preds]
		mask_preds = [mp.unsqueeze(1) for mp in mask_preds]
		pose_pred = pose_preds[-1]
		flow_pred = flow_preds[-1]
		sc_flow_pred = sc_flow_preds[-1]
		mask_pred = mask_preds[-1]

		results['adds_1'] += E.ADDS(ref['RT'][:, 0], que['RT'][:, 0], ref['pts'], ref['symmetry'], ref['scale'])
		results['adds_2'] += E.ADDS(pose_pred[:, 0], que['RT'][:, 0], ref['pts'], ref['symmetry'], ref['scale'])
		#results['obj_id'] += ids['obj_id'].tolist()
		results['obj_id'] += ids['obj_id']
		results['diameter'] += [s * d for s, d in zip(ref['scale'], ref['diameter'])]
		
		# results['adds_1'] += E.ADDS(pred['RT_1'][:, 0], que['RT'][:, 0], obj['pts'], obj['symmetry'], obj['scale'])
		# results['adds_2'] += E.ADDS(pred['RT_2'][:, 0], que['RT'][:, 0], obj['pts'], obj['symmetry'], obj['scale'])
		# results['obj_id'] += ids['obj_id'].tolist()
		# results['diameter'] += [s * d for s, d in zip(obj['scale'], obj['diameter'])]
		if (i == random_seed):
			stitched_image = convert_tensor_to_images_scflow_base(que, ref, flow_pred, pose_pred, flow_gt, sc_flow_pred, mask_pred, mask_gt, cfg)
			writer.add_image("validation results", stitched_image, steps, dataformats='HWC')
		if cfg.debug and i==10:
			break
	metrics = total_metric(results['obj_id'], results['diameter'], results['adds_1'], results['adds_2'])
	writer.add_scalar('validation/ref ADDS_mean', metrics["adds_1_mean"], steps)
	writer.add_scalar('validation/pred ADDS_mean', metrics["adds_2_mean"], steps)
	writer.add_scalar('validation/ref ADD_S_0.1', metrics["adds_1_10"], steps)
	writer.add_scalar('validation/pred ADD_S_0.1', metrics["adds_2_10"], steps)
	writer.add_scalar('validation/pred ADD_S_0.05', metrics["adds_2_05"], steps)
	writer.add_scalar('validation/pred ADD_S_0.02', metrics["adds_2_02"], steps)
	writer.add_scalar('validation/pred ADD_S_AUC', metrics["adds_2_auc"], steps)
	
	

	return metrics['adds_2_10'], results



def train(model, train_dataloader, val_dataloader, cfg):
	torch.autograd.set_detect_anomaly(True)

	optim = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr)
	if cfg.optim.mode == 'cosine':
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=cfg.optim.update_after, T_mult=1)
	else:
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, cfg.optim.update_after, gamma=cfg.optim.gamma)
	gradscaler = torch.cuda.amp.GradScaler(enabled=cfg.use_mixed_precision)
	exp_dir = os.path.join('saved', cfg.exp_tag)
	logdir = os.path.join(exp_dir, 'logs')
	ckpt_dir = os.path.join(exp_dir, 'ckpts')
	ckpt_path = os.path.join(ckpt_dir, 'latest.pt')
	val_dir = os.path.join(exp_dir, 'valid_score')
	os.makedirs(exp_dir, exist_ok=True)
	os.makedirs(ckpt_dir, exist_ok=True)
	os.makedirs(val_dir, exist_ok=True)
	writer = SummaryWriter(logdir=logdir) if rank == 0 else None
	steps = 1
	best_score = 0

	if cfg.load_path is not None:
		steps, best_score = load_model(cfg.load_path, model, optim, lr_scheduler, gradscaler)
		steps += 1
		print(f'best_score: {best_score}')
	else:
		if cfg.raft_load_path is not None:
			print("loading raft model...")
			raft_dict = torch.load(cfg.raft_load_path)
			model_dict = model.state_dict()
			#pretrained_dict = {k: v for k, v in raft_dict.items() if k in model_dict.keys()}
			#pretrained_dict = {k: v for k, v in raft_dict.items() if k in model_dict.keys() and 'update_block' not in k}
			#pretrained_dict = {k: v for k, v in raft_dict.items() if k in model_dict.keys() and 'flow_head' not in k}
			pretrained_dict = {k: v for k, v in raft_dict.items() if k in model_dict.keys() and ('flow_head' not in k and 'recurrent_block' not in k)}
			model_dict.update(pretrained_dict)
			model.load_state_dict(model_dict)
	#occaug = OcclusionAugmentation(p=0.5)
	imgaug = ImageAugmentation()
	l1 = torch.nn.L1Loss(reduce=False)
	ce = torch.nn.CrossEntropyLoss()
	model.train()
	start = time()
	#first_batch = next(iter(train_dataloader))
	for i, (ids, que) in enumerate(train_dataloader):
		#ids,que = first_batch
		que = {k: v.cuda(non_blocking=True) for k, v in que.items()}
		ref = train_dataloader.dataset.batch_reference(ids, que)   

		#que['image'], que['mask_visib'] = occaug(que['image'], que['mask_visib'], ids['obj_id']) if cfg.occlusion_augmentation else (que['image'], que['mask_visib'])
		que['image'] = imgaug(que['image'])
		que, ref, flow_gt = train_dataloader.dataset.preprocess_batch(que, ref)

		#import matplotlib.pyplot as plt
		#from torchvision.utils import flow_to_image
		#plt.imsave("mask.jpg",mask_gt[0].squeeze().cpu().numpy(), cmap="gray")
		#plt.imsave("ref.jpg", ref['image'][0][0].permute(1,2,0).cpu().numpy())
		mask_occ, mask_self_occ, mask_external_occ, mask_invalid, mask_valid = get_mask_by_flow(flow_gt.squeeze(1), que['mask'].squeeze(1).squeeze(1), que['mask_visib'].squeeze(1).squeeze(1), ref['mask'].squeeze(1).squeeze(1))
		flow_gt1 = filter_flow_by_mask(flow_gt.squeeze(1), mask_valid.squeeze(1)).unsqueeze(1)
		# flow_gt2 = filter_flow_by_mask(flow_gt.squeeze(1), ref['mask'].squeeze(1).squeeze(1)).unsqueeze(1)
		#flow_gt, mask_gt = filter_flow_by_mask(flow_gt.squeeze(1), que['mask_visib'].squeeze(1).squeeze(1))
		mask_occ, mask_self_occ, mask_external_occ, mask_invalid, mask_valid = mask_occ.unsqueeze(1), mask_self_occ.unsqueeze(1), mask_external_occ.unsqueeze(1), mask_invalid.unsqueeze(1), mask_valid.unsqueeze(1)
		mask_gt1 = mask_valid
		# mask_gt2 = ref['mask']

		flow_gt = flow_gt1
		mask_gt = mask_gt1

		obj_ids = torch.tensor(ids['obj_id']).to(flow_gt).to(torch.int)
		optim.zero_grad()
		with torch.cuda.amp.autocast(enabled=cfg.use_mixed_precision):
			flow_preds, pose_preds, sc_flow_preds, mask_preds = model(ref['image'].squeeze(1), que['image'].squeeze(1), ref['K_img'].squeeze(1), que['K'].squeeze(1), 
						 ref['RT'].squeeze(1), ref['geo'].squeeze(1), ref['mask'].squeeze(1), flow_gt.squeeze(1), mask_gt.squeeze(1),obj_ids, 8)
			# que['error'] = torch.abs(pred['coord_0']-que['coord']).detach()
			# coord_loss = (que['mask_visib'] * distance(pred['coord_0'], que['coord'])).mean() if 'coord' in cfg.loss else 0
			# error_loss = (que['mask_visib'] * distance(pred['error'], que['error'])).mean() if 'error' in cfg.loss else 0
			# mask_loss = distance(pred['mask'], que['mask']).mean() if 'mask' in cfg.loss else 0
			# mask_visib_loss = distance(pred['mask_visib'], que['mask_visib']).mean() if 'mask_visib' in cfg.loss else 0
			# grid_loss_1, _ = grid_matching_loss(pred['RT_1_candidate'], que['RT'], que['K_ref'], cfg.ref_size, cfg.pool_size) 
			# grid_loss_2, _ = grid_matching_loss(pred['RT_2'], que['RT'], que['K_ref'], cfg.ref_size, cfg.pool_size) 

			# loss = 20 * (coord_loss + mask_loss + mask_visib_loss + error_loss) + grid_loss_1 + grid_loss_2
			pose_preds = [pp.unsqueeze(1) for pp in pose_preds]
			flow_preds = [fp.unsqueeze(1) for fp in flow_preds]
			mask_preds = [mp.unsqueeze(1) for mp in mask_preds]
			

			loss = defaultdict()
			#pose_losses =  [grid_matching_loss(pp, que['RT'], que['K'], cfg.img_size, 8, 8) for pp in pose_preds]
			#pose_losses = torch.stack(pose_losses)
			pose_losses = [DisentanglePointMatchingLoss(pp, que['RT'], ref['pts']) for pp in pose_preds]
			pose_losses = 10.0 * torch.stack(pose_losses)
			flow_losses = 0.1 * flow_loss_with_mask(flow_preds, flow_gt, mask_gt) #0.1
			mask_losses = [l1(mp, mask_gt).mean() for mp in mask_preds]
			mask_losses = 10.0 * torch.stack(mask_losses)
			sequence_loss = sequence_loss2(pose_losses + flow_losses + mask_losses)

			loss = sequence_loss 
		if cfg.use_mixed_precision:
			gradscaler.scale(loss).backward()
			gradscaler.unscale_(optim)
			gradscaler.step(optim)
			gradscaler.update()
			lr_scheduler.step()
		else:
			loss.backward()
			optim.step()
			lr_scheduler.step()

		if steps % 10 == 0 and rank==0:
			log = (f'[Step: {steps} time: {(time()-start):.2f}] '
				   f'Loss: {float(loss):.3f} '
				   f'pose_loss_final: {float(pose_losses[-1]):.3f} '
				   f'flow_loss_final: {float(flow_losses[-1]):.3f} '
				   f'mask_loss_final: {float(mask_losses[-1]):.3f} '
				   f'sequence_loss: {float(sequence_loss):.3f} '
				   )
			print(log)
			writer.add_scalar('train/total_loss', loss, steps)
			writer.add_scalar('train/sequence_loss', sequence_loss, steps)
			writer.add_scalar('train/pose_loss_final', pose_losses[-1], steps)
			writer.add_scalar('train/flow_loss_final',flow_losses[-1], steps)
			writer.add_scalar('train/mask_loss_final', mask_losses[-1], steps)

			writer.add_scalar('train/lr', optim.param_groups[0]['lr'], steps)

		if steps % 100 == 0:
			train_dataloader.dataset.shuffle_data()
			
		if (steps % cfg.logging.visualize_after == 0 and rank==0):
			model_valid = model.module if is_ddp else model
			model_valid.eval()
			with torch.no_grad():
				flow_preds, pose_preds, sc_flow_preds, mask_preds = model(ref['image'].squeeze(1), que['image'].squeeze(1), ref['K_img'].squeeze(1), que['K'].squeeze(1), 
						 ref['RT'].squeeze(1), ref['geo'].squeeze(1), ref['mask'].squeeze(1), flow_gt.squeeze(1), mask_gt.squeeze(1), obj_ids, 8)
				pose_preds = [pp.unsqueeze(1) for pp in pose_preds]
				flow_preds = [fp.unsqueeze(1) for fp in flow_preds]
				sc_flow_preds = [scfp.unsqueeze(1) for scfp in sc_flow_preds]
				mask_preds = [mp.unsqueeze(1) for mp in mask_preds]
				pose_pred = pose_preds[-1]
				flow_pred = flow_preds[-1]
				sc_flow_pred = sc_flow_preds[-1]
				mask_pred = mask_preds[-1]
				
				# pred['coord_1'] = render_geometry(obj['structure'], que['K_ref'], pred['RT_1'], cfg.ref_size, cfg.n_freqs, cfg.represent_mode, cfg.ray_mode)
				# pred['coord_2'] = render_geometry(obj['structure'], que['K_ref'], pred['RT_2'], cfg.ref_size, cfg.n_freqs, cfg.represent_mode, cfg.ray_mode)
				stitched_image = convert_tensor_to_images_scflow_base(que, ref, flow_pred, pose_pred, flow_gt, sc_flow_pred, mask_pred, mask_gt, cfg)
				writer.add_image("training results", stitched_image, steps, dataformats='HWC')
				# for level in fw_attn_l.keys():
				#     attn_image = convert_attn_to_images(ref['image'][0], que['image'][0], fw_attn_l[level], fw_attn_p[level])
				#     writer.add_image(f"training attention {level}", attn_image, steps, dataformats='HWC')
		
		if (steps % cfg.validation.validate_after == 0 and rank == 0):
			model_valid = model.module if is_ddp else model
			score, results = validate_challenge_add(cfg, model_valid, val_dataloader, steps, writer)
			if score >= best_score:
				_ = total_metric(
					results['obj_id'], 
					results['diameter'], 
					results['adds_1'], 
					results['adds_2'], 
					is_save=True, 
					is_print=False,
					prefix=f'{val_dir}/{steps}')
				best_score = score
				torch.save({
					'steps': steps,
					'best_score': best_score,
					'model_state_dict': model_valid.state_dict(),
					'optimizer_state_dict': optim.state_dict(),
					'lr_scheduler_state_dict': lr_scheduler.state_dict(),
					'scaler_state_dict': gradscaler.state_dict()}, ckpt_path)
		model.train()
		steps += 1
		if steps > cfg.num_steps:
			break

def trainer(cfg, is_memory):
	train_dataset = get_dataset(cfg.dataset, is_train=True, is_memory=is_memory)
	val_dataset = challenge_get_dataset(cfg.dataset)
	#val_dataset = get_dataset(cfg.dataset, is_train=False, is_memory=is_memory)
	model = SCFlow(cfg.model).cuda()
	if is_ddp:
		model = SyncBatchNorm.convert_sync_batchnorm(model)
		model = DDP(model, device_ids=[rank])
		batch_size = cfg.dataset.batch_size // world_size
	else:
		batch_size = cfg.dataset.batch_size
	least_loaded_cores = get_least_loaded_cores(rank)
	valid_worker = min(cfg.training.num_workers, len(least_loaded_cores)-2)
	train_init_fn = partial(worker_init_fn, least_loaded_cores=least_loaded_cores)
	valid_init_fn = partial(worker_init_fn, least_loaded_cores=least_loaded_cores[valid_worker:])
	train_dataloader = DataLoader(
		dataset=train_dataset, 
		batch_size=batch_size,
		num_workers=cfg.training.num_workers,
		collate_fn=train_dataset.collate_fn,
		#prefetch_factor=4,
		worker_init_fn=train_init_fn)
	valid_worker = min(cfg.training.num_workers, 
					   len(least_loaded_cores)-2)
	val_dataloader = DataLoader(
		dataset=val_dataset, 
		batch_size=2, 
		sampler=SequentialSampler(val_dataset),
		num_workers=2,
		collate_fn=val_dataset.collate_fn,
		worker_init_fn=valid_init_fn)
	train(model, train_dataloader, val_dataloader, cfg.training)

def get_cfg(cfg_path):
	cfg = OmegaConf.load(cfg_path)
	exp_dir = f'saved/{cfg.training.exp_tag}'
	to_path = f'{exp_dir}/{os.path.basename(cfg_path)}'
	cfg.infer.load_path = f'{exp_dir}/ckpts/latest.pt'
	if not os.path.exists(to_path):
		cfg_copy = cfg.copy()
		os.makedirs(exp_dir, exist_ok=True)
		cfg_copy.training.load_path = f'{exp_dir}/ckpts/latest.pt'
		OmegaConf.save(config=cfg_copy, f=to_path)
	return cfg

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--scheduler', type=str2bool, default=False)
	parser.add_argument('--is_memory', type=str2bool, default=False)
	parser.add_argument('--config-path', type=str, required=True)
	parser.add_argument('--local_rank', type=int, default=0, required=False)
	parser.add_argument('--debug', type=str2bool, default=False, required=False)
	args = parser.parse_args() 
	is_ddp = (torch.cuda.device_count() > 1)        # global variable
	world_size = torch.cuda.device_count()
	rank = args.local_rank
	cfg = get_cfg(args.config_path)                 # global variable
	cfg.training.debug = args.debug
	if cfg.training.debug:
		cfg.dataset.batch_size = 8
		cfg.training.logging.visualize_after = 10 #10
		cfg.training.validation.validate_after = 10
		cfg.dataset.obj_list = [1, 2]
		cfg.model.num_class = 2
		cfg.training.exp_tag = 'bop_specific/debugging'
		cfg.training.num_steps = 100000 #20
	is_ddp = (torch.cuda.device_count() > 1)        # global variable
	if is_ddp:
		world_size = torch.cuda.device_count()  # global variable
		dist.init_process_group(
			backend='gloo', 
			init_method='env://', 
			world_size=world_size,
			rank=rank)
	torch.cuda.set_device(rank)
	SEED = 123 + rank
	torch.manual_seed(SEED + rank)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(SEED + rank)
	if args.scheduler or rank != 0:
		def print_pass(*args): pass
		builtins.print=print_pass
		tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
	print(f'N of gpus: {world_size}')
	trainer(cfg, args.is_memory)
