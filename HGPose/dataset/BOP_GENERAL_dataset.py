from PIL import Image
import numpy as np
import random
import torch
from pytorch3d.io import IO, load_objs_as_meshes
import matplotlib.pyplot as plt
from HGPose.utils.geometry import (positive_RT, negative_RT, get_ref_RT, jitter_RT, bbox_add_noise, get_flow_from_delta_pose_and_xyz, apply_forward_flow,
	get_K_crop_resize, squaring_boxes, image_cropping, mesh_convert_coord, random_RT, eval_rot_error, carving_feature, 
	apply_backward_flow, 
	default_K, R_T_to_RT, render_geometry, RT_from_boxes, encode_coordinate)
from HGPose.utils.object import normalize_mesh, convert_to_textureVertex
from collections import defaultdict
from bop_toolkit.bop_toolkit_lib.dataset.bop_webdataset import decode_sample
from HGPose.utils.dataset import load_json_fast
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
import webdataset as wds
import os

def get_dataset(cfg, is_web=True, given_data=None):
	data_src = []
	if is_web:
		if 'megapose-gso' in cfg.train_dataset:
			data_src += ["https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/train_pbr_web/shard-{000000..001039}.tar"]
		if 'megapose-shapenet' in cfg.train_dataset:
			data_src += ["https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-shapenet/train_pbr_web/shard-{000000..001039}.tar"]
	else:
		if 'megapose-gso' in cfg.train_dataset:
			data_src += glob('Dataset/BOP_GENERAL/megapose-gso/data/*.tar')
			# data_src = [given_data]
		if 'megapose-shapenet' in cfg.train_dataset:
			data_src += glob('Dataset/BOP_GENERAL/megapose-shapeneto/data/*.tar')
	dataset = BOP_GENERAL(cfg, data_src)
	return dataset

class BOP_GENERAL():
	def __init__(self, cfg, data_src):
		self.N_que = cfg.N_que
		self.N_ref = cfg.N_ref
		self.N_z = cfg.N_z
		self.N_freq = cfg.N_freq
		self.represent_mode = cfg.represent_mode
		self.N_template = cfg.N_template
		self.img_size = cfg.img_size
		self.carving_size = cfg.carving_size
		self.use_carved = cfg.use_carved
		self.root_path = f'Dataset/BOP_GENERAL'
		self.base_ref_path = 'ref'
		self.gso_obj_list = load_json_fast(f'{self.root_path}/megapose-gso/gso_models.json')
		# self.shapenet_obj_list = load_json_fast(f'{self.root_path}/megapose-shapenet/shapenet_models.json')

		self.generate_reference()

		self.dataset = wds.DataPipeline(
			wds.SimpleShardList(data_src), 			# iterator over all the shards
			wds.shuffle(100),
		    wds.split_by_worker,						# list of decompressed training samples from each shard
			wds.shuffle(100),
			wds.tarfile_to_samples(),
			wds.shuffle(100),
			self.scene_decoder,
			wds.shuffle(100),
			self.img_decoder)

	def scene_decoder(self, scene):
		for img in scene:
			im = decode_sample(
				img,
				decode_camera=True, 
				decode_rgb=True, 
				decode_gray=False, 
				decode_depth=False, 
				decode_gt=True, 
				decode_gt_info=True, 
				decode_mask=False,
				decode_mask_visib=True,
				rescale_depth=False)
			K = torch.tensor(np.array(im['camera']['cam_K'], np.float64).reshape((3, 3))).to(torch.float)
			RT_cam = torch.tensor(R_T_to_RT(
				np.array(im['camera']['cam_R_w2c'], np.float64).reshape(3, 3), 
				np.array(im['camera']['cam_t_w2c'], np.float64).reshape(3, 1) * 0.001)).to(torch.float)
			image = torch.tensor(im['im_rgb']).unsqueeze(0).permute(0, 3, 1, 2)/255.0
			data_id = im['__url__'].split('/')[-3]
			data = [{
				'data_id': data_id, 
				'K' : K,
				'RT_cam': RT_cam,
				'image' : image, 
				'gt': gt,
				'info': info,
				'mask_visib': mask_visib} 
				for gt, info, mask_visib in zip(im['gt'], im['gt_info'], im['mask_visib'])]
			yield data

	def img_decoder(self, data):
		for img in data:
			for inst in img:
				if inst['info']['visib_fract'] > 0.5:
					inst['info']['bbox_obj'][2] += inst['info']['bbox_obj'][0]
					inst['info']['bbox_obj'][3] += inst['info']['bbox_obj'][1]
					bbox = torch.tensor(inst['info']['bbox_obj']).to(torch.float)
					bbox = bbox_add_noise(bbox, std_rate=0.1)
					bbox = squaring_boxes(bbox, lamb=1.1)
					K_img = get_K_crop_resize(inst['K'], bbox, [self.img_size, self.img_size])
					RT = torch.tensor(R_T_to_RT(
						np.array(inst['gt']['cam_R_m2c'], np.float64).reshape(3, 3), 
						np.array(inst['gt']['cam_t_m2c'], np.float64).reshape(3, 1) * 0.001)).to(torch.float)
					mask_visib = torch.tensor(inst['mask_visib']).unsqueeze(0).unsqueeze(1).to(torch.float)
					image = image_cropping(inst['image'], bbox, [self.img_size, self.img_size])
					mask_visib = image_cropping(mask_visib, bbox, [self.img_size, self.img_size]) 
					# RT_coarse = jitter_RT(RT)
					sample = {
						'data_id' : inst['data_id'],
						'obj_id' : inst['gt']['obj_id'],
						'image' : image,
						'mask_visib' : mask_visib,
						'RT' : RT,
						# 'RT_coarse': RT_coarse,
						'bbox' : bbox,
						'K_img' : K_img}
					yield sample

	def collate_fn(self, batch):
		que = defaultdict()
		ids = defaultdict()
		batch = {k: [scene[k] for scene in batch] for k in batch[0].keys()}
		ids['data_id'] = batch['data_id']
		ids['obj_id'] = batch['obj_id']
		que['image'] = torch.stack(batch['image'])
		que['mask'] = torch.stack(batch['mask_visib'])
		que['RT'] = torch.stack(batch['RT']).unsqueeze(1)
		# que['RT_coarse'] = torch.stack(batch['RT_coarse']).unsqueeze(1)
		que['K_img'] = torch.stack(batch['K_img']).unsqueeze(1)
		que['bbox'] = torch.stack(batch['bbox']).unsqueeze(1)
		return ids, que

	def generate_reference(self, f_img=10000, rendering_k=16):
		print('generate gso references...')
		path = f'{self.root_path}/megapose-gso/{self.base_ref_path}'
		os.makedirs(path, exist_ok=True)
		for obj in tqdm(self.gso_obj_list):
			o_path = f'{self.root_path}/megapose-gso/model/{obj["gso_id"]}/meshes/model.obj'
			ref_dir = f"{path}/{obj['obj_id']:06}"
			if not os.path.isdir(ref_dir):
				os.makedirs(ref_dir, exist_ok=True)
				mesh = load_objs_as_meshes([o_path], device='cuda')
				mesh = normalize_mesh(mesh, scale=0.1)
				scale = float(torch.norm(mesh.verts_packed(), dim=1, p=2).max())
				mesh = mesh.scale_verts(1/scale)
				xyz_mesh = mesh_convert_coord(mesh.clone())
				pts = mesh.verts_packed()
				K = default_K(self.img_size, f_img).view(1, 3, 3).repeat(self.N_template, 1, 1).cuda()
				RT = random_RT(K, self.img_size, self.N_template).cuda()
				ref_images = []
				for idxs in range(0, self.N_template, rendering_k):
					idx = list(range(idxs, idxs+rendering_k))
					image = render_geometry(
						[mesh.cuda()], K[idx][None], RT[idx][None], self.img_size, self.img_size, represent_mode='rgb')
					image = (image[0] + 1) / 2
					for i, img in zip(idx, image):
						img = img.permute(1, 2, 0).cpu().numpy()
						plt.imsave(f'{ref_dir}/{i:06}.png', img)
					ref_images += image
				ref_images = torch.stack(ref_images)
				ref_masks = (ref_images.sum(-3, keepdims=True) > 0).to(ref_images)
				carved = carving_feature(ref_masks, RT, K, self.carving_size)
				ref = defaultdict()
				ref['RT'] = RT
				ref['K'] = K
				ref['pts'] = pts.cpu().numpy()
				ref['xyz_mesh'] = xyz_mesh
				ref['carved'] = carved
				ref['scale'] = scale 
				torch.save(ref, f'{ref_dir}/ref.pt')


	def batch_reference(self, ids, que):
		device = que['image'].device
		ref = [torch.load(f'{self.root_path}/{data_id}/{self.base_ref_path}/{obj_id:06}/ref.pt') 
		 	for data_id, obj_id in zip(ids['data_id'], ids['obj_id'])]
		ref = {k: [sample[k] for sample in ref] for k in ref[0].keys()}
		ref['RT'] = torch.stack(ref['RT']).to(device)
		ref['K_img'] = torch.stack(ref['K']).to(device)
		ref_idx = positive_RT(ref['RT'], que['RT'], self.N_ref)
		ref_idx = ref_idx[:, torch.randperm(ref_idx.size()[1])]
		ref['RT'] = torch.stack([ref['RT'][i, r_i] for i, r_i in enumerate(ref_idx)])
		ref['K_img'] = torch.stack([ref['K_img'][i, r_i] for i, r_i in enumerate(ref_idx)])
		ref['image'] = []
		for r_i, data_id, obj_id in zip(ref_idx, ids['data_id'], ids['obj_id']):
			imgs = []
			for i in r_i:
				img = plt.imread(f'{self.root_path}/{data_id}/{self.base_ref_path}/{obj_id:06}/{i:06}.png')[..., :3]
				imgs += [torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)]
			ref['image']  += [torch.cat(imgs, dim=0).unsqueeze(0)]
		ref['image'] = torch.cat(ref['image'], dim=0).to(device)
		if self.use_carved:
			ref['source'] = torch.stack(ref['carved']).to(device)
		else:
			ref['source'] = ref['xyz_mesh']
		ref['mask'] = (ref['image'] > 0.0).to(torch.float).mean(dim=-3, keepdims=True)
		return ref

	def preprocess_batch(self, que, ref):
		device = que['RT'].device
		bsz, N_ref = ref['K_img'].shape[:2]
		scale = torch.tensor(ref['scale']).view(-1, 1, 1).to(device)
		que['RT'][..., :3, 3] = que['RT'][..., :3, 3] / scale
		# que['RT_coarse'][..., :3, 3] = que['RT_coarse'][..., :3, 3] / scale

		if self.use_carved:
			source = ref['source'].expand(bsz, N_ref, 4, self.carving_size, self.carving_size, self.carving_size)
		else:
			source = ref['source']
		ref['geo'] = render_geometry(source, ref['K_img'], 
			ref['RT'], self.img_size, self.img_size, self.N_z, self.N_freq, self.represent_mode)
		que['geo'] = render_geometry(ref['source'], que['K_img'], 
			que['RT'], self.img_size, self.img_size, self.N_z, self.N_freq, self.represent_mode)
		ref['xyz'] = render_geometry(source, ref['K_img'], 
			ref['RT'], self.img_size, self.img_size, self.N_z, self.N_freq, 'xyz')
		que['xyz'] = render_geometry(ref['source'], que['K_img'], 
			que['RT'], self.img_size, self.img_size, self.N_z, self.N_freq, 'xyz')
		flow_gt, flowconf_gt = get_flow_from_delta_pose_and_xyz(que['RT'], que['K_img'], ref['xyz'], ref['mask'], que['xyz'])
		return que, ref, flow_gt, flowconf_gt