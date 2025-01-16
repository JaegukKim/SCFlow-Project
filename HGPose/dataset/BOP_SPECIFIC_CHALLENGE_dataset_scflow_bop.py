from PIL import Image
import os
import numpy as np
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from HGPose.utils.geometry import (
	get_K_crop_resize, squaring_boxes, image_cropping, bbox_add_noise, 
	farthest_rotation_sampling, carving_feature, R_T_to_RT, render_geometry, render_geometry_scflow, RT_from_boxes, mesh_convert_coord, get_flow_from_delta_pose_and_xyz_scflow)
from collections import defaultdict
from bop_toolkit.bop_toolkit_lib.inout import load_scene_camera, load_scene_gt, load_json, load_ply
from glob import glob
from tqdm import tqdm
from abc import abstractmethod
from copy import deepcopy
import psutil
import time
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
import trimesh
from pytorch3d.io import IO
import csv

def get_dataset(cfg):
	if cfg.dataset_name == 'ycbv':
		dataset = ycbv_dataset(cfg)
	return dataset

def get_least_loaded_cores(rank):
	time.sleep(rank)
	cpu_percentages = psutil.cpu_percent(percpu=True, interval=5.0)
	return sorted(range(len(cpu_percentages)), key=lambda x: cpu_percentages[x])

def worker_init_fn(worker_id, least_loaded_cores):
	# 사용량이 가장 낮은 core 선택
	core_to_use = least_loaded_cores[worker_id % len(least_loaded_cores)]
	print(f'worker core: {core_to_use}')
	# 해당 worker의 프로세스에 CPU affinity 설정
	current_process = psutil.Process()
	current_process.cpu_affinity([core_to_use])

class BOP_Dataset(Dataset):
	def __init__(self, cfg, is_memory=False):
		self.num_query_views = cfg.num_query_views
		self.num_input_views = cfg.num_input_views
		self.image_size = cfg.image_size
		self.ref_size = cfg.ref_size
		self.geo_size = cfg.geo_size
		self.represent_mode = cfg.represent_mode
		self.ray_mode = cfg.ray_mode
		self.obj_list = cfg.obj_list
		self.use_carved = cfg.use_carved
		self.is_memory = is_memory
		self.data_domain = cfg.data_domain
		self.dataset_name = cfg.dataset_name


	def load_infos(self, coarse_csv):
		infos = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
		with open(f'Dataset/BOP_SPECIFIC/cosypose_init/{coarse_csv}', newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			head = next(spamreader)
			for row in spamreader:
				scene_id, im_id, obj_id = int(row[0]), int(row[1]), int(row[2])
				score, time = float(row[3]), float(row[6])
				R = np.array([float(e) for e in row[4].split(' ')]).reshape(3, 3)
				t = np.array([float(e) for e in row[5].split(' ')]).reshape(3, 1)
				RT_coarse = R_T_to_RT(R, t)
				infos[scene_id][im_id][obj_id] += [{
					'RT_coarse': torch.tensor(RT_coarse).to(torch.float),
					'score': score, 
					'time': time}]
	
		targets = load_json(f'{self.root_path}/test_targets_bop19.json')		
		inst_N = {(target['scene_id'], target['im_id'], target['obj_id']): target['inst_count'] 
			for target in targets}
		for scene_id, im_dict in infos.items():		# detections.items()
			for im_id, obj_dict in im_dict.items():
				for obj_id, inst_list in obj_dict.items():
					if (scene_id, im_id, obj_id) in inst_N.keys():
						N = inst_N[(scene_id, im_id, obj_id)]
						# top_N = sorted(inst_list, key=lambda x: x['score'], reverse=True)[:N]
						top_1 = sorted(inst_list, key=lambda x: x['score'], reverse=True)[:1]
						infos[scene_id][im_id][obj_id] = top_1
		return infos

	def load_dataset(self, info, name='test', mode='rgb'):
		if self.dataset_name in ('lmo', 'ycbv', 'tless', 'icbin', 'tudl'):  #test gt exists
			if mode == 'rgb': 
				img_type = 'png'
			else: 
				img_type = 'tif'
			image_data = defaultdict(dict)
			instance_data = defaultdict(lambda: defaultdict(dict))
			for scene_id, im_dict in info.items():
				scene_str = "{0:0=6d}".format(scene_id)
				gt_path = f'{self.root_path}/{name}/{scene_str}/scene_gt.json'		
				camera_path = f'{self.root_path}/{name}/{scene_str}/scene_camera.json'
				info_path = f'{self.root_path}/{name}/{scene_str}/scene_gt_info.json'
				scene_gt = load_scene_gt(gt_path)
				scene_camera = load_scene_camera(camera_path)
				scene_info = load_json(info_path, keys_to_int=True)
				for im_id, obj_dict in im_dict.items():
					im_str = "{0:0=6d}".format(im_id)
					camera = scene_camera[im_id]
					im_gt = scene_gt[im_id]
					im_info = scene_info[im_id]
					image = f'{self.root_path}/{name}/{scene_str}/{mode}/{im_str}.{img_type}'
					depth = f'{self.root_path}/{name}/{scene_str}/depth/{im_str}.png'
					image_data[(scene_id, im_id)] = {
						'K' : torch.tensor(camera['cam_K']).to(torch.float),
						'depth_scale' : camera['depth_scale'],
						'image' : image,
						'depth': depth,
						'scene_id' : scene_id,
						'im_id' : im_id}

					for obj_id, inst_list in obj_dict.items():
						for inst_id, instance in enumerate(inst_list):
							idx = [l for l, g in enumerate(im_gt) if obj_id==g['obj_id']]
							if len(idx) > 0: 
								idx = idx[0]
								inst_str = "{0:0=6d}".format(idx)
								gt = im_gt[idx]
								inf = im_info[idx]
								RT = torch.tensor(R_T_to_RT(gt['cam_R_m2c'], gt['cam_t_m2c'])).to(torch.float)

								# instance['bbox'][2] += instance['bbox'][0]
								# instance['bbox'][3] += instance['bbox'][1]

								inf['bbox_visib'][2] += inf['bbox_visib'][0]
								inf['bbox_visib'][3] += inf['bbox_visib'][1]


								RT_coarse = instance['RT_coarse']
								mask = f'{self.root_path}/{name}/{scene_str}/mask/{im_str}_{inst_str}.png'
								mask = np.asarray(Image.open(mask))
								mask_visib = f'{self.root_path}/{name}/{scene_str}/mask_visib/{im_str}_{inst_str}.png'
								mask_visib = np.asarray(Image.open(mask_visib))
								
								instance_data[(scene_id,im_id,obj_id)][inst_id] = {
									'RT': RT,
									'RT_coarse': RT_coarse,
									'mask': mask,
									'mask_visib': mask_visib,
									'bbox_gt' : torch.tensor(inf['bbox_visib']).to(torch.float),
									'score' : instance['score'],
									'time' : instance['time'],
									'inst_id': inst_id,
									'obj_id' : obj_id}
							else:
								continue
							
			return {'image_data': image_data, 'instance_data': instance_data}
		
		else:
			return None

	def load_sample(self, input_data, bbox=None, noisy=False):
		data = dict(list(deepcopy(input_data[0]).items()) + list(deepcopy(input_data[1]).items()))
		image = np.asarray(Image.open(data['image']).convert("RGB")) #if isinstance(data['image'], str) else data['image']
		mask = data['mask'] / data['mask'].max() #np.asarray(Image.open(data['mask'])) if isinstance(data['mask'], str) else data['mask']
		mask_visib = data['mask_visib'] / data['mask_visib'].max() #np.asarray(Image.open(data['mask_visib'])) if isinstance(data['mask_visib'], str) else data['mask_visib']
		image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)/255.0
		mask = torch.tensor(mask).to(torch.float).unsqueeze(0).unsqueeze(1) #torch.tensor(mask).unsqueeze(0).unsqueeze(1)/255.0
		mask_visib = torch.tensor(mask_visib).to(torch.float).unsqueeze(0).unsqueeze(1) #torch.tensor(mask_visib).unsqueeze(0).unsqueeze(1)/255.0
		image = image_cropping(image, bbox, [self.image_size, self.image_size])
		mask = image_cropping(mask, bbox, [self.ref_size, self.ref_size]).round()
		mask_visib = image_cropping(mask_visib, bbox, [self.ref_size, self.ref_size]).round()
		K = get_K_crop_resize(data['K'], bbox, [self.image_size, self.image_size])
		K_ref = get_K_crop_resize(data['K'], bbox, [self.ref_size, self.ref_size])
		obj_index = self.obj_list.index(data['obj_id']) 
		sample = {
			'scene_id': data['scene_id'],
			'im_id': data['im_id'],
			'inst_id': data['inst_id'],
			'obj_id': data['obj_id'],
			'score': data['score'],
			'obj_index': obj_index,
			'image': image,
			'mask': mask,
			'mask_visib': mask_visib,
			'RT': data['RT'],
			'bbox': bbox,
			'K_full': data['K'],
			'K': K,
			'K_ref': K_ref}
		return sample

	def __getitem__(self, step):
		scene_id, im_id, obj_id, inst_id = self.id_list[step]
		image_data = self.dataset['image_data'][(scene_id, im_id)]
		instance_data = self.dataset['instance_data'][(scene_id, im_id, obj_id)][inst_id]


		coarse_RT = instance_data['RT_coarse'] 
		coarse_RT[..., :3, 3] = coarse_RT[..., :3, 3] / torch.tensor(self.object_param[obj_id]['scale'])

		###########coarse_Rt -> bbox###############
		pts = torch.from_numpy(self.object_param[obj_id]['pts'])
		k = image_data['K']
		rotation = coarse_RT[:3,:3]
		translation = coarse_RT[:3,3][...,None]
		pts_3d_camera = torch.matmul(rotation, pts.transpose(0,1)) + translation
		pts_2d = torch.matmul(k, pts_3d_camera).transpose(0,1)
		pts_2d[..., 0] = pts_2d[..., 0]/ (pts_2d[..., -1] + 1e-8)
		pts_2d[..., 1] = pts_2d[..., 1]/ (pts_2d[..., -1] + 1e-8)
		pts_2d = pts_2d[..., :-1]
		points_x, points_y = pts_2d[:, 0], pts_2d[:, 1]
		left, right = points_x.min(), points_x.max()
		top, bottom = points_y.min(), points_y.max()
		bbox = torch.tensor([left, top, right, bottom], dtype=torch.float32)

		#############bbox crop and resize################
		size_ratio = torch.tensor(1.1)
		x1, y1, x2, y2 = bbox
		bbox_w, bbox_h = x2 - x1, y2 - y1
		xc, yc = (x1 + x2)/2, (y1 + y2)/2
		aspect_ratio = torch.tensor(1.0)
		bbox_w = max(bbox_w, bbox_h * aspect_ratio)
		bbox_h = max(bbox_w/aspect_ratio, bbox_h)
		old_bbox_w, old_bbox_h = bbox_w, bbox_h
		new_bbox_w, new_bbox_h = bbox_w*size_ratio, bbox_h*size_ratio
		bbox_w, bbox_h = new_bbox_w, new_bbox_h
		crop_x1, crop_x2 = int(xc - bbox_w/2), int(xc + bbox_w/2)
		crop_y1, crop_y2 = int(yc - bbox_h/2), int(yc + bbox_h/2)

		bbox = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2])
				
		query = self.load_sample((image_data, instance_data), bbox=bbox, noisy=False)
		query['RT_coarse'] = coarse_RT
		query['RT'][..., :3, 3] = query['RT'][..., :3, 3] / torch.tensor(self.object_param[obj_id]['scale'])
		return query
			

	def __len__(self):
		return len(self.id_list)


	def preprocess_batch(self, que, obj):
		ref = defaultdict()
		device = que['RT'].device
		bsz, N_ref = que['K'].shape[:2]
		scale = torch.tensor(obj['scale']).view(-1, 1, 1).to(device)

		ref['RT'] = que['RT_coarse']
		ref['K'] = que['K']
		# ref['K'] = get_K_crop_resize(
		# 	que['K_full'][:, 0], que['bbox'], [self.image_size, self.image_size]).unsqueeze(1)
		ref['K_img'] = ref['K']
		xyz_mesh = [x.to(device) for x in obj['xyz_mesh']]
		full_mesh = [x.to(device) for x in obj['full_mesh']]
		ref['source'] = xyz_mesh
		# ref['xyz'] = render_geometry(xyz_mesh, ref['K'], ref['RT'], self.image_size, self.image_size, self.N_z, self.N_freq, 'xyz')
		# que['xyz'] = render_geometry(xyz_mesh, que['K'], que['RT'], self.image_size, self.image_size, self.N_z, self.N_freq, 'xyz')
		ref['xyz'] = render_geometry_scflow(xyz_mesh, ref['K'], ref['RT'], self.image_size, self.geo_size, represent_mode = 'xyz')
		ref['xyz_origin'] = render_geometry_scflow(xyz_mesh, ref['K'], ref['RT'], self.image_size, self.image_size, represent_mode = 'xyz')
		que['xyz'] = render_geometry_scflow(xyz_mesh, que['K'], que['RT'], self.image_size, self.geo_size, represent_mode = 'xyz')
		que['xyz_origin'] = render_geometry_scflow(xyz_mesh, que['K'], que['RT'], self.image_size, self.image_size, represent_mode = 'xyz')
		ref['image'] = render_geometry(full_mesh, ref['K'], ref['RT'], self.image_size, self.image_size, represent_mode='rgb')
		ref['image'] = (ref['image'] + 1) / 2
		ref['mask'] = torch.round((ref['image'] > 0.0).to(torch.float).mean(dim=-3, keepdims=True))
		flow_gt = get_flow_from_delta_pose_and_xyz_scflow(que['RT'], que['K'], ref['xyz_origin'], ref['mask'], que['xyz'])


		return que, ref, flow_gt


	def batch_object(self, batch_obj_id):
		obj = defaultdict()
		batch_obj_id = batch_obj_id.tolist()
		obj['full_mesh'] = [self.object_param[id]['full_mesh'] for id in batch_obj_id]
		obj['xyz_mesh'] = [self.object_param[id]['xyz_mesh'] for id in batch_obj_id]
		obj['diameter'] = [self.object_param[id]['diameter'] for id in batch_obj_id]
		obj['symmetry'] = [self.object_param[id]['symmetry'] for id in batch_obj_id]
		obj['pts'] = [self.object_param[id]['pts'] for id in batch_obj_id]
		obj['scale'] = [self.object_param[id]['scale'] for id in batch_obj_id]
		return obj 

	def collate_fn(self, batch):
		ids = defaultdict()
		que = defaultdict()
		if batch[0] is not None:
			ids['scene_id'] = torch.tensor([q['scene_id'] for q in batch])
			ids['im_id'] = torch.tensor([q['im_id'] for q in batch])
			ids['inst_id'] = torch.tensor([q['inst_id'] for q in batch])
			ids['obj_id'] = torch.tensor([q['obj_id'] for q in batch])
			ids['score'] = torch.tensor([q['score'] for q in batch])
			que['obj_index'] = torch.tensor([q['obj_index'] for q in batch])
			que['RT'] = torch.stack([q['RT'] for q in batch]).unsqueeze(1)
			que['K_full'] = torch.stack([q['K_full'] for q in batch]).unsqueeze(1)
			que['K'] = torch.stack([q['K'] for q in batch]).unsqueeze(1)
			que['K_ref'] = torch.stack([q['K_ref'] for q in batch]).unsqueeze(1)
			que['image'] = torch.stack([q['image'] for q in batch])
			que['mask'] = torch.stack([q['mask'] for q in batch])
			que['mask_visib'] = torch.stack([q['mask_visib'] for q in batch])
			que['bbox'] = torch.stack([q['bbox'] for q in batch])
			que['RT_coarse'] = torch.stack([q['RT_coarse'] for q in batch]).unsqueeze(1)
			return ids, que
		else:
			return None, None

	def set_object_scflow(self, obj_list, include_textures=False):
		object_info = load_json(f'{self.root_path}/models/models_info.json')
		obj_param = defaultdict(dict)
		for obj_id in obj_list:
			info = object_info[str(obj_id)]
			full_mesh = IO().load_mesh(f'{self.root_path}/models/obj_{obj_id:06}.ply', include_textures=True, device='cuda')
			if full_mesh.textures == None:
								file_list = os.listdir(f'{self.root_path}/models')
								png_files = [file for file in file_list if file.endswith(".png")]
								if png_files:
									full_mesh = get_texture_mesh_with_trimesh(f'{self.root_path}/models/obj_{obj_id:06}.ply')
			mesh = IO().load_mesh(f'{self.root_path}/models_eval/obj_{obj_id:06}.ply', include_textures=False, device='cuda')
			scale = float(torch.norm(mesh.verts_packed(), dim=1, p=2).max())
			full_mesh = full_mesh.scale_verts(1/scale)
			mesh = mesh.scale_verts(1/scale)
			pts = mesh.verts_packed()
			obj_param[obj_id]['full_mesh'] = full_mesh
			obj_param[obj_id]['xyz_mesh'] = mesh_convert_coord(mesh.clone())
			obj_param[obj_id]['pts'] = pts.cpu().numpy()
			obj_param[obj_id]['diameter'] = info['diameter'] / scale
			obj_param[obj_id]['symmetry'] = ('symmetries_discrete' in info.keys() or 'symmetries_continuous' in info.keys())
			obj_param[obj_id]['scale'] = scale
		return obj_param
	

class ycbv_dataset(BOP_Dataset):
	def __init__(self, cfg, is_memory=False):
		super(ycbv_dataset, self).__init__(cfg, is_memory)
		self.root_path = f'Dataset/BOP_SPECIFIC/{cfg.dataset_name}'

		self.object_param = self.set_object_scflow(self.obj_list, include_textures=True)
		if ('real' in self.data_domain) and not ('pbr' in self.data_domain):
			info = self.load_infos('challenge2020-223026_ycbv-test.csv')
		elif ('real' in self.data_domain) and ('pbr' in self.data_domain):
			info = self.load_infos('challenge2020-815712_ycbv-test.csv')
		self.dataset = self.load_dataset(info, 'test', 'rgb')

		self.id_list = []
		for (scene_id, im_id, obj_id), instances in self.dataset['instance_data'].items():
			for inst_id, _ in instances.items():
				self.id_list.append([scene_id, im_id, obj_id, inst_id])

def get_texture_mesh_with_trimesh(path):
	mesh = trimesh.load_mesh(path)
	vertices = torch.FloatTensor(mesh.vertices)
	faces = torch.Tensor(mesh.faces)
	faces = faces.type(torch.int32)
	uv_coordinates = torch.FloatTensor(mesh.visual.uv)
	verts_uvs = uv_coordinates[None,...]
	faces_uvs = faces[None,...]
	ply_model = load_ply(path)
	texture_path = f'{os.path.dirname(path)}/{ply_model["texture_file"]}'
	texture = transforms.ToTensor()(Image.open(texture_path))
	texture = torch.unsqueeze(texture, 0)
	texture = torch.permute(texture, (0, 2, 3, 1))
	tex = TexturesUV(maps=texture, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
	mesh = Meshes(verts=[vertices], faces=[faces], textures=tex)
	return mesh