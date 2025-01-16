import torch
from collections import defaultdict
import torch
from pytorch3d.io import IO, load_objs_as_meshes
from HGPose.utils.geometry import mesh_convert_coord
from bop_toolkit.bop_toolkit_lib.inout import load_json

from pytorch3d.renderer.mesh.textures import TexturesUV, TexturesVertex
from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures.utils import packed_to_list

import os

def set_object(path, obj_list, include_textures=False):
	object_info = load_json(f'{path}/models_info.json')
	obj_param = defaultdict(dict)
	for obj_id in obj_list:
		info = object_info[str(obj_id)]
		mesh = IO().load_mesh(f'{path}/obj_{obj_id:06}.ply', include_textures=include_textures, device='cuda')
		scale = float(torch.norm(mesh.verts_packed(), dim=1, p=2).max())
		mesh = mesh.scale_verts(1/scale)
		pts = mesh.verts_packed()
		obj_param[obj_id]['mesh'] = mesh
		obj_param[obj_id]['xyz_mesh'] = mesh_convert_coord(mesh.clone())
		obj_param[obj_id]['pts'] = pts.cpu().numpy()
		obj_param[obj_id]['diameter'] = info['diameter'] / scale
		x_min, y_min, z_min = info['min_x'], info['min_y'], info['min_z']
		x_max, y_max, z_max = x_min+info['size_x'], y_min+info['size_y'], y_min+info['size_y']
		obj_param[obj_id]['bbox_3d'] = torch.tensor([
			[x_min, y_min, z_min], [x_max, y_min, z_min], 
			[x_max, y_max, z_min], [x_min, y_max, z_min],
			[x_min, y_min, z_max], [x_max, y_min, z_max], 
			[x_max, y_max, z_max], [x_min, y_max, z_max]]).to(torch.float) / scale
		obj_param[obj_id]['symmetry'] = ('symmetries_discrete' in info.keys() or 'symmetries_continuous' in info.keys())
		obj_param[obj_id]['scale'] = scale
	return obj_param


def set_object_GSO(path, obj_list):
	obj_param = defaultdict(dict)
	for obj in obj_list:
		obj_id, gso_id = obj['obj_id'], obj['gso_id']
		mesh = load_objs_as_meshes([f'{path}/model/{gso_id}/meshes/model.obj'])
		scale = float(torch.norm(mesh.verts_packed(), dim=1, p=2).max())
		mesh = mesh.scale_verts(1/scale)
		obj_param[obj_id]['mesh'] = mesh
		obj_param[obj_id]['xyz_mesh'] = mesh_convert_coord(mesh.clone())	
		obj_param[obj_id]['scale'] = scale
	return obj_param


def set_object_ShapeNet(path):
	obj_mapping = load_json(f'{path}/mapping.json')
	obj_param = defaultdict(dict)
	for obj in obj_mapping:
		obj_id = obj['obj_id']
		sysnet_id = obj['shapenet_synset_id']
		source_id = obj['shapenet_source_id']
		mesh = IO().load_mesh(f'{path}/model/ShapeNetCore.v2/{sysnet_id}/{source_id}/models/model_normalized.obj')
		scale = float(torch.norm(mesh.verts_packed(), dim=1, p=2).max())
		obj_param[obj_id]['mesh'] = mesh.scale_verts(1/scale)
		obj_param[obj_id]['xyz_mesh'] = mesh_convert_coord(obj_param[obj_id]['mesh'])				
		pts = obj_param[obj_id]['mesh'].verts_packed()
		obj_param[obj_id]['pts'] = pts
		# farthest_pts, farthest_idx = sample_farthest_points(pts.unsqueeze(0), K=N_region)
		# obj_param[obj_id]['region_center'] = farthest_pts
		# obj_param[obj_id]['diameter'] = info['diameter'] / scale
		# x_min, y_min, z_min = info['min_x'], info['min_y'], info['min_z']
		# x_max, y_max, z_max = x_min+info['size_x'], y_min+info['size_y'], y_min+info['size_y']
		# obj_param[obj_id]['bbox_3d'] = torch.tensor([
		# 	[x_min, y_min, z_min], [x_max, y_min, z_min], 
		# 	[x_max, y_max, z_min], [x_min, y_max, z_min],
		# 	[x_min, y_min, z_max], [x_max, y_min, z_max], 
		# 	[x_max, y_max, z_max], [x_min, y_max, z_max]]).to(torch.float) / scale
		obj_param[obj_id]['symmetry'] = False
		obj_param[obj_id]['scale'] = scale
	return obj_param

def normalize_mesh(mesh, scale=0.1):
	pts = mesh.verts_packed()
	center_x = (pts[:, 0].max() + pts[:, 0].min()) / 2
	center_y = (pts[:, 1].max() + pts[:, 1].min()) / 2
	center_z = (pts[:, 2].max() + pts[:, 2].min()) / 2
	offset = -torch.tensor([center_x, center_y, center_z])
	offset = offset.to(mesh.device).view(1,3).repeat(pts.shape[0], 1)
	mesh = mesh.offset_verts(offset)
	max_ = float(mesh.verts_packed().max())
	mesh = mesh.scale_verts(1/max_ * scale)
	return mesh

def convert_to_textureVertex(textures_uv: TexturesUV, meshes:Meshes) -> TexturesVertex:
	verts_colors_packed = torch.zeros_like(meshes.verts_packed())
	verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()  # (*)
	return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))
