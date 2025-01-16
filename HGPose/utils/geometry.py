import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import cv2
import math
from scipy.spatial.transform import Rotation
from kornia.geometry.transform import crop_and_resize
from pytorch3d.renderer import RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader, look_at_rotation
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.renderer.camera_conversions import _opencv_from_cameras_projection, _cameras_from_opencv_projection
from pytorch3d.renderer.lighting import AmbientLights
from pytorch3d.transforms import rotation_6d_to_matrix, axis_angle_to_matrix, acos_linear_extrapolation, so3_relative_angle
from pytorch3d.transforms.rotation_conversions import random_rotations
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.ops import sample_farthest_points, efficient_pnp
from pytorch3d.utils import ico_sphere
from scipy.spatial.transform import Rotation
from kornia.geometry.transform import remap
from kornia.utils import create_meshgrid
import kornia

def pnp_ransac(p3d, p2d, K, thr=1, is_compensation=False):     # https://github.dev/kirumang/Pix2Pose/tree/9c553cee3b845516d3a.0769382fad2547bceabae/tools
	ret, rvec, t, inliers = cv2.solvePnPRansac(
		p3d, p2d, K, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=thr, iterationsCount=200)
	R = np.eye(3)
	cv2.Rodrigues(rvec, R)
	RT = R_T_to_RT(R[np.newaxis, ...], t[np.newaxis, ...])[0].astype(np.float32)
	if is_compensation:
		compensation = [[0, 2], [0, 3], [1, 2], [1, 3], [2, 0], [2, 1]]
		for x, y in compensation:
			RT[x, y] = -RT[x, y]
	if inliers is None:
		mask_index = 0
		confidence = 0.0
	else:     
		mask_index = p2d[inliers[:, 0]]
		confidence = len(inliers)/p3d.shape[0]
	return RT, confidence, mask_index

def xyz2pose(coordinate, mask, K, thr=1):
	_, valid_y, valid_x = torch.where(mask > 0.1)
	inlier_mask = torch.zeros_like(mask[0])
	if len(valid_x) > 4:
		point_2d = torch.stack((valid_x, valid_y), dim=1).to(torch.float)
		point_3d = coordinate[:, valid_y, valid_x].permute(1, 0)
		RT, confidence, mask_index = pnp_ransac(
			point_3d.detach().cpu().numpy().astype(np.float32), 
			point_2d.detach().cpu().numpy().astype(np.float32), 
			K.detach().cpu().numpy().astype(np.float32),
			thr)
		for idx in mask_index:
			inlier_mask[int(idx[1]), int(idx[0])] = 1
	else:
		RT = np.eye(4, 4)
		confidence = 0.0
	inlier_mask = F.interpolate(inlier_mask[None][None], size=[512, 512], mode='nearest')[0, 0]
	inlier_mask = inlier_mask.detach().cpu().numpy()
	inlier_mask = np.tile(inlier_mask[..., np.newaxis], (1, 1, 3))
	return torch.tensor(RT), confidence, inlier_mask

def camera_to_KRT(cameras):
	R, T, K = _opencv_from_cameras_projection(cameras, cameras.image_size)
	RT = R_T_to_RT(R, T)
	return K, RT

def KRT_to_camera(K, RT, image_size):
	cameras = _cameras_from_opencv_projection(RT[..., :3, :3], RT[..., :3, 3], K, image_size)
	return cameras

def R_T_to_RT(R, T, is_pytorch3d=False):
	if R.dtype == np.float64 or R.dtype == np.float32:
		RT = np.concatenate((R, T), axis=-1)
		const = np.zeros_like(RT)[..., [0], :]
		const[..., 0, -1] = 1
		RT = np.concatenate((RT, const), axis=-2)
	else:
		if is_pytorch3d:
			if T.shape[-1] != 1:
				T = T.unsqueeze(-1)
			T = T.transpose(-1, -2)
			RT = torch.cat((R, T), dim=-2)
			const = torch.zeros_like(RT)[..., [0]]
			const[..., -1, 0] = 1
			RT = torch.cat((RT, const), dim=-1)
		else:
			if T.shape[-1] != 1:
				T = T.unsqueeze(-1)
			RT = torch.cat((R, T), dim=-1)
			const = torch.zeros_like(RT)[..., [0], :]
			const[..., 0, -1] = 1
			RT = torch.cat((RT, const), dim=-2)
	return RT

def get_medoid(K, RT, ref_size):
	bsz, n_inp = RT.shape[0:2]
	K = K.expand(bsz, n_inp, 3, 3)
	grid = object_space_grid(K.flatten(0, 1), RT.flatten(0, 1), [ref_size, ref_size], [2, 2])
	grid = grid.unflatten(0, (bsz, n_inp)).flatten(-4, -1)
	medoid_idx = torch.cdist(grid, grid).sum(2).min(1).indices
	medoid_RT = torch.cat([RT[[i], idx] for i, idx in enumerate(medoid_idx)], dim=0).unsqueeze(1)
	return medoid_RT

def get_separate_medoid(RT):
	bsz, n_inp = RT.shape[0:2]
	medoid_RT = RT.clone()
	R, T = RT[..., :3, :3], RT[..., :3, 3]
	R_tile = R.tile(1, n_inp, 1, 1).flatten(0, 1)
	R_repeat = R.repeat_interleave(n_inp, dim=1).flatten(0, 1)
	R_dist = so3_relative_angle(R_tile, R_repeat, eps=1e-1).unflatten(0, (bsz, n_inp, n_inp))
	R_idx = R_dist.sum(2).min(1).indices
	T_idx = torch.cdist(T, T).sum(2).min(1).indices
	for i, (r_idx, t_idx) in enumerate(zip(R_idx, T_idx)):
		medoid_RT[i, 0, :3, :3] = R[i, r_idx]
		medoid_RT[i, 0, :3, 3] = T[i, t_idx]
	medoid_RT = medoid_RT[:, 0].unsqueeze(1)
	return medoid_RT

def get_average(RT):
	average_RT = RT.clone()[:, [0]]
	R, T = RT[..., :3, :3], RT[..., :3, 3]
	R_mean = rotation_6d_to_matrix(R.mean(1, keepdims=True)[..., :2, :3].flatten(-2, -1))
	average_RT[..., [0], :3, :3] = R_mean
	average_RT[..., [0], :3, 3] = T.mean(1, keepdims=True)
	return average_RT

def get_best_candidate(RT, RT_gt):
	bsz, n_inp = RT.shape[0:2]
	best_RT = RT_gt.clone()
	R, T = RT[..., :3, :3], RT[..., :3, 3]
	R_gt, T_gt = RT_gt[..., :3, :3], RT_gt[..., :3, 3]
	R_tile = R.flatten(0, 1)
	R_repeat = R_gt.repeat_interleave(n_inp, dim=1).flatten(0, 1)
	R_dist = so3_relative_angle(R_tile, R_repeat, eps=1e-1).unflatten(0, (bsz, n_inp))
	R_idx = R_dist.min(1).indices
	T_idx = torch.cdist(T, T_gt).min(1).indices
	for i, (r_idx, t_idx) in enumerate(zip(R_idx, T_idx)):
		best_RT[i, 0, :3, :3] = R[i, r_idx]
		best_RT[i, 0, :3, 3] = T[i, t_idx]
	return best_RT

def default_RT(K):
	shape = K.shape[:-2]
	RT = torch.eye(4, 4).to(torch.float).to(K.device)
	RT = RT.view(*[1 for i in range(len(shape))], 4, 4).repeat(*shape, 1, 1)
	return RT

def default_K(img_size=128, focal_length=128):
	K = torch.tensor([
		[focal_length, 0, img_size/2],
		[0, focal_length, img_size/2],
		[0, 0, 1]])
	return K

def random_RT(K, ref_size, n_views=4):
	RT = default_RT(K)
	fxfy = K[..., [0, 1], [0, 1]]
	z_from_dx = fxfy[..., [0]] * 2 / ref_size
	z_from_dy = fxfy[..., [1]] * 2 / ref_size
	scale = (z_from_dy + z_from_dx) / 2
	RT[..., :3, 3] = torch.tensor([0, 0, 1]).to(RT.device).view(1, 1, 3).repeat(1, n_views, 1) * scale
	RT[..., :3, :3] = random_rotations(n_views, device=RT.device).unsqueeze(0)
	return RT

def eval_rot_error(gt_r:np.ndarray, pred_r:np.ndarray):
	error_cos = np.trace(np.matmul(pred_r, np.linalg.inv(gt_r)), axis1=-2, axis2=-1)
	error_cos = 0.5 * (error_cos - 1.0)
	error_cos = np.clip(error_cos, a_min=-1.0, a_max=1.0)
	error =  np.arccos(error_cos)
	error = 180.0 * error / np.pi
	return error

def jitter_RT_with_scale(RT, scale, angle_limit=45, translation_limit=200, 
		   jitter_angle_dis=(0, 15), jitter_x_dis=(0, 15), jitter_y_dis=(0, 15), jitter_z_dis=(0, 50)):
	rotation, translation = RT[..., :3, :3].cpu().numpy(), RT[..., :3, 3].cpu().numpy()
	size = rotation.shape[:-2]
	found_proper_jitter_flag = False
	while not found_proper_jitter_flag:
		angle = np.random.uniform(jitter_angle_dis[0], jitter_angle_dis[1], size=[*size, 3])
		delta_rotation = Rotation.from_euler('zyx', angle, degrees=True).as_matrix().astype(np.float32)
		jittered_rotation = np.matmul(delta_rotation, rotation)
		rotation_error = eval_rot_error(rotation, jittered_rotation)
		if rotation_error.max() > angle_limit:
			continue
		# translation jitter
		x_noise = np.random.uniform(jitter_x_dis[0]/scale, jitter_x_dis[1]/scale, size=[*size, 1])
		y_noise = np.random.uniform(jitter_y_dis[0]/scale, jitter_y_dis[1]/scale, size=[*size, 1])
		z_noise = np.random.uniform(jitter_z_dis[0]/scale, jitter_z_dis[1]/scale, size=[*size, 1])
		translation_noise = np.concatenate([x_noise, y_noise, z_noise], axis=-1)
		translation_error = np.linalg.norm(translation_noise, axis=-1)
		if translation_error.max() > translation_limit/scale:
			continue
		jittered_translation = translation + translation_noise
		noisy_RT = R_T_to_RT(jittered_rotation, jittered_translation[..., np.newaxis])
		noisy_RT = torch.tensor(noisy_RT).to(RT.device).to(torch.float)
		return noisy_RT
	
# def jitter_RT(RT, angle_limit=90, translation_limit=1, 
# 		   jitter_angle_dis=(-90, 90), jitter_x_dis=(-0.02, 0.02), jitter_y_dis=(-0.02, 0.02), jitter_z_dis=(-0.1, 0.1)):
# 	rotation, translation = RT[..., :3, :3].cpu().numpy(), RT[..., :3, 3].cpu().numpy()
# 	size = rotation.shape[:-2]
# 	found_proper_jitter_flag = False
# 	while not found_proper_jitter_flag:
# 		angle = np.random.uniform(jitter_angle_dis[0], jitter_angle_dis[1], size=[*size, 3])
# 		delta_rotation = Rotation.from_euler('zyx', angle, degrees=True).as_matrix().astype(np.float32)
# 		jittered_rotation = np.matmul(delta_rotation, rotation)
# 		rotation_error = eval_rot_error(rotation, jittered_rotation)
# 		if rotation_error.max() > angle_limit:
# 			continue
# 		# translation jitter
# 		x_noise = np.random.uniform(jitter_x_dis[0], jitter_x_dis[1], size=[*size, 1])
# 		y_noise = np.random.uniform(jitter_y_dis[0], jitter_y_dis[1], size=[*size, 1])
# 		z_noise = np.random.uniform(jitter_z_dis[0], jitter_z_dis[1], size=[*size, 1])
# 		translation_noise = np.concatenate([x_noise, y_noise, z_noise], axis=-1)
# 		translation_error = np.linalg.norm(translation_noise, axis=-1)
# 		if translation_error.max() > translation_limit:
# 			continue
# 		jittered_translation = translation + translation_noise
# 		noisy_RT = R_T_to_RT(jittered_rotation, jittered_translation[..., np.newaxis])
# 		noisy_RT = torch.tensor(noisy_RT).to(RT.device).to(torch.float)
# 		return noisy_RT

def RT_from_boxes(boxes_2d, K):
	shape = boxes_2d.shape[:-1]
	fxfy = K[..., [0, 1], [0, 1]]
	cxcy = K[..., [0, 1], [2, 2]]
	RT = torch.eye(4, 4).to(torch.float).to(boxes_2d.device)
	RT = RT.view(*[1 for i in range(len(shape))], 4, 4).repeat(*shape, 1, 1)
	bb_xy_centers = (boxes_2d[..., [0, 1]] + boxes_2d[..., [2, 3]]) / 2
	deltax_3d = 2   
	deltay_3d = 2   
	bb_deltax = (boxes_2d[..., [2]] - boxes_2d[..., [0]])
	bb_deltay = (boxes_2d[..., [3]] - boxes_2d[..., [1]])
	z_from_dx = fxfy[..., [0]] * deltax_3d / bb_deltax
	z_from_dy = fxfy[..., [1]] * deltay_3d / bb_deltay
	z = (z_from_dy + z_from_dx) / 2
	xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
	RT[..., :2, 3] = xy_init
	RT[..., 2, 3] = z.flatten()
	return RT


def apply_imagespace_prev(K, RT, relative, input_size):
	RT_out = RT.clone()
	dR = rotation_6d_to_matrix(relative[..., 0:6]).transpose(-1, -2)
	zsrc = RT[..., [2], 3]
	vxvyvz = relative[..., -3:]
	vz = vxvyvz[..., [2]]
	ztgt = (vz + 1) * zsrc
	vx = vxvyvz[..., 0] * input_size[1]
	vy = vxvyvz[..., 1] * input_size[0]
	vxvy = torch.stack([vx, vy], dim=-1)
	fxfy = K[..., [0, 1], [0, 1]]
	xsrcysrc = RT[..., :2, 3]
	RT_out[..., [2], 3] = ztgt
	RT_out[..., :2, 3] = ((vxvy / fxfy) + (xsrcysrc / zsrc)) * ztgt.clone() ### 논문에서는 fxfy가1이므로 scale issue
	RT_out[..., :3, :3] = dR @ RT[..., :3, :3].clone()
	return RT_out

def get_pose_from_delta_pose_scflow(rotation_delta, translation_delta, rotation_src, translation_src, weight=10., depth_transform='exp', detach_depth_for_xy=False):
	'''Get transformed pose
	Args:
		rotation_delta (Tensor): quaternion to represent delta rotation shape (n, 4)(Quaternions) or (n, 6)(orth 6D )
		translation_delta (Tensor): translation to represent delta translation shape (n, 3)
		rotation_src (Tensor): rotation matrix to represent source rotation shape (n, 3, 3)
		translation_src (Tensor): translation vector to represent source translation shape (n, 3)
	'''
	if rotation_delta.size(1) == 4:
		rotation_delta = kornia.geometry.conversions.quaternion_to_rotation_matrix(rotation_delta)
	else:
		rotation_delta = get_rotation_matrix_from_ortho6d(rotation_delta)
	rotation_dst = torch.bmm(rotation_delta, rotation_src)
	if depth_transform == 'exp':
		vz = torch.div(translation_src[:, 2], torch.exp(translation_delta[:, 2]))
	else:
		# vz = torch.div(translation_src[:, 2], translation_delta[:, 2] + 1)
		vz = translation_src[:, 2] * (translation_delta[:, 2] + 1)
	if detach_depth_for_xy:
		vx = torch.mul(vz.detach(), torch.addcdiv(translation_delta[:, 0] / weight, translation_src[:, 0], translation_src[:, 2]))
		vy = torch.mul(vz.detach(), torch.addcdiv(translation_delta[:, 1] / weight, translation_src[:, 1], translation_src[:, 2]))
	else:
		vx = torch.mul(vz, torch.addcdiv(translation_delta[:, 0] / weight, translation_src[:, 0], translation_src[:, 2]))
		vy = torch.mul(vz, torch.addcdiv(translation_delta[:, 1] / weight, translation_src[:, 1], translation_src[:, 2]))
	translation_dst = torch.stack([vx, vy, vz], dim=-1)
	return rotation_dst, translation_dst

def get_rotation_matrix_from_ortho6d(ortho6d):
	'''
	https://github.com/papagina/RotationContinuity/blob/sanity_test/code/tools.py L47
	'''
	x_raw = ortho6d[:,0:3]#batch*3
	y_raw = ortho6d[:,3:6]#batch*3
		
	x = F.normalize(x_raw, p=2, dim=1) #batch*3
	z = torch.cross(x, y_raw, dim=1)
	z = F.normalize(z, p=2, dim=1)
	y = torch.cross(z, x, dim=1)#batch*3
		
	x = x.view(-1,3,1)
	y = y.view(-1,3,1)
	z = z.view(-1,3,1)
	matrix = torch.cat((x,y,z), 2) #batch*3*3
	return matrix

def apply_imagespace_relative(K1, K2, RT1, relative, input_size):
	RT2 = RT1.clone()
	fxfy1 = K1[..., [0, 1], [0, 1]]    
	fxfy2 = K2[..., [0, 1], [0, 1]]
	f1 = fxfy1.mean(dim=-1, keepdim=True)
	f2 = fxfy2.mean(dim=-1, keepdim=True)
	r = f2 / f1
	allo_dR = rotation_6d_to_matrix(relative[..., 0:6]).transpose(-1, -2)
	t1 = RT1[..., :3, 3].clone()
	o1_ = (K1 @ t1.unsqueeze(-1))[..., 0]
	o1 = (o1_ / o1_[..., [2]])
	zsrc = t1[..., [2]]
	vxvyvz = relative[..., -3:]
	vz = vxvyvz[..., [2]]
	ztgt = r * (vz + 1) * zsrc
	vx = vxvyvz[..., 0] * input_size[1]
	vy = vxvyvz[..., 1] * input_size[0]
	zero = torch.zeros_like(vx)
	vxvy = torch.stack([vx, vy, zero], dim=-1)
	RT2[..., [2], 3] = ztgt
	xtgtytgt = ((K2.inverse() @ (vxvy + o1).unsqueeze(-1))[..., 0] * ztgt.clone())[..., :2]
	RT2[..., :2, 3] = xtgtytgt
	allo_R1 = ego_R_to_allo_R(RT1[..., :3, :3], RT1[..., :3, 3])
	allo_R2 = allo_dR @ allo_R1.to(allo_dR.dtype).clone().detach()
	ego_R2 = allo_R_to_ego_R(allo_R2, RT2[..., :3, 3].clone().detach())
	RT2[..., :3, :3] = ego_R2
	return RT2

def ego_R_to_allo_R(ego_R, ego_t):
	cam_ray = torch.zeros_like(ego_t)
	cam_ray[..., 2] = cam_ray[..., 2] + 1.0
	R_view = vectors_to_R(ego_t, cam_ray)
	allo_R = R_view @ ego_R
	return allo_R

def allo_R_to_ego_R(allo_R, ego_t):
	cam_ray = torch.zeros_like(ego_t)
	cam_ray[..., 2] = cam_ray[..., 2] + 1.0
	R_view = vectors_to_R(cam_ray, ego_t)
	ego_R = R_view @ allo_R
	return ego_R

def vectors_to_R(v1, v2):
	v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
	v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
	axis = F.normalize(torch.linalg.cross(v1, v2, dim=-1), dim=-1)
	angle = acos_linear_extrapolation((v1 * v2).sum(dim=-1, keepdim=True)) + 1e-8
	axis_angle = axis * angle
	R = axis_angle_to_matrix(axis_angle)
	return R

def positive_RT(RT_src, RT_tgt, k):
	bsz, N_template = RT_src.shape[:2]
	allo_R_src = ego_R_to_allo_R(RT_src[..., :3, :3], RT_src[..., :3, 3])
	allo_R_tgt = ego_R_to_allo_R(RT_tgt[..., :3, :3], RT_tgt[..., :3, 3]).expand(bsz, N_template, 3, 3)
	dist_R = torch.vstack([so3_relative_angle(a_R_s, a_R_t, eps=1e-2) for a_R_s, a_R_t in zip(allo_R_src, allo_R_tgt)]) #eps=1e-5
	positive_idxes = torch.vstack([torch.topk(-d_R, k=k)[1] for d_R in dist_R]).to(torch.int64)
	return positive_idxes

def negative_RT(RT_src, RT_tgt, k):
	bsz, N_template = RT_src.shape[:2]
	allo_R_src = ego_R_to_allo_R(RT_src[..., :3, :3], RT_src[..., :3, 3])
	allo_R_tgt = ego_R_to_allo_R(RT_tgt[..., :3, :3], RT_tgt[..., :3, 3]).expand(bsz, N_template, 3, 3)
	dist_R = torch.vstack([so3_relative_angle(a_R_s, a_R_t, eps=1e-5) for a_R_s, a_R_t in zip(allo_R_src, allo_R_tgt)])
	negative_idxes = torch.vstack([torch.topk(d_R, k=k)[1] for d_R in dist_R]).to(torch.int64)
	return negative_idxes

def get_ref_RT(ref_all_RT, ref_idx):
	bsz, N_ref = ref_idx.shape[:2]
	ref_idx = ref_idx.view(bsz, N_ref, 1, 1).expand(bsz, N_ref, 4, 4).cuda()
	ref_RT = ref_all_RT.gather(1, ref_idx)
	return ref_RT

def get_K_crop_resize(K, boxes, crop_resize):
	new_K = K.clone()
	w = boxes[..., 2] - boxes[..., 0]
	h = boxes[..., 3] - boxes[..., 1]
	W = crop_resize[1]
	H = crop_resize[0]
	x = boxes[..., 0]
	y = boxes[..., 1]
	new_K[..., 0, 0] = K[..., 0, 0] * W/w               # fx -> fx * W/w
	new_K[..., 1, 1] = K[..., 1, 1] * H/h               # fy -> fy * H/h
	new_K[..., 0, 2] = (K[..., 0, 2] - x) * W/w         # cx -> (cx - x) * W/w
	new_K[..., 1, 2] = (K[..., 1, 2] - y) * H/h         # cy -> (cy - y) * H/h
	return new_K

def bbox_add_inner_noise(bbox, std_rate=0.1):
	### from https://github.com/ylabbe/cosypose/cosypose/lib3d/transform.py
	"""
	bbox : ... x [x1, y1, x2, y2]
	noisy_bbox : ... x [x1, y1, x2, y2]
	noisy bbox inside given bbox
	"""
	device = bbox.device
	bbox_size = bbox[..., 2:4] - bbox[..., 0:2]
	bbox_std = torch.cat((bbox_size * std_rate, bbox_size * std_rate), -1)
	noisy_bbox = torch.normal(bbox, bbox_std).to(device)
	compare_box = torch.stack([bbox, noisy_bbox], dim=-2)
	noisy_bbox[..., 0:2] = torch.max(compare_box[..., 0:2], dim=-2).values
	noisy_bbox[..., 2:4] = torch.min(compare_box[..., 2:4], dim=-2).values
	return noisy_bbox

def bbox_add_noise(bbox, std_rate=0.1):
	### from https://github.com/ylabbe/cosypose/cosypose/lib3d/transform.py
	"""
	bbox : ... x [x1, y1, x2, y2]
	noisy_bbox : ... x [x1, y1, x2, y2]
	"""
	device = bbox.device
	bbox_size = bbox[..., 2:4] - bbox[..., 0:2]
	bbox_std = torch.cat((bbox_size * std_rate, bbox_size * std_rate), -1)
	noisy_bbox = torch.normal(bbox, bbox_std).to(device)
	return noisy_bbox

def squaring_boxes(obs_boxes, lamb=1.0, min_or_max='max'):
	centers = ((obs_boxes[..., [0, 1]] + obs_boxes[..., [2, 3]])/2)
	xc, yc = centers[..., [0]], centers[..., [1]]
	lobs, robs, uobs, dobs = obs_boxes[..., [0]], obs_boxes[..., [2]], obs_boxes[..., [1]], obs_boxes[..., [3]]
	xdist = torch.abs(lobs - xc)
	ydist = torch.abs(uobs - yc)
	if min_or_max=='max':
		size = torch.max(torch.cat([xdist, ydist], -1), dim=-1, keepdims=True).values * 2.0 * lamb
	else:
		size = torch.min(torch.cat([xdist, ydist], -1), dim=-1, keepdims=True).values * 2.0 * lamb
	x1, y1, x2, y2 = xc - size/2, yc - size/2, xc + size/2, yc + size/2
	boxes = torch.cat([x1, y1, x2, y2], -1).to(obs_boxes.device)
	return boxes

def image_cropping(img_feature, bboxes_crop, output_size, mode='bilinear'):
	corners = box_to_corners(bboxes_crop)
	cropped = crop_and_resize(input_tensor=img_feature.view(-1, *img_feature.shape[-3:]), 
							  boxes=corners.view(-1, *corners.shape[-2:]), 
							  size=tuple(output_size),
							  mode=mode)
	cropped = cropped.view(*img_feature.shape[:-2], *output_size)
	return cropped

def box_to_corners(bbox):
	x_min, y_min, x_max, y_max = bbox[...,0], bbox[...,1], bbox[...,2], bbox[...,3]
	bbox_2d = [torch.stack([x_min, y_min], dim=-1), torch.stack([x_max, y_min], dim=-1), torch.stack([x_max, y_max], dim=-1), torch.stack([x_min, y_max], dim=-1)]
	bbox_2d = torch.stack(bbox_2d, dim=-2)
	return bbox_2d

def object_space_grid(K, RT, image_size, feature_size, N_z=2):
	device = K.device
	N = K.shape[0]
	f, px, py = (K[..., 0, 0] + K[..., 1, 1])/2, K[..., 0, 2], K[..., 1, 2]
	H, W = image_size
	h, w = feature_size

	# generate 3D image plane
	XY = create_meshgrid(H, W, False, device)
	XY = F.interpolate(XY.permute(0, 3, 1, 2), [h, w], mode='bilinear').permute(0, 2, 3, 1)
	X = XY[..., 0] - px.view(N, 1, 1)
	Y = XY[..., 1] - py.view(N, 1, 1)
	Z = torch.ones_like(X) * f.view(N, 1, 1)
	XYZ = torch.stack((X, Y, Z), -1)

	# generate 3D grid on the ray from camera to image plane
	L = torch.norm(XYZ, dim=-1).unsqueeze(-1)
	normalized_XYZ = XYZ / L
	steps = torch.linspace(-1, 1, N_z, device=device).view(1, N_z, 1, 1, 1).to(normalized_XYZ.device)
	cropped_grid = steps * normalized_XYZ.unsqueeze(1)

	# camera space grid
	dist = torch.norm(RT[..., :3, 3], 2, -1).view(RT.shape[0], 1, 1, 1, 1)
	pushing_direction = F.normalize(cropped_grid[:,[-1],...], dim=4)
	camera_space_grid = cropped_grid + pushing_direction * dist
	
	# trasnform camera space to object space
	grid = camera_space_grid.flatten(1, 3)
	object_space_grid = transform_pts(RT, grid, inverse=True).reshape(camera_space_grid.shape)
	return object_space_grid

def transform_pts(RT, pts, inverse=False):
	shape = pts.shape[:-1]
	ones = torch.ones(*shape, 1, dtype=pts.dtype, device=pts.device)
	pts = torch.cat([pts, ones], dim=-1).transpose(-2, -1)
	RT = RT.inverse() if inverse else RT
	pts = torch.bmm(RT, pts).transpose(-2, -1)
	pts = pts / pts[..., [3]]
	pts = pts[..., :3].reshape(*shape, 3)
	return pts

def get_2d_coord(size):
	xs = torch.linspace(0, size[1]-1, size[1])
	ys = torch.linspace(0, size[0]-1, size[0])
	meshgrid = torch.meshgrid(ys, xs)
	xy = torch.cat((meshgrid[1], meshgrid[0]), 0)
	return xy  

def farthest_rotation_sampling(ref_N, dataset, nth_ref=0):
	farthest_keys = []
	Rs = torch.tensor(np.stack([data['RT'][:3, :3] for data in dataset.values()]))
	sorted_data = sorted(dataset.items(), key=lambda x: x[1]['px_count_visib'])
	initial_key, initial_data = sorted_data[-nth_ref][0], sorted_data[-nth_ref][1]
	farthest_keys += [initial_key]
	initial_R = torch.tensor(initial_data['RT'][:3, :3]).unsqueeze(0).repeat(Rs.shape[0], 1, 1)
	distances = so3_relative_angle(initial_R, Rs, eps=1e-1)
	for _ in range(1, ref_N):
		next_index = torch.argmax(distances).item()
		selected_key, selected_data = list(dataset.items())[next_index]
		farthest_keys += [selected_key]
		farthest_R = torch.tensor(selected_data['RT'][:3, :3]).unsqueeze(0).repeat(Rs.shape[0], 1, 1)
		distances = torch.minimum(distances, so3_relative_angle(farthest_R, Rs, eps=1e-1))
	return farthest_keys

def carving_feature(masks, RT, K_crop, ref_size):
	img_size = masks.shape[-1]
	N_ref = masks.shape[0]
	index_3d = torch.zeros([ref_size, ref_size, ref_size, 3])
	idx = torch.linspace(0, img_size, ref_size)
	index_3d[..., 0], index_3d[..., 1], index_3d[..., 2] = torch.meshgrid(idx, idx, idx)
	normalized_idx = (index_3d - img_size/2)/(img_size/2)
	X = normalized_idx.reshape(1, -1, 3).repeat(N_ref, 1, 1)
	homogeneous_X = torch.cat((X, torch.ones(X.shape[0], X.shape[1], 1)), 2).transpose(1, 2).to(RT.device)
	xyz_KRT = torch.bmm(K_crop, torch.bmm(RT[:, :3, :], homogeneous_X))
	xyz = (xyz_KRT/xyz_KRT[:, [2], :]).transpose(1, 2).reshape(N_ref, ref_size, ref_size, ref_size, 3)
	xyz[..., :2] = (xyz[..., :2] - img_size/2)/(img_size/2)
	xyz[... ,2] = 0
	masks_3d = (masks.unsqueeze(2) > 0.5).to(torch.float)
	ref_mask_3d = F.grid_sample(masks_3d.to(torch.float), xyz, mode='nearest')
	ref_mask_3d = torch.prod(ref_mask_3d, 0, keepdim=True)
	ref_3d = (index_3d / img_size).permute(3, 0, 1, 2).unsqueeze(0)
	ref_3d = (ref_3d.to(ref_mask_3d.device) + 1) * ref_mask_3d - 1
	ref = torch.cat([ref_3d.transpose(2, 4),ref_mask_3d.transpose(2, 4)], dim=1)                    # XYZ to ZYX (DHW)
	return ref

def z_buffer_min(NDC_ref, NDC_mask):
	bsz, c, N_z, h, w = NDC_ref.shape
	z_grid = torch.arange(N_z-1, -1, -1).view(1, 1, N_z, 1, 1).to(NDC_ref.device)   # bsz, d, h, w
	index = NDC_mask * z_grid
	index = torch.max(index, 2, keepdim=True).indices.repeat(1, c, 1, 1, 1)
	img = torch.gather(NDC_ref, 2, index).squeeze(2)
	return img

def project_ref(obj_grid, ref):
	####### DProST grid push & transform
	ref_3d, ref_3d_mask = ref[..., :3, :, :, :], ref[..., [3], :, :, :]
	NDC_ref = F.grid_sample(ref_3d, obj_grid, mode='bilinear', align_corners=True)
	NDC_mask = F.grid_sample(ref_3d_mask, obj_grid, mode='bilinear', align_corners=True)
	proj_img = z_buffer_min(NDC_ref, NDC_mask)
	return proj_img

def render_ref(ref, K, RT, img_size, out_size, N_z):
	bsz, N = K.shape[:-2]
	feat_size = ref.shape[-1]
	ref = ref.flatten(0, -5)
	K = K.flatten(0, -3)
	RT = RT.flatten(0, -3)
	obj_grid = object_space_grid(K, RT, [img_size, img_size], [feat_size, feat_size], N_z=N_z)
	rendered = project_ref(obj_grid, ref)
	rendered = F.interpolate(rendered, [out_size, out_size], mode='bilinear')
	rendered = rendered.unflatten(0, (bsz, N))
	rendered = torch.clip(rendered, 0, 1)
	return rendered

def project_mesh(mesh, camera):
	raster_settings = RasterizationSettings(
		image_size=camera.image_size[0].to(torch.int).tolist(),
		max_faces_per_bin=0) #10000
	rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)
	light = AmbientLights(device=mesh.device)
	shader = HardPhongShader(device=mesh.device, cameras=camera, lights=light)
	renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
	coord = renderer(meshes_world=mesh)
	coord = coord[..., :3] * coord[..., [-1]]
	return coord.permute(0, 3, 1, 2)

def mesh_convert_coord(mesh):  
	xyz = mesh.verts_packed()
	mesh.textures = TexturesVertex(verts_features=[(xyz + 1) / 2])          # padding value of texture is -1 so range shuld be 0~1
	return mesh

def render_mesh(mesh_list, K, RT, out_size):
	bsz, n_view = K.shape[0:2]
	size = torch.tensor([out_size, out_size]).view(1, 2).expand(bsz * n_view, 2)
	K = K.flatten(0, -3)
	RT = RT.flatten(0, -3)
	mesh_list = sum([[mesh] * n_view for mesh in mesh_list], [])
	mesh = join_meshes_as_batch(mesh_list)
	camera_batch = KRT_to_camera(K, RT, size)
	rendered = project_mesh(mesh, camera_batch)
	rendered = rendered.reshape(bsz, n_view, 3, out_size, out_size)
	return rendered

def get_ray_direction(RT, coordinate):
	cam_origin = RT.inverse()[..., :3, 3].unsqueeze(-1).unsqueeze(-1)
	rays = coordinate - cam_origin
	rays = rays / rays.norm(dim=-3, keepdim=True)
	return rays

def render_geometry(source, K, RT, img_size=256, out_size=256, N_z=64, n_freqs=5, represent_mode='xyz'):
	if torch.is_tensor(source[0]):
		geometry = render_ref(source, K, RT, img_size, out_size, N_z=N_z)
	else:
		geometry = render_mesh(source, K, RT, out_size)
	mask = geometry.mean(-3, keepdims=True).type(torch.bool).type(torch.float)
	geometry = (geometry * 2) - 1 
	if represent_mode == 'positional':
		geometry = positional_encoding(geometry, n_freqs=n_freqs)
	geometry = ((geometry + 1) * mask) - 1
	geometry = geometry.detach()
	return geometry

def render_mesh_scflow(mesh_list, K, RT, img_size, out_size):
	bsz, n_view = K.shape[0:2]
	size = torch.tensor([out_size, out_size]).view(1, 2).expand(bsz * n_view, 2)
	scale_factor = out_size / img_size
	K_scaled = K.clone()
	K_scaled[..., 0, 0] *= scale_factor  # Scale fx
	K_scaled[..., 1, 1] *= scale_factor  # Scale fy
	K_scaled[..., 0, 2] *= scale_factor  # Scale cx
	K_scaled[..., 1, 2] *= scale_factor  # Scale cy
	K_scaled = K_scaled.flatten(0, -3)
	RT = RT.flatten(0, -3)
	mesh_list = sum([[mesh] * n_view for mesh in mesh_list], [])
	mesh = join_meshes_as_batch(mesh_list)
	camera_batch = KRT_to_camera(K_scaled, RT, size)
	rendered = project_mesh(mesh, camera_batch)
	rendered = rendered.reshape(bsz, n_view, 3, out_size, out_size)
	return rendered

def render_geometry_scflow(source, K, RT, img_size=256, out_size=256, N_z=64, n_freqs=5, represent_mode='xyz'):
	if torch.is_tensor(source[0]):
		geometry = render_ref(source, K, RT, img_size, out_size, N_z=N_z)
	else:
		geometry = render_mesh_scflow(source, K, RT, img_size, out_size)
	mask = geometry.mean(-3, keepdims=True).type(torch.bool).type(torch.float)
	geometry = (geometry * 2) - 1 
	if represent_mode == 'positional':
		geometry = positional_encoding(geometry, n_freqs=n_freqs)
	geometry = ((geometry + 1) * mask) - 1
	geometry = geometry.detach()
	return geometry

def encode_coordinate(coordinate, n_freqs):
	mask = (coordinate + 1).mean(-3, keepdims=True).type(torch.bool).type(torch.float)
	geometry = positional_encoding(coordinate, n_freqs=n_freqs)
	geometry = ((geometry + 1) * mask) - 1
	return geometry

def make_region(coordinate, centers):
	coord = coordinate.unsqueeze(2)
	cent = torch.stack(centers).unsqueeze(-1).unsqueeze(-1).to(coord.device)
	dist = torch.norm(coord-cent, dim=3)
	idx = torch.argmin(dist, dim=2)
	mask = torch.abs((coordinate + 1) / 2).mean(2, keepdim=True).to(torch.bool)
	region = torch.zeros_like(dist).scatter_(2, idx.unsqueeze(2), 1.) * mask
	region = torch.cat([(1-mask.to(torch.float)), region], dim=2)
	return region

def coordinate_distance(coord1, coord2, eps=1e3):
	# coord1 -> (B, Q, 3)
	# coord2 -> (B, P, 3)
	Q = coord1.shape[1]
	P = coord2.shape[1]
	coord1[coord1 == -1.0] = eps
	coord2[coord2 == -1.0] = -eps
	coord1 = coord1.unsqueeze(2).repeat(1, 1, P, 1)  # (B, Q, P, 3)
	coord2 = coord2.unsqueeze(1).repeat(1, Q, 1, 1)  # (B, Q, P, 3)
	distance = torch.norm(coord1-coord2, dim=-1)
	return distance

def interpolate_shapes(f, shapes):
	result = {}
	for k, shape in shapes.items():
		f_shape = F.interpolate(f.flatten(0, -4), shape).unflatten(0, (f.shape[:-3]))
		result[k] = f_shape
	return result

def get_patch_coordinates(coordinates, feature_size):
	batch, numviews, _, H, W = coordinates.shape
	size_y = H // feature_size[0]
	size_x = W // feature_size[1]
	patch_coordinates = coordinates.unfold(-2, size_y, size_y).unfold(-2, size_x, size_x)      # (bsz, n_inp, 3, feature_size[0], feature_size[1], size_y, size_x)
	patch_coordinates = patch_coordinates.permute(0, 1, 3, 4, 2, 5, 6)                         # (bsz, n_inp, feature_size[0], feature_size[1], 3, size_y, size_x)
	patch_coordinates_mean = masked_mean(patch_coordinates)                                    # (bsz, n_inp, feature_size[0], feature_size[1], 3)
	patch_coordinates_mean = patch_coordinates_mean.reshape(batch, numviews * feature_size[0] * feature_size[1], 3)
	return patch_coordinates_mean

def masked_mean(input_img):
	img = (input_img + 1) / 2
	mask = torch.sum(img, dim=-3, keepdim=True).to(torch.bool)
	denom = torch.sum(mask, dim=[-1, -2]).repeat(1, 1, 1, 1, 3)
	non_zero_index = (denom != 0)
	masked_img = torch.zeros(img.shape[:-2]).to(img.device)
	masked_img[non_zero_index] = torch.sum(img * mask, dim=[-1, -2])[non_zero_index] / denom[non_zero_index]
	masked_img = masked_img * 2 - 1
	return masked_img

def get_3d_bbox_from_pts(pts):
  """Calculates 3D bounding box of the given set of 3D points.
  :param pts: N x 3 ndarray with xyz-coordinates of 3D points.
  :return: 3D bounding box (8,2) of bbox 8 points
	  7 -------- 6
	 /|         /|
	4 -------- 5 .
	| |        | |
	. 3 -------- 2
	|/         |/
	0 -------- 1
  """
  xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
  x_min, x_max, y_min, y_max, z_min, z_max = xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max()
  bbox_3d = [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
			 [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]]
  bbox_3d = torch.tensor(bbox_3d)
  return bbox_3d

def get_distmap(coord1, coord2, feature_size, temperature, is_softmax=False):
	c1 = get_patch_coordinates(coord1.clone(), feature_size).detach() 
	c2 = get_patch_coordinates(coord2.clone(), feature_size).detach()
	distmap = -coordinate_distance(c1, c2) / temperature
	if is_softmax:
		distmap = softmax(distmap, dim=-1)
	return c1, c2, distmap

def positional_encoding(coord, n_freqs=10, start_freq=0, is_barf=False, steps=None, total_steps=None):
	freq_bands = 2. ** torch.arange(start_freq, start_freq+n_freqs) * np.pi
	encodings = [torch.cat([torch.cos(coord * freq), torch.sin(coord * freq)], dim=-3) for freq in freq_bands]
	pos_embeddings = torch.cat(encodings, dim=-3)  # B, num_input_views, P, 3 * 2 * n_freqs
	if is_barf:
		alpha = steps / total_steps * n_freqs
		k = torch.arange(n_freqs, dtype=torch.float32, device=pos_embeddings.device) - 1
		weight = (1 - (alpha - k).clamp_(min=0,max=1).mul_(np.pi).cos_()) / 2
		weight = weight.repeat_interleave(6).view(1, 1, n_freqs*3*2, 1, 1)
		pos_embeddings = pos_embeddings * weight
	return pos_embeddings

class PositionEmbeddingSine3D(nn.Module):
	def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
		super().__init__()
		self.num_pos_feats = num_pos_feats
		self.temperature = temperature
		self.normalize = normalize
		if scale is not None and normalize is False:
			raise ValueError("normalize should be True if scale is passed")
		if scale is None:
			scale = 2 * math.pi
		self.scale = scale

	def forward(self, x, mask=None):
		# b, t, c, h, w
		assert x.dim() == 5, f"{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"
		if mask is None:
			mask = torch.zeros((x.size(0), x.size(1), x.size(3), x.size(4)), device=x.device, dtype=torch.bool)
		not_mask = ~mask
		z_embed = not_mask.cumsum(1, dtype=torch.float32)
		y_embed = not_mask.cumsum(2, dtype=torch.float32)
		x_embed = not_mask.cumsum(3, dtype=torch.float32)
		if self.normalize:
			eps = 1e-6
			z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
			y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
			x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

		dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
		dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

		dim_t_z = torch.arange((self.num_pos_feats * 2), dtype=torch.float32, device=x.device)
		dim_t_z = self.temperature ** (2 * (dim_t_z // 2) / (self.num_pos_feats * 2))

		pos_x = x_embed[:, :, :, :, None] / dim_t
		pos_y = y_embed[:, :, :, :, None] / dim_t
		pos_z = z_embed[:, :, :, :, None] / dim_t_z
		pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
		pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
		pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
		pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z).permute(0, 1, 4, 2, 3)  # b, t, c, h, w
		return pos
	
def get_flow_from_delta_pose_and_xyz_scflow(RT_dst, K_geo_dst, geo_src, mask_src=None, geo_dst=None, invalid_flow=400):
	device = geo_src.device
	bsz, n_src, c, H, W = geo_src.shape
	rotation_dst = RT_dst[..., :3, :3].squeeze(1)
	translation_dst = RT_dst[..., :3, 3].squeeze(1)
	points_3d_list = []
	points_2d_list = []
	for i in range(bsz):
		geo = geo_src[i].squeeze()
		matching_indices = (geo[0] != -1).nonzero(as_tuple=True)
		points_3d = geo[:, matching_indices[0], matching_indices[1]].T
		swapped_points_2d = torch.stack(matching_indices[:], dim=-1)
		points_2d = swapped_points_2d.clone()
		points_2d[:, 0], points_2d[:, 1] = swapped_points_2d[:, 1], swapped_points_2d[:, 0]
		points_3d_list.append(points_3d)
		points_2d_list.append(points_2d)
	

	flow = rotation_dst.new_ones((bsz, 2, H, W)) * invalid_flow
	for i in range(bsz):
		points_2d, points_3d = points_2d_list[i], points_3d_list[i]
		points_3d_transpose = points_3d.t()
		points_2d_dst = torch.mm(K_geo_dst[i].squeeze(), torch.mm(rotation_dst[i], points_3d_transpose)+translation_dst[i][:, None]).t()
		points_2d_dst_x, points_2d_dst_y = points_2d_dst[:, 0]/points_2d_dst[:, 2], points_2d_dst[:, 1]/points_2d_dst[:, 2]
		flow_x, flow_y = points_2d_dst_x - points_2d[:, 0], points_2d_dst_y - points_2d[:, 1]
		flow = flow.to(flow_x.dtype)
		flow[i, 0, points_2d[:, 1].to(torch.int64), points_2d[:, 0].to(torch.int64)] = flow_x
		flow[i, 1, points_2d[:, 1].to(torch.int64), points_2d[:, 0].to(torch.int64)] = flow_y
	
	return flow.unsqueeze(1)


def get_flow_from_delta_pose_and_xyz(RT_dst, K_geo_dst, geo_src, mask_src=None, geo_dst=None, invalid_flow=400):
	device = geo_src.device
	bsz, n_src, c, H, W = geo_src.shape
	n_dst = RT_dst.shape[1]
	N = max(n_src, n_dst)
	geo_src = geo_src.expand(bsz, N, c, H, W)[..., :3, :, :]
	RT_dst = RT_dst.expand(bsz, N, 4, 4)
	K_geo_dst = K_geo_dst.expand(bsz, N, 3, 3)

	rotated_3d_pts = torch.einsum('bqij,bqjhw->bqihw', RT_dst[..., :3, :3], geo_src)
	rotated_and_translated_3d_pts = rotated_3d_pts + RT_dst[..., :3, 3].view(bsz, N, 3, 1, 1)
	projected_2d_dst = torch.einsum('bqij,bqjhw->bqihw', K_geo_dst, rotated_and_translated_3d_pts)
	pts_2d_dst_x = projected_2d_dst[..., [0], :, :] / projected_2d_dst[..., [2], :, :]
	pts_2d_dst_y = projected_2d_dst[..., [1], :, :] / projected_2d_dst[..., [2], :, :]
	pts_2d_dst_xy = torch.cat([pts_2d_dst_x, pts_2d_dst_y], dim=-3)

	xy_2d = create_meshgrid(H, W, False, device)
	xy_2d = xy_2d.expand(bsz*N, H, W, 2).permute(0, 3, 1, 2).view(bsz, N, 2, H, W)
	flow_s2d = pts_2d_dst_xy - xy_2d

	# if mask_src == None:
	# 	mask_src = (flow_s2d != invalid_flow).to(flow_s2d)
	flow_s2d = mask_src * flow_s2d + (1-mask_src) * invalid_flow

	# if geo_dst != None:
	# 	recon_geo_src = apply_backward_flow(flow_s2d, geo_dst[..., :3, :, :], invalid_val=-1.0)
	# 	visible_mask = (torch.abs(recon_geo_src - geo_src).mean(dim=-3, keepdim=True) < 5e-2).to(flow_s2d)
	# 	flow_mask = visible_mask * mask_src
	# 	flow_s2d = flow_mask * flow_s2d + (1-flow_mask) * invalid_flow
	# 	return flow_s2d, flow_mask
	
	return flow_s2d


def apply_backward_flow(flow_s2d, dst, invalid_val, invalid_flow=400.0):
	bsz, N, _, H, W = flow_s2d.shape
	c = dst.shape[-3]
	dst, flow_s2d = dst.expand(bsz, N, c, H, W).flatten(0, 1), flow_s2d.flatten(0, 1)
	mask = (flow_s2d[..., [0], :, :] != invalid_flow).to(dst)

	xy_src = create_meshgrid(H, W, False, flow_s2d.device)
	xy_src = xy_src.expand(bsz*N, H, W, 2)
	xy_dst = xy_src + flow_s2d.permute(0, 2, 3, 1)
	x_dst = xy_dst[..., 0]
	y_dst = xy_dst[..., 1]
	x_dst = torch.clip(x_dst, 0, W-1) / (W-1) * 2 - 1
	y_dst = torch.clip(y_dst, 0, H-1) / (H-1) * 2 - 1
	xy_dst = torch.stack([x_dst, y_dst], dim=-1)
	src = F.grid_sample(dst, xy_dst, padding_mode='zeros')
	src = mask * src + (1-mask) * invalid_val
	src = src.unflatten(0, (bsz, N))
	return src


def apply_forward_flow(flow_s2d, src, invalid_val, invalid_flow=400.0):
	bsz, N, _, H, W = flow_s2d.shape
	_, _, c, h, w = src.shape
	src, flow_s2d = src.expand(bsz, N, c, h, w).flatten(0, 1), flow_s2d.flatten(0, 1)
	mask = (flow_s2d[..., 0, :, :] != invalid_flow).to(src)

	dst = torch.ones_like(src) * invalid_val
	idx, y_src, x_src = torch.nonzero(mask, as_tuple=True)
	valid_flow = flow_s2d[idx, :, y_src, x_src]
	valid_src = src[idx, :, y_src, x_src]
	x_dst, y_dst = x_src+valid_flow[:, 0], y_src+valid_flow[:, 1]
	x_dst = torch.clamp(x_dst, min=0, max=W-1)
	y_dst = torch.clamp(y_dst, min=0, max=H-1)
	dst[idx, :, y_dst.to(torch.int64), x_dst.to(torch.int64)] = valid_src
	dst = F.interpolate(dst, [H, W])
	dst = dst.unflatten(0, (bsz, N))
	return dst


def combine_flow(flow_list):
	bsz, N, _, H, W = flow_list[0].shape
	xy = create_meshgrid(H, W, False, flow_list[0].device).unsqueeze(0)
	xy = xy.expand(bsz, N, H, W, 2).permute(0, 1, 4, 2, 3)
	flow = flow_list[0]
	for fl in flow_list:
		xy_loc = apply_forward_flow(fl, xy, invalid_val=0.0)
		xy_loc = xy_loc.permute(0, 1, 3, 4, 2)
		x_loc = (xy_loc[..., 0] - W//2) / (W//2)
		y_loc = (xy_loc[..., 1] - H//2) / (H//2)
		xy_loc = torch.stack([x_loc, y_loc], dim=-1)
		flow = F.grid_sample(flow.flatten(0, 1), xy_loc.flatten(0, 1)).unflatten(0, (bsz, N))
		flow = flow + fl
	return flow

def get_region_smoothness_and_variation(flow, region):
	B, C, num_region, H, W = region.shape
	region_smoothness = torch.zeros(B, num_region, device=region.device)
	smoothness_map = torch.zeros(B, 1, 1, H, W, device=region.device)
	region_variation = torch.zeros(B, num_region, device=region.device)
	variation_map = torch.zeros(B, 1, 1, H, W, device=region.device)
	
	for b in range(B):
		delta_x = torch.abs(flow[b,:,:,:,:-1] - flow[b,:,:,:,1:])
		delta_y = torch.abs(flow[b,:,:,:-1,:] - flow[b,:,:,1:,:])
		for r in range(num_region):
			region_mask = region[b, 0, r, : ,:] == 1
			flow_in_region = flow[b, 0, :, region_mask]
			if region_mask.sum() > 1:
				smoothness_x = delta_x[:, :, region_mask[:,:-1]].mean()
				smoothness_y = delta_y[:, :, region_mask[:-1,:]].mean()
				total_smoothness = smoothness_x + smoothness_y
				region_smoothness[b,r] = total_smoothness
				smoothness_map[b,:,:,region_mask] = total_smoothness
				
				variation = flow_in_region.std(dim=1).mean()
				region_variation[b,r] = variation
				variation_map[b,:,:,region_mask] = variation

				smoothness_map = 1 - smoothness_map/smoothness_map.max()
				variation_map = 1 - variation_map/variation_map.max()
	
	return region_smoothness, smoothness_map ,region_variation, variation_map


# def gso_convert_to_ply(self):
# 	path = f'{self.root_path}/megapose-gso/model_ply'
# 	if not os.path.isdir(path): 
# 		print('convert large obj to small ply...')
# 		os.makedirs(path, exist_ok=True)
# 		for obj in tqdm(self.gso_obj_list):
# 			o = f'{self.root_path}/megapose-gso/model/{obj["gso_id"]}/meshes/model.obj'
# 			t = f'{self.root_path}/megapose-gso/model/{obj["gso_id"]}/meshes/texture.png'
# 			mesh = load_objs_as_meshes([o], device='cuda')
# 			textures_uv_rgb = plt.imread(t)
# 			textures_uv_rgb = torch.tensor(textures_uv_rgb).unsqueeze(0).to(mesh.device)
# 			mesh.textures._maps_padded = textures_uv_rgb
# 			mesh.textures = convert_to_textureVertex(mesh.textures, mesh)
# 			mesh = normalize_mesh(mesh, scale=0.1)
# 			IO().save_mesh(mesh, f'{path}/{obj["obj_id"]}.ply')
