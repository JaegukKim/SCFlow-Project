import numpy as np
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
import psutil
import time
import orjson

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

class OcclusionAugmentation(object):
	def __init__(self, p=0.5):
		self.p = p
		self.affine = transforms.RandomAffine((0, 180), (0.3, 0.3), (0.5, 0.8))
	def __call__(self, image, visib_mask, obj_id):
		bsz, n_view, c, h, w = image.shape
		mask_h, mask_w = visib_mask.shape[-2:]
		obj_id = obj_id if isinstance(obj_id, list) else obj_id.tolist()
		v_mask = F.interpolate(visib_mask.flatten(0, 1), [h, w], mode='nearest').unflatten(0, (bsz, n_view))
		occlusion = torch.cat([image * v_mask, v_mask], dim=2)
		occlusion = self.affine(occlusion.flatten(0, 1)).unflatten(0, (bsz, n_view))
		occlusion_rgb, occlusion_mask = occlusion[:, :, :3], occlusion[:, :, 3:]
		inv_occlusion_mask = 1 - occlusion_mask
		for i, id in enumerate(obj_id):
			valid_idx = list(np.where(np.array(obj_id) != id)[0])
			if len(valid_idx) > 0 and self.p > np.random.uniform():
				occ_index = random.sample(valid_idx, 1)[0]
				image[i] = image[i] * inv_occlusion_mask[occ_index] + occlusion_rgb[occ_index] * occlusion_mask[occ_index]
				new_visib_mask = (v_mask[i] * inv_occlusion_mask[occ_index])
				visib_mask[i] = F.interpolate(new_visib_mask, [mask_h, mask_w], mode='nearest')
		return image, visib_mask

class ImageAugmentation(object):
	def __init__(self):
		color_aug =[transforms.ColorJitter([0.95, 1.05], [0.95, 1.05], [0.95, 1.05], [-0.05, 0.05]), 
					transforms.GaussianBlur([5, 5], (0.05, 0.1))]
		self.applier = transforms.RandomApply(transforms=color_aug, p=0.5)
	def __call__(self, image):
		bsz, n_view, c, h, w = image.shape
		image = self.applier(image.flatten(0, 1)).unflatten(0, (bsz, n_view))
		return image

def load_json_fast(src):
	if type(src) is str:
		with open(src, 'r') as f:
			content = orjson.loads(f.read())
	elif type(src) is bytes:
		content = orjson.loads(src)
	if type(content) is dict:
		content = {int(k) if k.lstrip('-').isdigit() else k: v for k, v in content.items()}
	return content
