import torch
import os
from collections import OrderedDict
class ModuleDataParallel(torch.nn.DataParallel):
	"""This class extends nn.DataParallel to access custom attributes of the module being wrapped
	(by default DataParallel does not allow accessing members after wrapping).
	Read more: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
	"""
	def __getattr__(self, name):
		try:
			return super().__getattr__(name)
		except AttributeError:
			return getattr(self.module, name)


def load_model(load_path, model, optim=None, lr_scheduler=None):
	print(f"Loading model from [{load_path}]")
	load_dict = torch.load(load_path, map_location=torch.device('cuda'))
	new_dict = OrderedDict()
	
	for k, v in load_dict['model_state_dict'].items():
		if k.split('.')[0] == 'module':
			new_dict[k[7:]] = v
		else:
			new_dict[k] = v
	model.load_state_dict(new_dict)

	if optim:
		optim.load_state_dict(load_dict['optimizer_state_dict'])
		
	if lr_scheduler:
		lr_scheduler.load_state_dict(load_dict['lr_scheduler_state_dict'])

	steps = load_dict['steps'] if 'steps' in load_dict else None
	best_AR = load_dict['best_AR'] if 'best_AR' in load_dict else None 
	return steps, best_AR

def freeze_batch_norm(model):
	for m in model.modules():
		if isinstance(m, torch.nn.BatchNorm2d):
			m.eval()