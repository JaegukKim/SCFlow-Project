import torch
import torch.nn.functional as F
from HGPose.utils.geometry import object_space_grid, apply_backward_flow
from torch import nn
import math

def grid_matching_loss(pred_RT, query_RT, query_K, img_size, pool_size=2, N_z=2):
	bsz, n_inp = pred_RT.shape[0:2]
	query_RT = query_RT.expand(bsz, n_inp, 4, 4).flatten(0, 1)
	query_K = query_K.expand(bsz, n_inp, 3, 3).flatten(0, 1)
	pred_RT = pred_RT.flatten(0, 1)
	query_grid = object_space_grid(query_K, query_RT, [img_size, img_size], [pool_size, pool_size], N_z)
	pred_grid = object_space_grid(query_K, pred_RT, [img_size, img_size], [pool_size, pool_size], N_z)
	query_dist = torch.norm(query_RT[..., :3, 3], 2, -1)
	pred_dist = torch.norm(pred_RT[..., :3, 3], 2, -1)
	grid_error = F.mse_loss(pred_grid, query_grid.detach(), reduce=False).unflatten(0, (bsz, n_inp)).sum(-1) + 1e-9
	grid_error = grid_error.permute(0, 2, 3, 4, 1).mean([1, 2, 3])
	loss = torch.sqrt(grid_error).mean()
	loss = loss + F.l1_loss(pred_dist, query_dist.detach())
	return loss

def confidence_loss(pred_confi, grid_error):
	nearest = F.one_hot(grid_error.argmin(-1), grid_error.shape[-1]).to(torch.float)
	loss = F.cross_entropy(pred_confi, nearest)
	return loss

def distilation_loss(pred_affinity, gt_affinity):
	loss = F.kl_div(torch.log(pred_affinity+1e-7), gt_affinity+1e-7)
	loss = loss.mean()
	return loss

def focal_loss(pred_affinity, gt_affinity, alpha=0.5, gamma=2.0, thr=0.05):
	p_aff = torch.clamp(pred_affinity, 1e-6, 1-1e-6)
	g_aff = (gt_affinity > thr)
	loss_pos = -alpha		* torch.pow(1 - p_aff[g_aff == 1], 	gamma) * (p_aff[g_aff == 1] 	+ 1e-6).log()
	loss_neg = -(1-alpha)	* torch.pow(p_aff[g_aff == 0], 		gamma) * (1 - p_aff[g_aff == 0] + 1e-6).log()
	if torch.isnan(loss_neg.mean()) or torch.isnan(loss_pos.mean()):
		print('stop')
	loss = loss_pos.mean() + loss_neg.mean()
	return loss


def sequence_loss(flow_preds, flow_gt, gamma=0.8, max_flow=400):
	"""Loss function defined over sequence of flow predictions"""
	# exclude invalid pixels and extremely large diplacements
	flow_norm = torch.sum(flow_gt**2, dim=-3, keepdim=True).sqrt()
	valid_flow_mask = (flow_norm < max_flow).to(flow_gt)
	valid_flow_mask = valid_flow_mask[None]
	flow_gt = flow_gt[None]
	flow_preds = torch.stack(flow_preds)  # shape = (num_flow_updates, batch_size, 2, H, W)
	abs_diff = (flow_preds - flow_gt).abs()
	abs_diff = (abs_diff * valid_flow_mask).mean(axis=(1, 2, 3, 4, 5))
	num_predictions = flow_preds.shape[0]
	weights = gamma ** torch.arange(num_predictions - 1, -1, -1).to(flow_gt.device)
	flow_loss = (abs_diff * weights).sum()
	return flow_loss


def sequence_loss2(loss, gamma=0.8):
	num_predictions = loss.shape[0]
	weights = gamma ** torch.arange(num_predictions - 1, -1, -1).to(loss.device)
	sequence_loss = (loss * weights).sum()
	return sequence_loss

def flow_loss(flow_preds, flow_gt, max_flow=400):
	"""Loss function defined over sequence of flow predictions"""
	# exclude invalid pixels and extremely large diplacements
	flow_norm = torch.sum(flow_gt**2, dim=-3, keepdim=True).sqrt()
	valid_flow_mask = (flow_norm < max_flow).to(flow_gt)
	valid_flow_mask = valid_flow_mask[None]
	flow_gt = flow_gt[None]
	flow_preds = torch.stack(flow_preds)  # shape = (num_flow_updates, batch_size, 2, H, W)
	abs_diff = (flow_preds - flow_gt).abs()
	flow_loss = (abs_diff * valid_flow_mask).mean(axis=(1, 2, 3, 4, 5))
	return flow_loss

def flow_loss_with_mask(flow_preds, flow_gt, valid=None, max_flow=400, eps=1e-10):
	"""Loss function defined over sequence of flow predictions"""
	bsz = len(flow_gt)
	# exclude invalid pixels and extremely large diplacements
	flow_norm = torch.sum(flow_gt**2, dim=-3, keepdim=True).sqrt()
	if valid is None:
		valid = (flow_norm < max_flow).to(flow_gt)
	else:
		valid = ((valid >= 0.5) & (flow_norm < max_flow)).to(flow_gt)
	
	# eroded_valid = morphological_operation(valid.squeeze(1), 'erosion', 5).unsqueeze(1)
	# eroded_valid = eroded_valid[None]



	valid = valid[None]
	flow_gt = flow_gt[None]
	flow_preds = torch.stack(flow_preds)
	loss = (flow_preds - flow_gt).abs()
	loss1 = (valid * loss).sum(axis=(1,2,3,4,5)) / (valid.sum(axis=(1,2,3,4,5)) + eps)

	# loss2 = (eroded_valid * loss).sum(axis=(1,2,3,4,5)) / (eroded_valid.sum(axis=(1,2,3,4,5)) + eps)

	return loss1

def flow_NLLMixtureLaplace_loss_with_mask(flow_preds, uncert_preds, flow_gt, valid=None, max_flow=400, eps=1e-10):
	"""Loss function defined over sequence of flow predictions"""
	bsz = len(flow_gt)
	# exclude invalid pixels and extremely large diplacements
	flow_norm = torch.sum(flow_gt**2, dim=-3, keepdim=True).sqrt()
	if valid is None:
		valid = (flow_norm < max_flow).to(flow_gt)
	else:
		valid = ((valid >= 0.5) & (flow_norm < max_flow)).to(flow_gt)
	
	# eroded_valid = morphological_operation(valid.squeeze(1), 'erosion', 5).unsqueeze(1)
	# eroded_valid = eroded_valid[None]



	valid = valid[None]
	flow_gt = flow_gt[None]
	flow_preds = torch.stack(flow_preds)
	uncert_preds = torch.stack(uncert_preds)
	log_var_preds = uncert_preds[:,:,:,:2,:,:]
	weight_preds = uncert_preds[:,:,:,2:,:,:]

	l1 = torch.logsumexp(weight_preds, -3, keepdim=True)
	reg = math.sqrt(2) * torch.sum(torch.abs(flow_gt - flow_preds), -3, keepdim=True)
	exponent = weight_preds - math.log(2) - log_var_preds - reg*torch.exp(-0.5*log_var_preds)
	l2 = torch.logsumexp(exponent, -3, keepdim=True)
	loss = l1 - l2
	loss1 = (valid * loss).sum(axis=(1,2,3,4,5)) / (valid.sum(axis=(1,2,3,4,5)) + eps)
	#loss1 = loss1 / bsz

	return loss1

def coord_NLLMixtureLaplace_loss_with_mask(coord_preds, uncert_preds, coord_gt, valid, eps=1e-10):
	"""Loss function defined over sequence of flow predictions"""
	bsz = len(coord_gt)

	valid = valid >= 0.5
	
	log_var_preds = uncert_preds[:,:,:2,:,:]
	weight_preds = uncert_preds[:,:,2:,:,:]

	l1 = torch.logsumexp(weight_preds, -3, keepdim=True)
	reg = math.sqrt(2) * torch.sum(torch.abs(coord_gt - coord_preds), -3, keepdim=True)
	exponent = weight_preds - math.log(2) - log_var_preds - reg*torch.exp(-0.5*log_var_preds)
	l2 = torch.logsumexp(exponent, -3, keepdim=True)
	loss = l1 - l2
	loss1 = (valid * loss).sum() / (valid.sum() + eps)


	return loss1

def laplace_pdf(x, mu, var): 
	return 1 / (2*var) * torch.exp(-torch.sum(torch.abs(x-mu), -3, keepdim=True) * torch.sqrt(2/var))

def flow_NLLMixtureLaplace_loss_with_mask2(flow_preds, uncert_preds, flow_gt, valid=None, max_flow=400, eps=1e-10):
	"""Loss function defined over sequence of flow predictions"""
	bsz = len(flow_gt)
	# exclude invalid pixels and extremely large diplacements
	flow_norm = torch.sum(flow_gt**2, dim=-3, keepdim=True).sqrt()
	if valid is None:
		valid = (flow_norm < max_flow).to(flow_gt)
	else:
		valid = ((valid >= 0.5) & (flow_norm < max_flow)).to(flow_gt)
	
	# eroded_valid = morphological_operation(valid.squeeze(1), 'erosion', 5).unsqueeze(1)
	# eroded_valid = eroded_valid[None]



	valid = valid[None]
	flow_gt = flow_gt[None]
	flow_preds = torch.stack(flow_preds)
	uncert_preds = torch.stack(uncert_preds)
	#log_var_preds = uncert_preds[:,:,:,:2,:,:]
	var_preds = uncert_preds[:,:,:,:2,:,:]
	weight_preds = uncert_preds[:,:,:,2:,:,:]

	#var_preds = torch.exp(log_var_preds)
	laplace1 = laplace_pdf(flow_gt, flow_preds, var_preds[:,:,:,0:1,:,:])
	laplace2 = laplace_pdf(flow_gt, flow_preds, var_preds[:,:,:,1:2,:,:])
	weighted_laplace1 = weight_preds[:,:,:,0:1,:,:] * laplace1
	weighted_laplace2 = weight_preds[:,:,:,1:2,:,:] * laplace2
	loss  = -torch.log(weighted_laplace1 + weighted_laplace2 + eps)
	loss1 = (valid * loss).sum(axis=(1,2,3,4,5)) / (valid.sum(axis=(1,2,3,4,5)) + eps)
	#loss1 = loss1 / bsz

	return loss1

def coord_NLLMixtureLaplace_loss_with_mask2(coord_preds, uncert_preds, coord_gt, valid, eps=1e-10):
	"""Loss function defined over sequence of flow predictions"""
	bsz = len(coord_gt)

	valid = valid >= 0.5
	
	#log_var_preds = uncert_preds[:,:,:2,:,:]
	var_preds = uncert_preds[:,:,:2,:,:]
	weight_preds = uncert_preds[:,:,2:,:,:]

	#var_preds = torch.exp(log_var_preds)
	laplace1 = laplace_pdf(coord_gt, coord_preds, var_preds[:,:,0:1,:,:])
	laplace2 = laplace_pdf(coord_gt, coord_preds, var_preds[:,:,1:2,:,:])
	weighted_laplace1 = weight_preds[:,:,0:1,:,:] * laplace1
	weighted_laplace2 = weight_preds[:,:,1:2,:,:] * laplace2
	loss  = -torch.log(weighted_laplace1 + weighted_laplace2 + eps)
	loss1 = (valid * loss).sum() / (valid.sum() + eps)

	return loss1

def flow_Laplace_loss_with_mask2(flow_preds, uncert_preds, flow_gt, valid=None, max_flow=400, eps=1e-10):
	"""Loss function defined over sequence of flow predictions"""
	bsz = len(flow_gt)
	# exclude invalid pixels and extremely large diplacements
	flow_norm = torch.sum(flow_gt**2, dim=-3, keepdim=True).sqrt()
	if valid is None:
		valid = (flow_norm < max_flow).to(flow_gt)
	else:
		valid = ((valid >= 0.5) & (flow_norm < max_flow)).to(flow_gt)
	
	# eroded_valid = morphological_operation(valid.squeeze(1), 'erosion', 5).unsqueeze(1)
	# eroded_valid = eroded_valid[None]



	valid = valid[None]
	flow_gt = flow_gt[None]
	flow_preds = torch.stack(flow_preds)
	uncert_preds = torch.stack(uncert_preds)
	#log_var_preds = uncert_preds[:,:,:,:2,:,:]
	var_preds = uncert_preds

	#var_preds = torch.exp(log_var_preds)
	laplace = laplace_pdf(flow_gt, flow_preds, var_preds)
	loss  = -torch.log(laplace + eps)
	loss1 = (valid * loss).sum(axis=(1,2,3,4,5)) / (valid.sum(axis=(1,2,3,4,5)) + eps)
	#loss1 = loss1 / bsz

	return loss1

def coord_Laplace_loss_with_mask2(coord_preds, uncert_preds, coord_gt, valid, eps=1e-10):
	"""Loss function defined over sequence of flow predictions"""
	bsz = len(coord_gt)

	valid = valid >= 0.5
	
	#log_var_preds = uncert_preds[:,:,:2,:,:]
	var_preds = uncert_preds

	#var_preds = torch.exp(log_var_preds)
	laplace = laplace_pdf(coord_gt, coord_preds, var_preds)
	loss  = -torch.log(laplace + eps)
	loss1 = (valid * loss).sum() / (valid.sum() + eps)

	return loss1

def flow_NLLGaussian_loss_with_mask(flow_preds, uncert_preds, flow_gt, valid=None, max_flow=400, eps=1e-10):
	"""Loss function defined over sequence of flow predictions"""
	bsz = len(flow_gt)
	# exclude invalid pixels and extremely large diplacements
	flow_norm = torch.sum(flow_gt**2, dim=-3, keepdim=True).sqrt()
	if valid is None:
		valid = (flow_norm < max_flow).to(flow_gt)
	else:
		valid = ((valid >= 0.5) & (flow_norm < max_flow)).to(flow_gt)
	
	# eroded_valid = morphological_operation(valid.squeeze(1), 'erosion', 5).unsqueeze(1)
	# eroded_valid = eroded_valid[None]



	valid = valid[None]
	flow_gt = flow_gt[None]
	flow_preds = torch.stack(flow_preds)
	uncert_preds = torch.stack(uncert_preds)
	log_var_preds = uncert_preds[:,:,:,:2,:,:]
	var_preds = torch.exp(log_var_preds)

	loss_fn = nn.GaussianNLLLoss(reduction='none')
	losses = loss_fn(flow_preds, flow_gt, var_preds)
	masked_losses = (valid * losses).sum(axis=(1,2,3,4,5)) / (valid.sum(axis=(1,2,3,4,5)) + eps)


	return masked_losses

def coord_NLLGaussian_loss_with_mask(coord_preds, uncert_preds, coord_gt, valid, eps=1e-10):
	"""Loss function defined over sequence of flow predictions"""
	bsz = len(coord_gt)

	valid = valid >= 0.5
	
	#log_var_preds = uncert_preds[:,:,:2,:,:]
	log_var_preds = uncert_preds
	var_preds = torch.exp(log_var_preds)

	loss_fn = nn.GaussianNLLLoss(reduction='none')
	loss = loss_fn(coord_preds, coord_gt, var_preds)
	masked_loss = (valid*loss).sum() / (valid.sum() + eps)

	

	return masked_loss

def morphological_operation(mask, operation='erosion', kernel_size=3):
	padding = kernel_size//2
	if operation == "erosion":
		return -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
	elif operation == 'dilation':
		return F.max_pool2d(mask, kernel_size, stride=1, padding=padding)



def geometry_loss(pred_que_geo, que_geo_gt, mask_visib_que):
	loss = F.l1_loss(pred_que_geo, que_geo_gt.detach(), reduce=False)
	loss = loss * mask_visib_que
	loss = loss.mean()
	return loss

def recon_geo_loss(flow_pred, flow_gt, que_geo, pred_geo, mask_visib_que, mode=['G_rec', 'f_rec']):
	masked_que_geo = (que_geo + 1) * mask_visib_que - 1
	recon_gt = apply_backward_flow(flow_gt, masked_que_geo, -1.0)
	all_l, flow_l, geo_l = 0, 0, 0
	if 'G_rec' in mode:
		recon_geo = apply_backward_flow(flow_gt, pred_geo, -1.0)
		geo_l = F.l1_loss(recon_geo, recon_gt.detach(), reduce=False).mean()
	loss = flow_l + geo_l + all_l
	return loss

def mask_l1_loss(pred_que_mask, que_mask):
	loss = F.l1_loss(pred_que_mask, que_mask.detach(), reduce=False)
	loss = loss.mean()
	return loss


def DisentanglePointMatchingLoss(pred_RT, query_RT, points_list):
	query_RT = query_RT.expand_as(pred_RT).flatten(0, 1)
	pred_RT = pred_RT.flatten(0, 1)

	pred_r = pred_RT[..., :3, :3]
	pred_t = pred_RT[..., :3, 3]
	gt_r = query_RT[..., :3, :3]
	gt_t = query_RT[..., :3, 3]

	loss = 0.
	batch_size = len(pred_r)
	scaled_pred_t, scaled_gt_t = torch.clone(pred_t), torch.clone(gt_t)
	
	for i in range(batch_size):
		points = torch.tensor(points_list[i]).to(gt_r[i])
		points_gt_rot = torch.matmul(gt_r[i], points.transpose(0, 1)).transpose(0, 1)
		points_gt_rt = points_gt_rot + scaled_gt_t[i][None]
		# rotation part, pred rotation, ground truth translation
		points_pred_rot = torch.matmul(pred_r[i], points.transpose(0, 1)).transpose(0, 1) + scaled_gt_t[i][None]
		loss_rotation_i = torch.mean(torch.linalg.norm(points_pred_rot - points_gt_rt, dim=-1, ord=1))
		# translation part
		points_pred_trans = points_gt_rot + scaled_pred_t[i][None]
		loss_trans_i = torch.mean(torch.linalg.norm(points_pred_trans - points_gt_rt, dim=-1, ord=1))
		loss_i = loss_trans_i + loss_rotation_i
		loss = loss + loss_i 
	loss = loss / batch_size
	return loss


def InfoNCE(coarse_pred, pos_idx):
	loss = F.cross_entropy(coarse_pred, pos_idx)
	return loss