from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def one_hot_affinity(affinity, dim=2):
	N_ref = affinity.shape[dim]
	idx = torch.argmax(affinity, dim=dim)
	one_hot = F.one_hot(idx, num_classes=N_ref).to(torch.float)
	return one_hot

def normalize_feature(f, axes=[-3, -4]):
	mean = torch.mean(f, dim=axes, keepdim=True)
	std = torch.std(f, dim=axes, keepdim=True)
	f = (f - mean) / (std + 1e-8)
	return f

def get_dual_affinity(K, Q, temperature=1, k=-1):
	B, TQ, qk_dim, H, W = Q.shape
	B, TK, qk_dim, H, W = K.shape
	# affinity = torch.einsum('brchw,bqcij->bqijrhw', K, Q).view(B, TQ*H*W, TK*H*W)		# B, TQ HW(N of que), TK HW(N of ref)
	# affinity = affinity / math.sqrt(qk_dim)
	# affinity = affinity / (Q.norm(dim=2).view(B, TQ*H*W, 1) * K.norm(dim=2).view(B, 1, TK*H*W))	# cosine similarity
	Q_vec = Q.permute(0, 2, 1, 3, 4).view(B, qk_dim, TQ*H*W, 1)
	K_vec = K.permute(0, 2, 1, 3, 4).view(B, qk_dim, 1, TK*H*W)
	Q_mask = (Q_vec.sum(dim=1) > -3)
	K_mask = (K_vec.sum(dim=1) > -3)
	affinity_mask = Q_mask * K_mask
	affinity = -torch.norm(Q_vec - K_vec, dim=1)										# B, TQ HW(N of que), TK HW(N of ref)
	affinity = affinity / temperature
	if k > 0:
		tk_val, _ = torch.topk(affinity, dim=-1, k=k)
		tk_val_min, _ = torch.min(tk_val, dim=-1,keepdim=True)
		affinity_mask = affinity_mask * (affinity >= tk_val_min)
	affinity = affinity.masked_fill(affinity_mask==0, -1e9)
	affinity = F.softmax(affinity, dim=1) * affinity 
	dual_affinity = F.softmax(affinity, dim=2)
	dual_affinity = dual_affinity * affinity_mask.to(dual_affinity.dtype)
	return dual_affinity


def get_affinity(K, Q, temperature=1, k=-1):
	B, TQ, qk_dim, H, W = Q.shape
	B, TK, qk_dim, H, W = K.shape
	# affinity = torch.einsum('brchw,bqcij->bqijrhw', K, Q).view(B, TQ*H*W, TK*H*W)		# B, TQ HW(N of que), TK HW(N of ref)
	# affinity = affinity / math.sqrt(qk_dim)
	# affinity = affinity / (Q.norm(dim=2).view(B, TQ*H*W, 1) * K.norm(dim=2).view(B, 1, TK*H*W))	# cosine similarity
	Q_vec = Q.permute(0, 2, 1, 3, 4).view(B, qk_dim, TQ*H*W, 1)
	K_vec = K.permute(0, 2, 1, 3, 4).view(B, qk_dim, 1, TK*H*W)
	Q_mask = (Q_vec.sum(dim=1) > -3)
	K_mask = (K_vec.sum(dim=1) > -3)
	affinity_mask = Q_mask * K_mask
	affinity = -torch.norm(Q_vec - K_vec, dim=1)										# B, TQ HW(N of que), TK HW(N of ref)
	affinity = affinity / temperature
	if k > 0:
		tk_val, _ = torch.topk(affinity, dim=-1, k=k)
		tk_val_min, _ = torch.min(tk_val, dim=-1,keepdim=True)
		affinity_mask = affinity_mask * (affinity >= tk_val_min)
	affinity = affinity.masked_fill(affinity_mask==0, -1e9)
	affinity = F.softmax(affinity, dim=2) 
	affinity = affinity * affinity_mask.to(affinity.dtype)
	return affinity


def find_topk(affinity, k, HW):
	B, TQ_HW, TK_HW = affinity.shape 
	T, H, W = TK_HW // HW, int(math.sqrt(HW)), int(math.sqrt(HW))
	_, topk_idx = torch.topk(affinity, k=k, dim=2, sorted=True)							# B, HW, k
	valid = torch.zeros_like(affinity).to(topk_idx.device)								# B, HW, THW
	valid.scatter_(2, topk_idx, 1.)														# B, HW, THW 
	valid = valid.view(B, H, W, T, H, W)												# B, H, W, T, H, W
	idx = torch.nonzero(valid)															# B*H*W*k, 6
	return idx

def get_affinity_topk(ref_k, que_k, idx, scale, temperature=1):
	B, T, k_dim, H, W = ref_k.shape
	Bhwk, _ = idx.shape
	s = scale
	h, w = H // s, W // s
	k = Bhwk // (B * h * w)
	ref_k = ref_k.view(B, T, k_dim, h, s, w, s) 									# B, T, k_dim, h, s, w, s
	ref_k = ref_k.permute(0, 1, 3, 5, 4, 6, 2)										# B, T, h, w, s, s, k_dim
	ref_k = ref_k[idx[:, 0], idx[:, 3], idx[:, 4], idx[:, 5]]						# B*h*w*k, s, s, k_dim
	ref_k = ref_k.reshape(B, h, w, k, s, s, k_dim)									# B, h, w, k, s, s, k_dim
	ref_k = ref_k.permute(0, 3, 4, 5, 1, 2, 6)										# B, k, s, s, h, w, k_dim
	ref_k = ref_k.reshape(B, k*s*s, h*w, k_dim)										# B, k*s*s(N of ref per que with scale), hw(N of que), k_dim
	que_k = que_k.view(B, 1, k_dim, h, s, w, s)										# B, 1, k_dim, h, s, w, s
	que_k = que_k.permute(0, 1, 4, 6, 3, 5, 2)										# B, 1, s, s, h, w, k_dim
	que_k = que_k.reshape(B, 1*s*s, h*w, k_dim)										# B, 1*s*s(scale), hw(N of que), k_dim
	affinity = torch.einsum('brlc,bqlc->bqlr', ref_k, que_k)						# B, 1*s*s(que scale), hw(N of que), k*s*s(N of ref)
	affinity = affinity / math.sqrt(k_dim)											# B, 1*s*s(que scale), hw(N of que), k*s*s(N of ref)
	affinity = affinity / temperature
	affinity = F.softmax(affinity, dim=3)											# B, 1*s*s(que scale), hw(N of que), k*s*s(N of ref)
	# at this point, reshape affinity form
	affinity = affinity.view(B, 1, s, s, h, w, k*s*s)
	affinity = affinity.permute(0, 1, 4, 2, 5, 3, 6)								# B, 1, h, s, w, s, k*s*s
	affinity = affinity.reshape(B, H*W, k*s*s)										# B, HW(N of que), k*s*s(N of ref)
	return affinity

def attention(ref_v, affinity):
	B, TV, _, H, W = ref_v.shape
	TK = TV
	TQ = affinity.shape[-2] // (H * W)
	mask = (affinity.mean(dim=2).view(B, TQ, 1, H, W) > 0)
	affinity = affinity.view(B, TQ, H, W, TK, H, W) 
	que_v = torch.einsum('brchw,bqijrhw->bqcij', ref_v, affinity)	# B, v_dim, H, W
	que_v = ((que_v + 1) * mask) - 1
	return que_v													# B, TQ, v_dim, H, W

def attention_topk(ref_v, affinity, idx, scale):
	B, T, v_dim, H, W = ref_v.shape
	_, _, kss = affinity.shape
	s = scale
	h, w, k = H // s, W // s, kss // (s * s)
	ref_v = ref_v.view(B, T, v_dim, h, s, w, s) 										# B, T, v_dim, h, s, w, s
	ref_v = ref_v.permute(0, 1, 3, 5, 4, 6, 2)											# B, T, h, w, v_dim, s, s
	ref_v = ref_v[idx[:, 0], idx[:, 3], idx[:, 4], idx[:, 5]]							# B*k*hw, v_dim, s, s
	ref_v = ref_v.reshape(B, k, h, w, v_dim, s, s)										# B, k, h, w, v_dim, s, s	
	ref_v = ref_v.permute(0, 1, 5, 6, 2, 3, 4)											# B, k, s, s, h, w, v_dim
	ref_v = ref_v.reshape(B, k*s*s, h*w, v_dim)											# B, k*s*s(N of ref per que with scale), hw(N of que), v_dim
	# at this point, undo reshape affinity
	affinity = affinity.view(B, 1, h, s, w, s, k*s*s)									# B, 1, h, s, w, s, k*s*s
	affinity = affinity.permute(0, 1, 3, 5, 2, 4, 6)									# B, 1, s, s, h, w, k*s*s
	affinity = affinity.reshape(B, 1*s*s, h*w, k*s*s)									# B, 1*s*s(que scale), hw(N of que), k*s*s(N of ref)
	que_v = torch.einsum('brlc,bqlr->bqcl', ref_v, affinity)							# B, 1*s*s(que scale), v_dim, hw(N of que)
	que_v = que_v.view(B, 1, s, s, v_dim, h, w)
	que_v = que_v.permute(0, 1, 4, 5, 2, 6, 3)											# B, 1, v_dim, h, s, w, s
	que_v = que_v.reshape(B, 1, v_dim, H, W)											# B, 1, v_dim, H, W
	return que_v


class KeyValue(nn.Module):
	def __init__(self, indim, keydim, valdim, only_key=False):
		super(KeyValue, self).__init__()
		self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
		self.only_key = only_key
		if not self.only_key:
			self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
	def forward(self, x):
		B, n = x.shape[0:2]
		x = x.flatten(0, 1)
		K = self.Key(x).unflatten(0,(B, n))
		v = self.Value(x).unflatten(0,(B, n)) if not self.only_key else None
		return K, v

class QueryKeyValue(nn.Module):
	def __init__(self, indim, qk_dim, valdim, is_que=False, is_key=False, is_val=False):
		super(QueryKeyValue, self).__init__()
		self.is_que, self.is_key, self.is_val = is_que, is_key, is_val
		if self.is_que:
			self.Query = nn.Conv2d(indim, qk_dim, kernel_size=(3,3), padding=(1,1), stride=1)
		if self.is_key:
			self.Key = nn.Conv2d(indim, qk_dim, kernel_size=(3,3), padding=(1,1), stride=1)
		if self.is_val:
			self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
	def forward(self, x):
		B, n = x.shape[0:2]
		x = x.flatten(0, 1)
		Q, K, V = None, None, None
		if self.is_que:
			Q = self.Query(x).unflatten(0,(B, n))
		if self.is_key:
			K = self.Key(x).unflatten(0,(B, n))
		if self.is_val:
			V = self.Value(x).unflatten(0,(B, n))
		return Q, K, V
