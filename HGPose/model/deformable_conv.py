import torch
import torchvision.ops
from torch import nn


class DeformableConv2d(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channels,
				 kernel_size=3,
				 stride=1,
				 padding=1,
				 dilation=1,
				 bias=False):
		super(DeformableConv2d, self).__init__()

		assert type(kernel_size) == tuple or type(kernel_size) == int

		kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
		self.stride = stride if type(stride) == tuple else (stride, stride)
		self.padding = padding
		self.dilation = dilation

		self.offset_conv = nn.Conv2d(in_channels,
									 2 * kernel_size[0] * kernel_size[1],
									 kernel_size=kernel_size,
									 stride=stride,
									 padding=self.padding,
									 dilation=self.dilation,
									 bias=True)

		nn.init.constant_(self.offset_conv.weight, 0.)
		nn.init.constant_(self.offset_conv.bias, 0.)

		self.modulator_conv = nn.Conv2d(in_channels,
										1 * kernel_size[0] * kernel_size[1],
										kernel_size=kernel_size,
										stride=stride,
										padding=self.padding,
										dilation=self.dilation,
										bias=True)

		nn.init.constant_(self.modulator_conv.weight, 0.)
		nn.init.constant_(self.modulator_conv.bias, 0.)

		self.regular_conv = nn.Conv2d(in_channels=in_channels,
									  out_channels=out_channels,
									  kernel_size=kernel_size,
									  stride=stride,
									  padding=self.padding,
									  dilation=self.dilation,
									  bias=bias)

	def forward(self, x):
		# h, w = x.shape[2:]
		# max_offset = max(h, w)/4.

		offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
		modulator = 2. * torch.sigmoid(self.modulator_conv(x))
		# op = (n - (k * d - 1) + 2p / s)
		x = torchvision.ops.deform_conv2d(input=x,
										  offset=offset,
										  weight=self.regular_conv.weight,
										  bias=self.regular_conv.bias,
										  padding=self.padding,
										  mask=modulator,
										  stride=self.stride,
										  dilation=self.dilation)
		return x
	






class DeformableConv2dGuided(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channels,
				 N_ref, 
				 kernel_size=3,
				 stride=1,
				 padding=1,
				 dilation=1,
				 bias=False):
		super(DeformableConv2dGuided, self).__init__()
		
		self.N_ref = N_ref

		assert type(kernel_size) == tuple or type(kernel_size) == int

		kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
		self.stride = stride if type(stride) == tuple else (stride, stride)
		self.padding = padding
		self.dilation = dilation

		self.offset_conv = nn.Conv2d(in_channels * self.N_ref,
									 2 * self.N_ref,
									 kernel_size=kernel_size,
									 stride=stride,
									 padding=self.padding,
									 dilation=self.dilation,
									 bias=True)

		nn.init.constant_(self.offset_conv.weight, 0.)
		nn.init.constant_(self.offset_conv.bias, 0.)

		self.modulator_conv = nn.Conv2d(in_channels * self.N_ref,
										1 * self.N_ref,
										kernel_size=kernel_size,
										stride=stride,
										padding=self.padding,
										dilation=self.dilation,
										bias=True)

		nn.init.constant_(self.modulator_conv.weight, 0.)
		nn.init.constant_(self.modulator_conv.bias, 0.)

		self.regular_conv = nn.Conv2d(in_channels=in_channels,
									  out_channels=out_channels,
									  kernel_size=1,
									  stride=1,
									  padding=0,
									  dilation=1,
									  bias=bias)

	def forward(self, x, offset=None):
		# h, w = x.shape[2:]
		# max_offset = max(h, w)/4.

		channel_concat_x = x.flatten(1, 2)
		if offset == None:
			offset = -1.0 * self.offset_conv(channel_concat_x) #.clamp(-max_offset, max_offset)
		modulator = 2. * torch.sigmoid(self.modulator_conv(channel_concat_x))
		# op = (n - (k * d - 1) + 2p / s)

		each_ref_x = [ref_x.squeeze(1) for ref_x in x.split(1, dim=1)]
		each_ref_offset = offset.split(2, dim=1)
		each_ref_modulator = modulator.split(1, dim=1)

		all_ref = []
		all_off = []
		for ref_x, ref_offset, ref_modulator in zip(each_ref_x, each_ref_offset, each_ref_modulator):
			result = torchvision.ops.deform_conv2d(input=ref_x,
												  offset=ref_offset,
												  weight=self.regular_conv.weight,
												  bias=self.regular_conv.bias,
												  padding=0,
												  mask=ref_modulator,
												  stride=1,
												  dilation=1)
			all_ref.append(result)
			all_off.append(ref_offset)
		all_ref = torch.mean(torch.stack(all_ref, dim=1), dim=1)
		all_off = torch.stack(all_off, dim=1)
		return all_ref, offset
	
