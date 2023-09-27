
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision import models
import functools
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# No Norm Layers.
class NoneNorm(torch.nn.Module):
	def __init__(self, *args):
		super(NoneNorm, self).__init__()
	
	def forward(self, x):
		return x


def get_norm_layer(norm_type='instance'):
	"""Return a normalization layer

	Parameters:
		norm_type (str) -- the name of the normalization layer: batch | instance | group | none

	For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
	For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
	"""
	
	if norm_type == 'batch2d':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
	elif norm_type == 'batch1d':
		norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
	elif norm_type == 'group':
		norm_layer = functools.partial(nn.GroupNorm, affine=True) # Group Norm doesn't contain "track_running_stats".
	elif norm_type == 'none':
		norm_layer = NoneNorm
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

	return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		# using normal method to initialize the kernel's weights.
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		if classname.find(
				'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)
		
		# for swin.
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.DistributedDataParallel):
		if torch.distributed.get_rank() == 0:
			print('initialize network {} with {}'.format(type(net.module).__name__, init_type))
	else:
		print('initialize network {} with {}'.format(type(net).__name__, init_type))
	
	net.apply(init_func)

def init_net(net, device, init_params=True, init_type='normal', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 1:
		assert torch.cuda.is_available()
		# net = BalancedDataParallel(2, net, gpu_ids)
		# net = torch.nn.DataParallel(net, gpu_ids)
		# local_rank = int(os.environ["LOCAL_RANK"])
		# net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
		local_rank = torch.distributed.get_rank()
		net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
		# net.to(gpu_ids[0])
	if init_params: # if init the net using defaults methods.
		init_weights(net, init_type=init_type, init_gain=init_gain)
	return net


def print_net(net, name, verbose):
	num_params =  0
	for param in net.parameters():
		num_params += param.numel()
	
	if verbose:
		print(net)
		
	print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))


def set_requires_grad(nets, requires_grad=False):
	if not isinstance(nets, list):
		nets = [nets]
	
	for net in nets:
		if net is not None:
			for params in net.parameters():
				params.requires_grad = requires_grad


def update_lr(optimizers, lr):
	# update learning rate with: lr 
	if not isinstance(optimizers, list):
		optimizers = [optimizers]
	
	for opti in optimizers:
		for p_gp in opti.param_groups:
			p_gp['lr'] = lr
	
	return lr
