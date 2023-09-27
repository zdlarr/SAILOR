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


class Vgg19(torch.nn.Module):
	def __init__(self, requires_grad=False, pretrained=True):
		super(Vgg19, self).__init__()
		self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
		for param in self.parameters():
			param.requires_grad = requires_grad

	def forward(self, X, indices=None):
		if indices is None:
			indices = [2, 7, 12, 21, 30]
   
		out = []
		for i in range(indices[-1]):
			X = self.vgg_pretrained_features[i](X)
			if (i+1) in indices:
				out.append(X)
		
		return out


class VGGLoss(nn.Module):
	def __init__(self, device='cuda:0', vgg=None, weights=None, indices=None, normalize=True):
		super(VGGLoss, self).__init__()
		
		if vgg is None:
			self.vgg = Vgg19().to(device)
		else:
			self.vgg = vgg.to(device)

		self.criterion = nn.L1Loss()
		self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
		# default conv layers: 'conv_1_2, conv_2_2, conv_3_2, conv_4_2, conv_5_2.'
		self.indices = indices or [2, 7, 12, 21, 30]
		self._normalize = normalize
		self._mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)
		self._std  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32) # [3]
  
	def normalize(self, x):
		# x: [B, 3, H, W]
		return (x - self._mean[None, :, None, None]) / self._std[None, :, None, None]

	def __call__(self, x, y):
		if self._normalize:
			x = self.normalize(x)
			y = self.normalize(y)
   
		x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
		loss = 0
		for i in range(len(x_vgg)):
			loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

		return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        # 3 layers.
        blocks.append(vgg_pretrained_features[:4].eval())
        blocks.append(vgg_pretrained_features[4:9].eval())
        blocks.append(vgg_pretrained_features[9:16].eval())
        # blocks.append(vgg_pretrained_features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # print(self.blocks, 1)
        # print(vgg_pretrained_features)
        # for i, block in enumerate(self.blocks):
        #     if i in [0, 1, 2, 3]:
        #         print(i)
        #         print(block)


    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        # resize to control the total image's resolution;
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss