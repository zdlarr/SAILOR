'''
    Basic Network framework.
'''

import sys, os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from models.networks import init_net, set_requires_grad, print_net

class BasicNet(nn.Module):
    """
        Basic Network Frames.
    """

    def __init__(self, opts, device):
        super(BasicNet, self).__init__()
        self.opts = opts
        # basic params.
        self.device    = device
        self.gpu_ids   = opts.gpu_ids
        self.model_dir = opts.model_dir
        self.is_train  = opts.is_train
        self.is_val    = opts.is_val

        # saving params' names
        self.loss_names   = []
        self.visual_names = []
        # networks & optimizers.
        self.nets         = []
        self.optimizers   = []

    def setup(self, nets):
        if not isinstance(nets, list):
            nets = [nets]

        for i, net in enumerate(nets):
            nets[i] = init_net(net, self.device, init_params=self.opts.default_init_params, init_type=self.opts.net_init_type, init_gain=self.opts.net_init_gain, gpu_ids=self.gpu_ids)
            if (self.is_train and self.opts.continue_train) or (not self.is_train):
                # loading epoch of the network.
                self.load_network(nets[i], self.opts.epoch, type(net).__name__)

        return nets

    def load_network(self, net, epoch, name):
        # loading network's parameters.
        model_name = '%s_model_%s.pth' % (epoch, name)
        load_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(load_path):
            print('Not loading model from %s' % load_path)
            return
        
        state_dict = torch.load(load_path, map_location=torch.device('cpu')) # map to specific gpu.
        if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.DistributedDataParallel):
            net = net.module # update module.
        
        geo_model_dict = net.state_dict()
        new_state_dict = {k: v for k,v in state_dict.items() if k in geo_model_dict.keys()}
        geo_model_dict.update(new_state_dict)
        net.load_state_dict(geo_model_dict, strict=False) # strict is set to False here.
        print('Load the model from %s' % load_path)
        return

    def save(self, nets, epoch):
        if not isinstance(nets, list):
            nets = [nets]
        
        for net in nets:
            if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.DistributedDataParallel):
                self.save_network(net, epoch, type(net.module).__name__)
            else:
                self.save_network(net, epoch, type(net).__name__)

    def save_network(self, net, epoch, name):
        model_name = '%s_model_%s.pth' % (epoch, name)
        save_path = os.path.join(self.opts.model_dir, model_name)
        if isinstance(net, torch.nn.DataParallel):
            torch.save(net.module.cpu().state_dict(), save_path, _use_new_zipfile_serialization=False)
        elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
            if dist.get_rank() == 0: # only save the network in process 0.
                torch.save(net.module.state_dict(), save_path, _use_new_zipfile_serialization=False)
        else:
            torch.save(net.cpu().state_dict(), save_path, _use_new_zipfile_serialization=False)
        net.to('cuda:' + str(dist.get_rank())) # to original device.

    @torch.no_grad()
    def get_current_visuals(self):
        # get the current visuals results.
        visual_result = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_result[name] = getattr(self, name)
        
        return visual_result

    def get_current_losses(self):
        # get the current loss.
        loss_result = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                loss_result[name] = getattr(self, 'loss_' + name)
        
        return loss_result
    
    def get_total_losses(self):
        loss_items = self.get_current_losses()
        return sum(loss_items.values())
    
    def set_require_grads(self, net, require_grads=True):
        set_requires_grad(net, require_grads)
        return

    def print_networks(self, net, verbose):
        print_net(net, type(net).__name__, verbose)
        
    def update_learning_rate(self, optimizers, lr):
        if not isinstance(optimizers):
            optimizers = [optimizers]

        for opti in optimizers:
            for p_gp in opti.param_groups:
                p_gp['lr'] = lr
            
        return lr

    """
        Basic train & eval & inference interfaces.                
    """
    def _set_train(self, nets):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            net.train()

    def _set_eval(self, nets):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            net.eval()

    def inference(self):
        with torch.no_grad():
            self.forward()

    def forward(self):
        # feed the network with the needed data, (set input).
        pass

    def backward(self):
        # backward & caculate the gradients for parameters.
        # only when is_train is valid.
        pass
