
"""
    Training script for SRONet.
"""

import sys, os
import argparse
import warnings
import time
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import utils_render.util as util
# from SRONet.dataset.TrainRealDataset import RenderCombineTrainDataLoader
from SRONet.dataset.TrainDataset import RenTrainDataloader
from options.RenTrainOptions import RenTrainOptions
from SRONet.SRONet import SRONet
# from models.networks import update_lr
import torch.distributed as dist
import torch.multiprocessing as mp
from utils_render.utils_render import seed_everything, save_rgb_depths

warnings.filterwarnings('ignore')

def main_worker(local_rank, opts=None):
    # setup logging info.
    logging.basicConfig(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    
    if torch.cuda.is_available():
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = '1' # for tracking the errors.
        
        torch.backends.cudnn.enabled   = True
        torch.backends.cudnn.benchmark = True
        # for DDP module.
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method=opts.dist_url, world_size=len(opts.gpu_ids), rank=local_rank) # for DDP.
        device = torch.device('cuda', local_rank)
    else:
        deivce = torch.device('cpu')
        raise('Only support GPU training now...')
    
    #### 1. create the model dirs. ####
    opts.model_dir = os.path.join(opts.render_model_dir, opts.name)
    util.make_dir(opts.model_dir)
    opts.default_init_params = False
    opts.lr_render = 1e-4
    opts.num_epoch_render = 20
    opts.num_threads = 8
    opts.num_views   = 4

    #### 1.1 set sampling seed. ####
    seed_everything(opts.rand_seed)
        
    #### 2. create the dataset. ####
    train_dataset     = RenTrainDataloader(opts, phase='training')
    num_train_dataset = len(train_dataset)
    logging.info('Number of the training pairs = {} for each epoch'.format(num_train_dataset))
    #### validation dataset. ####
    opts_val = RenTrainOptions().parse()
    opts_val.batch_size       = 1
    opts_val.num_views        = opts.num_views
    opts_val.target_num_views = 1
    opts_val.num_sampled_rays = 1024 * 1
    opts_val.phase            = 'validating'
    opts_val.model_dir        = os.path.join(opts.render_model_dir, opts.name)
    
    validate_dataset = list(RenTrainDataloader(opts_val, phase='validating').get_iter())
    num_val_dataset  = len(validate_dataset)
    logging.info('Number of the validating pairs = {} for each epoch'.format(num_val_dataset))

    #### 3. create the models.
    model = SRONet(opts, device)
    model.setup_nets()
    model.set_train()
    model.set_val(False)

    # the model for evaluate.
    model_eval = SRONet(opts_val, device)
    model_eval.setup_nets() # to DDP model.
    model_eval.set_eval()
    model_eval.set_val(True)
    
    #### report the properties.
    num_total_params     = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Total params: {}; Trainable params: {}'.format(num_total_params, num_trainable_params))
    
    #### 4. training.
    start_epoch = 0 if not opts.continue_train else max(opts.resume_epoch, 0)
    lr = opts.lr_render
    
    with torch.autograd.set_detect_anomaly(True): # 
        for epoch in range(start_epoch, opts.num_epoch_render):
            train_dataset.dataloader.sampler.set_epoch(epoch)
            epoch_start_time = time.time()
            
            loss_rgbs  = util.AverageMeter()
            loss_depth = util.AverageMeter()
            loss_occ   = util.AverageMeter()
            loss_reg   = util.AverageMeter()
            loss_total = util.AverageMeter()
            
            for idx, item in enumerate(train_dataset):
                iter_start_time = time.time()                
                # feed data to model;
                model.set_input(item)
                model.backward() # forward & optimizing.
                
                # record training time.
                iter_train_time = time.time()
                eta = ((iter_train_time - epoch_start_time) / (idx + 1)) * num_train_dataset \
                    - (iter_train_time - epoch_start_time)
                
                # update all the loss terms.
                losses = model.get_current_losses()
                err_total = model.get_total_losses().item()
                
                err_rgb, err_depth, err_occ, err_reg = losses['color'].item(), losses['depth'].item(), losses['occ'].item(), losses['reg'].item()
                
                loss_rgbs.update(err_rgb)
                loss_depth.update(err_depth)
                loss_occ.update(err_occ)
                loss_reg.update(err_reg)
                
                if not model.get_total_losses().isnan():
                    loss_total.update(err_total)

                # print or save results.
                if idx % opts.freq_plot_render == 0:
                    info = 'Name: {0} | Epoch: {1}/{2} | {3}/{4} | Err(color): {5:.06f} | Err(depth): {6:.06f} | Err(occ): {7:.06f} | Err(reg): {8:.06f} | AvgErr(color): {9:.06f} | AvgErr(depth): {10:.06f} | AvgErr(occ): {11:.06f} | AvgErr(reg): {12:.06f} | AvgErr(total): {13:.06f} | LR: {14:.05f} | ETA: {15:02d}:{16:02d}'.format(
                        opts.name, epoch, opts.num_epoch_render, idx, num_train_dataset, 
                        err_rgb, err_depth, err_occ, err_reg,
                        loss_rgbs.avg, loss_depth.avg, loss_occ.avg, loss_reg.avg, loss_total.avg,
                        lr, int(eta // 60), int(eta - 60 * (eta // 60))
                    )
                    logging.info(info)

                # when meeting the freq, rendering the new results.
                # if idx % opts.freq_save_render == 0 and idx != 0: # saving the rendering properties (500).
                if idx % 200 == 0 and idx != 0: # saving the rendering properties (500).
                    val_data = validate_dataset[idx % num_val_dataset] # get the validation data.
                    # update the net's parameters. No Gradients here !!.
                    model_eval.update_parameters(model.render_net) # ! re-apply the params to eval model.
                    # predict the rgbs & depths.
                    model_eval.set_input(val_data)
                    model_eval.forward_build()
                    model_eval.forward_query()
                    
                    # get the predicted points & labels.
                    points = item['sampled_points'][0].transpose(0, 1).cpu() # [N, 3]
                    pred_occ = model._predicted_occ[0].cpu() # [1, N]
                    util.save_samples_truncted_prob(opts.model_dir, points, pred_occ)

                    # save the rendering results.
                    save_rgb_depths(opts_val, model_eval.get_current_visuals(), local_rank, val_data['name'], epoch, idx, phase='validating')
                
                if idx % opts.freq_save_render == 0 and idx != 0:
                    logging.info('Saving the latest model: (epoch: {}, iter: {})'.format(epoch, idx))
                    model.save_nets('latest')
                
            if epoch % opts.freq_save_epoch == 0:
                logging.info('Saving the model for epoch: {}'.format(epoch))
                model.save_nets('latest')
                model.save_nets(epoch)
                
            # update the optimizer.
            lr = model.update_optimizer()
            
            # incase the epoch is blocked.
            time.sleep(0.003)


if __name__ == '__main__':
    opts = RenTrainOptions().parse()
    port_id = 20000 + np.random.randint(0, 1000)
    opts.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    mp.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=len(opts.gpu_ids), args=(opts,))
