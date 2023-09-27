"""
    Utils for generating mesh, calculating errors.
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import sys, os
from tqdm import tqdm
from utils_render.mesh_util import save_obj_mesh, reconstruction
from models.data_utils import unnormalize_depth, normalize_depth, normalize_face_depth
import cv2

def distributed_concat(tensor):
    output_tensors = [tensor.clone().detach().contiguous() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor.detach().contiguous()) # reduce all tensors.
    # torch.distributed.gather(tensor, output_tensors, dst=torch.distributed.get_rank()) # reduce all tensors in a single process.
    concat = torch.cat(output_tensors, dim=0)
    return concat.contiguous()

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = float(0)
        self.val = float(0)
        self.sum = float(0)
        self.count = int(0)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def save_points(model_dir, points):
    ply_dir = os.path.join(model_dir, 'val_results')
    make_dir(ply_dir)
    save_path = os.path.join(ply_dir, 'boundary_points.ply') # the boundary points.
    points_ = points.detach().permute(1,0).cpu().numpy() # [N, 3]
    colors = np.zeros_like(points_) # [N, 3]
    to_save = np.concatenate([points_, colors], axis=-1)
    # points_ shape[0] : N_points.
    return np.savetxt(save_path,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points_.shape[0])
                      )

def save_samples_truncted_prob(model_dir, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    '''
    ply_dir = os.path.join(model_dir, 'val_results')
    make_dir(ply_dir)
    save_path = os.path.join(ply_dir, 'pred.ply')
    # r = (prob > 0.5).reshape([-1, 1]) * 255
    # b = (prob < 0.5).reshape([-1, 1]) * 255
    # g = np.zeros(r.shape)
    r = (prob > 0.5).reshape([-1, 1]) * 235
    b = (prob > 0.5).reshape([-1, 1]) * 235
    g = (prob > 0.5).reshape([-1, 1]) * 235

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(save_path,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )
# self.visual_names = ['rgbs', 'depths', 'masks', 'face_mask'
                                #  'd_body_rf', 'd_face_rf', 'rgb_body_rf', 'rgb_face_rf'

def save_images(opts, visuals, subject_name, epoch=0):
    # saving images for visualization.
    # [N_v * 1, 3, H, W]; [N_v * 1, 1, H, W], [N_v * 1, 1, H, W]; [N_v * 1,  3, H, W];
    # if opts.is_train:
    #     images, depths, normals_refine, depths_refine, masks, depths_gt = visuals['images'], visuals['depths'], visuals['normals_refine'], \
    #                                                                       visuals['depths_refine'], visuals['masks'], visuals['depths_gt']
    rgbs, depths, masks, face_masks, d_body_rf, nl_body, rgb_face_sr = \
        visuals['rgbs'], visuals['depths'], visuals['masks'], visuals['face_mask'], \
        visuals['d_body_rf'], visuals['nl_body'], \
        visuals['face_rgb']

    # if opts.is_train: # contains ground_truth depth.
        # front_face_depths_gt = visuals['front_face_depths_gt']
                                                          
    image_dir = os.path.join(opts.model_dir, 'val_results')
    make_dir(image_dir)
    depth_dir = os.path.join(opts.model_dir, 'val_results', '{}_epoch{}_raw_depth'.format(subject_name, epoch))
    make_dir(depth_dir)
    save_path = os.path.join(image_dir, '{}_epoch{}.jpg'.format(subject_name, epoch))

    save_list_rgbs    = []
    save_list_depths  = []
    save_list_d_rf    = []
    # save_list_rgbs_rf = []
    save_list_nl      = []
    
    assert rgbs.shape[0] == depths.shape[0], 'num of images must be same.'
    
    # padding the rgbs & depths, the length == num_views + 1;
    _rgbs = rgbs.view(-1, opts.num_views, 3, opts.load_size, opts.load_size) # [B, N, 3, H, W]
    rgb_face_sr = nn.UpsamplingBilinear2d(size=(opts.load_size, opts.load_size))(rgb_face_sr)
    _rgb_face_sr = rgb_face_sr.view(-1, 1, 3, opts.load_size, opts.load_size)
    _rgbs = torch.cat([_rgb_face_sr, _rgbs], 1).reshape(-1, 3, opts.load_size, opts.load_size) # [B * 4, 3, H, W]

    _depths = depths.view(-1, opts.num_views, 1, opts.load_size, opts.load_size)
    _depths = torch.cat([torch.ones_like(_depths)[:, 0:1] * 0.5, _depths], 1).reshape(-1, 1, opts.load_size, opts.load_size)
    
    _nl_body = nl_body.view(-1, opts.num_views, 3, opts.load_size, opts.load_size) # [B, N, 3, H, W]
    _normals = torch.cat([torch.ones_like(_nl_body)[:, 0:1] * 0.5, _nl_body], 1).reshape(-1, 3, opts.load_size, opts.load_size) # [B * 4, 3, H, W]
    
    _d_body_rf = d_body_rf.view(-1, opts.num_views, 1, opts.load_size, opts.load_size)
    _depths_rf = torch.cat([torch.ones_like(_d_body_rf)[:, 0:1] * 0.5, _d_body_rf], 1).reshape(-1, 1, opts.load_size, opts.load_size) # [B * 4, 3, H, W]
    
    _masks = masks.view(-1, opts.num_views, 1, opts.load_size, opts.load_size)
    face_masks = nn.UpsamplingBilinear2d(size=(opts.load_size, opts.load_size))(face_masks)
    _face_masks = face_masks.view(-1, 1, 1, opts.load_size, opts.load_size)
    _masks = torch.cat([_face_masks, _masks], 1).reshape(-1, 1, opts.load_size, opts.load_size)
    
    # saving depth & original images.
    for v in range(_rgbs.shape[0]):
        # unnormalize to [0,1]; [3, H, W] & [1, H, W]; [3, H, W] for normal.
        save_rgbs         = (np.transpose(_rgbs[v].detach().cpu().numpy(), (1, 2, 0))) * 255.0
        # save_rgbs_refine  = (np.transpose(rgbs_rf[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5) * 255.0
        save_depth        = np.transpose(_depths[v].detach().cpu().numpy(), (1, 2, 0)) # [H, W, 1]; 
        save_normals      = np.transpose(_normals[v].detach().cpu().numpy(), (1, 2, 0)) * 255.0 # located in [-1, 1]
        save_depth_refine = np.transpose(_depths_rf[v].detach().cpu().numpy(), (1, 2, 0)) # [H, W, 1];
        save_mask         = np.transpose(_masks[v].detach().cpu().numpy(), (1, 2, 0)) # [H, W, 1];
        # unnormalize: depth = ((depth - self.opts.z_size) / (self.opts.z_size // 2)) * (self.opts.load_size // 2) / self.opts.z_size
        # save refined depth.
        cv2.imwrite(os.path.join(depth_dir, 'depth_view_%d.png'%(v % (opts.num_views + 1))), (save_depth_refine * 10000).astype(np.uint16))
        cv2.imwrite(os.path.join(depth_dir, 'normal_view_%d.png'%(v % (opts.num_views + 1))), save_normals[:, :, ::-1].astype(np.uint8))
        # save facial mask; the facial mask.
        if v % (opts.num_views + 1) == 0:
            cv2.imwrite(os.path.join(depth_dir, 'facial_mask.png'), (save_mask * 255).astype(np.uint8))

        # for raw noised depth.
        if v % (opts.num_views + 1) != 0: # normalize
            save_depth = (save_depth - opts.z_size) / opts.z_bbox_len
            
        save_depth *= save_mask
        save_depth = np.clip((save_depth + 1) * 0.5, 0, 1)
        save_depth *= save_mask
        save_depth = np.concatenate([save_depth] * 3, axis=-1) # to [H,W,3];

        # for refined depth.
        if v % (opts.num_views + 1) == 0: # only for the faces;
            save_depth_refine = normalize_face_depth(opts, save_depth_refine)
            save_depth_refine *= save_mask # we don't multiply by the mask here and we want the background to be gray
            save_depth_refine = np.clip((save_depth_refine + 1) * 0.5, 0, 1)
        else:
            save_depth_refine = save_depth_refine * save_mask
            save_depth_refine = save_depth_refine + (1 - save_mask) * opts.z_size
            max_depth, min_depth = np.max(save_depth_refine), np.min(save_depth_refine)
            save_depth_refine = (save_depth_refine - min_depth) / (max_depth - min_depth)
        
        save_depth_refine = np.concatenate([save_depth_refine] * 3, axis=-1) # to [H,W,3];

        # save to list; TO RGB data type.
        save_list_rgbs.append(save_rgbs[:, :, ::-1])
        save_list_depths.append(save_depth * 255.0)
        # save_list_rgbs_rf.append(save_rgbs_refine[:, :, ::-1])
        save_list_d_rf.append(save_depth_refine * 255.)
        save_list_nl.append(save_normals[:, :, ::-1])
    
    # [2 * H, W * N_view, 3]
    im1 = np.concatenate(save_list_rgbs + save_list_depths, axis=1)
    im2 = np.concatenate(save_list_nl + save_list_d_rf, axis=1)
    im  = np.concatenate([im1, im2], axis=0)
    # cv2.imwrite(save_path, im)
    Image.fromarray(np.uint8(im[:,:,::-1])).save(save_path)

    # save the depth map to raw_file, the shape of the depths_refine: [B, 1, H, W]
    # if opts.save_depths_normals_npy:
    #     depths = (depths * masks).detach().cpu().numpy()
    #     depths_refine_unnorm = (depths_refine * masks).detach().cpu().numpy()
    #     # depths_gt = unnormalize_depth(opts, depths_gt).detach().cpu().numpy()
    #     normals_refine = normals_refine.detach().cpu().numpy()
    #     np.save(os.path.join(image_dir, '{}_depths_refined_epoch{}.npy'.format(subject_name, epoch)), depths_refine_unnorm)
    #     # np.save(os.path.join(image_dir, '{}_depths_gt_epoch{}.npy'.format(subject_name, epoch)), depths_gt)
    #     # np.save(os.path.join(image_dir, '{}_normals_refine_epoch{}.npy'.format(subject_name, epoch)), normals_refine)
    #     np.save(os.path.join(image_dir, '{}_depths_raw_epoch{}.npy'.format(subject_name, epoch)), depths)
    #     np.save(os.path.join(image_dir, '{}_mask_epoch{}.npy'.format(subject_name, epoch)), masks.detach().cpu().numpy())
    
    # to save the per-view depth mesh:
    # if opts.save_depths_ply:
    #     import open3d as o3d
    #     depths_refine_unnorm = (depths_refine * masks).detach().cpu().numpy()
    #     if opts.is_train:
    #         depths_refine_unnorm = depths_refine_unnorm[1, 0]
    #     else:
    #         depths_refine_unnorm = depths_refine_unnorm[2, 0]
            
    #     save_point_clouds(opts, depths_refine_unnorm, subject_name, epoch)


def save_point_clouds(opts, depth, subject_name, epoch=0):
    # To save the point clouds from raw input depth images, the intrinsic matrix from val_data;
    # the depths are from visuals results, the Ks are from validation data (the matrix RT is assigned as I)
    import open3d as o3d
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=0.8 / 128,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8
    )
    # assign the intrinsic matrix.
    # K = np.array([[320, 0, 256], [0, 320, 256], [0, 0, 1]])
    RT = np.eye(4)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(512, 512, 320, 320, 256, 256)
    rgb_color = np.ones([depth.shape[0], depth.shape[1], 3]).astype(np.uint8) * 255
    color = o3d.geometry.Image(rgb_color)
    depth = o3d.geometry.Image(depth.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1,
                                                                depth_trunc=1.5 + 1, convert_rgb_to_intensity=False)
    volume.integrate(rgbd, intrinsic, RT.astype(np.float64))
    mesh = volume.extract_triangle_mesh()
    image_dir = os.path.join(opts.model_dir, 'val_results')
    o3d.io.write_triangle_mesh(os.path.join(image_dir, '{}_depth.ply'.format(subject_name, epoch)), mesh)

def compute_acc(pred, gt, thresh=0.5):
    '''
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float() # the inter sum

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
            
        # inter / union; inter / pred_vol(1); inter / gt_vol(1);
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


def evaluate_metric(opts, model, dataset, device, num_val=100):
    # evaluate the error between the gt occ & predicted occ.
    if num_val > len(dataset):
        num_val = len(dataset)

    with torch.no_grad():
        IOUs, Precs, Recalls = [], [], []
        for idx in tqdm(range(num_val)):
            val_data = dataset[idx] # query a validate data.
            labels = val_data['labels'].to(device)
            # retrieve the input validate data.
            model.set_input(val_data)
            # calculate the output predicted occupancy.
            model.forward()
            # get labels for the queried 3d points.
            IOU, Prec, Recall = compute_acc(model.pred_occ, labels)
            IOUs.append(IOU.item())
            Precs.append(Prec.item())
            Recalls.append(Recall.item())
        
    return np.average(IOUs), np.average(Precs), np.average(Recalls)

def gen_mesh(opts, model, val_data, device, epoch, use_octree=True):
    # get bbox for generate data.
    bbox_min = val_data['b_min'][0] # [1, 3]
    bbox_max = val_data['b_max'][0] # [1, 3]

    mesh_dir = os.path.join(opts.model_dir, 'val_results')
    make_dir(mesh_dir)
    geo_save_path = os.path.join(mesh_dir, '{}_epoch{}.obj'.format(val_data['name'][0], epoch))
    # assert os.path.exists(geo_save_path), 'no exist path for generating mesh.'
    # try:
    with torch.no_grad():
        verts, faces, _, _ = reconstruction(opts, model, device, opts.num_views, opts.resolution, 
                                        bbox_min, bbox_max, use_octree=use_octree)
    save_obj_mesh(geo_save_path, verts, faces)

    # except Exception as e:
    #     print(e)
    #     print('can not generate mesh now.')

def gen_texed_mesh(opts, val_data, epoch, device):
    import open3d as o3d
    mesh_dir = os.path.join(opts.model_dir, 'val_results')
    make_dir(mesh_dir)
    geo_save_path = os.path.join(mesh_dir, '{}_epoch{}.obj'.format(val_data['name'][0], epoch))