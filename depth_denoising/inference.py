# test the depth denoising on real-captured data.

import os
import torch
import torch.nn.functional as F
from options.RenTrainOptions import RenTrainOptions

import numpy as np
import cv2
# two depth-refinement network.
from depth_denoising.net import BodyDRM2, DepthRefineModule, BodyDRM3
from utils_render.open3d_depth_fusion import O3DepthFusion
from utils_render.utils_render import normalize_rgbd, unnormalize_depth
from c_lib.VoxelEncoding.depth_normalization import DepthNormalizer
import utils_render.util as util

opts = RenTrainOptions().parse()
opts.batch_size = 1

# loading network.
def initialize(device='cuda:0'):
    # loading model.
    drm_path = './checkpoints_rend/SAILOR/latest_model_BodyDRM2.pth'
    bodydrm = BodyDRM2(opts, device).to(device).eval()

    o3d_fusion = O3DepthFusion(voxel_len=1 / 512.0, sdf_trunc=0.01, depth_trunc=1.5 + 1.0, write_able=True, view_able=False)
    bodydrm.load_state_dict(torch.load(drm_path), strict=False)
    depth_normalizer = DepthNormalizer(opts, divided2=False)
    return o3d_fusion, bodydrm, depth_normalizer

def inference(o3d_fusion, bodydrm, depth_normalizer, data_basic_dir, output_dir, frame_idx, view_id='4', device='cuda:0'):
    # load data.
    rgb = cv2.imread(os.path.join(data_basic_dir, 'COLOR', frame_idx, view_id + '.jpg'))[...,::-1] / 255.0
    depth = cv2.imread(os.path.join(data_basic_dir, 'DEPTH', frame_idx, view_id + '.png'), -1) / 10000.0
    mask = cv2.imread(os.path.join(data_basic_dir, 'MASK', frame_idx, view_id + '.png'))[..., 0] / 255.0
    cam_param = np.load(os.path.join(data_basic_dir, 'PARAM', frame_idx, view_id + '.npy'), allow_pickle=True) # K, RTS
    mask[mask >= 0.5] = 1; mask[mask < 0.5] = 0
    K, RT = cam_param.item().get('K'), cam_param.item().get('RT')
    K[:2] *= 0.5
    rgb *= mask[...,None]; depth *= mask
    data = np.concatenate([rgb, depth[...,None], mask[...,None]], axis=-1)
    data = torch.from_numpy(data).permute(2,0,1)[None].float()
    data = F.interpolate(data, (512, 512), mode='nearest').to(device)

    depths_normalized, depth_mid, depth_dist, rgbs_normalized = normalize_rgbd(data[:,:3], data[:, 3:4], depth_normalizer)
    data = torch.cat([rgbs_normalized, depths_normalized, data[:, -1:]], dim=1)
    output = bodydrm(data)
    output = unnormalize_depth(output, depth_mid, depth_dist) * data[:,-1:]

    rgb = data[:,:3]; depths_ds = output
    rgb_datas = rgb.permute(0,2,3,1).cpu().numpy() * 255
    rgb_datas = np.asarray(rgb_datas, order="C")
    rgb_datas *= 0
    rgb_datas += 200;
    depth_datas = depths_ds[:,0].cpu().detach().numpy()
    Ks = np.stack([K]*1, axis=0)
    RTs = np.stack([RT]*1, axis=0)
    new_K = np.array([[320, 0, 256],[0, 320, 256],[0,0,1]], dtype=np.float32)
    new_K = np.stack([new_K]*1, axis=0) # [4,3,3]
    new_RT = (np.linalg.inv(new_K) @ Ks) @ RTs # [4,3,4]
    # save the refined depth maps.
    util.make_dir(output_dir)
    o3d_fusion.fusion(rgb_datas, depth_datas, new_RT, new_K, True, output_dir + 'refined_pcd_' + frame_idx + '.ply')
    
    # save the original depth maps.
    rgb_datas *= 0
    rgb_datas[..., 0] += 150;
    rgb_datas[..., 1] += 150;
    rgb_datas[..., 2] += 200;
    depth_datas = cv2.resize(depth, (512,512), interpolation=cv2.INTER_NEAREST)[None]
    o3d_fusion.volume.reset()
    o3d_fusion.fusion(rgb_datas, depth_datas, new_RT, new_K, True, output_dir + 'ori_pcd_' + frame_idx + '.ply')

    print('write depth(point cloud) to ./depth_denoising/results/test_pcd.ply')

if __name__ == '__main__':
    basic_path = './test_data/static_data1/'
    output_dir = './depth_denoising/results/'
    frame_idx  = 'FRAME3746'
    o3d_fusion, bodydrm, depth_normalizer = initialize()
    inference(o3d_fusion, bodydrm, depth_normalizer, basic_path, output_dir, frame_idx, '4')