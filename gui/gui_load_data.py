
# loadding data from raw captured data.

import torch
import torch.nn as nn
import numpy as np
from torch2trt import TRTModule
import torch.multiprocessing as mp

from c_lib.VoxelEncoding.dist import VoxelEncoding # Voxel encoding library.
from utils_render.utils_render import rgb_normalizer
from c_lib.VoxelEncoding.depth_normalization import DepthNormalizer
from c_lib.VoxelEncoding.voxel_encoding_helper import OcTreeOptions, sampling_multi_view_feats, integrate_depths, volume_render, volume_render_infer, volume_render_torch
# from script.camera_parameters import parse_camera_parameters
import matplotlib.pyplot as plt
import torch.nn.functional as F

# The closed matting method;
class BGMatting(nn.Module):

    def __init__(self, path_matting, thres=0.8):
        super(BGMatting, self).__init__()
        self.matting_trt = TRTModule()
        self.matting_trt.load_state_dict( torch.load(path_matting) )
        self.matting_thres = thres

    def forward(self, imgs):
        # process two images individually.
        rgb, bg = imgs # [2,3,1440,2560]
        device = int(str(rgb.device).split(':')[-1])

        pha = self.matting_trt( rgb, bg )[0] # approx. 4ms
        output_img, mask = VoxelEncoding.tensor_selection(rgb, pha, 0.6, device) # < 1ms
        
        # the segmentented regions.
        return output_img, mask


class Undistortion(nn.Module):

    def __init__(self):
        super(Undistortion, self).__init__()

    def forward(self, bgs, rgbs, depths, kc, kd, disc, disd):
        device_Id = int(str(bgs.device).split(':')[1])
        # [N,h,w,3], [n,h,w,3], [n,h,w,1]
        # approx. 1.5ms.
        dis_bgs_rgbs, dis_d = VoxelEncoding.undistort_images(torch.cat([bgs, rgbs], 1), depths, 
                                                             kc, kd, 
                                                             disc, disd, 
                                                             device_Id)
        
        dis_bgs  = dis_bgs_rgbs[:,:3].clone()
        dis_rgbs = dis_bgs_rgbs[:,3:].clone()
        return dis_bgs, dis_rgbs, dis_d

class CropPadding(nn.Module):

    def __init__(self, opts, rank=0, vertical=True, resolution=1024, bbox_thres=100):
        super(CropPadding, self).__init__()
        self._is_vertical = vertical
        self._default_res = resolution
        self._bbox_thres  = bbox_thres
        self._rank        = rank
        self._opts        = opts
    
    def forward(self, rgbs, masks, kc):
        # masks: [N, 1, h, w]
        device_Id = int(str(rgbs.device).split(':')[1])
        # approx. 1.2ms
        bbox_min, bbox_max = VoxelEncoding.bbox_mask(masks, device_Id) # [n,2(h,w)]

        # crop, resize, padding.
        num_views, h, w = rgbs.shape[0], rgbs.shape[2], rgbs.shape[3]
        # refine the properties, wait for copy.
        rgbs_final   = torch.full([num_views,3,self._default_res,self._default_res], 0, device=rgbs.device, dtype=torch.float)
        masks_final  = torch.full([num_views,1,self._default_res,self._default_res], 0, device=rgbs.device, dtype=torch.float)
        kc_final     = torch.full([num_views,3,3], 0, device=rgbs.device, dtype=torch.float)
        
        # seperately process each view.
        for i in range(num_views):
            # get the regions.
            min_h = torch.clip(bbox_min[i,0] - self._bbox_thres, 0, h-1)
            min_w = torch.clip(bbox_min[i,1] - self._bbox_thres, 0, w-1)
            max_h = torch.clip(bbox_max[i,0] + self._bbox_thres, 0, h-1)
            max_w = torch.clip(bbox_max[i,1] + self._bbox_thres, 0, w-1)
            # cropped images.
            rgb_crop  = rgbs[i:i+1,:,min_h:max_h,min_w:max_w] # [1,3,h,w]
            mask_crop = masks[i:i+1,:,min_h:max_h,min_w:max_w] # [1,1,h,w]
            kc_new    = torch.tensor([[1,0,-min_w],[0,1,-min_h],[0,0,1]], device=kc.device, dtype=torch.float) @ kc[i] # [3,3]
            # resize rgbs. & padding.
            h_n, w_n = rgb_crop.shape[-2:]
            if h_n >= w_n: # pad l&r
                w_n_ = int(w_n/(h_n/self._default_res))
                rgb_resize  = F.interpolate( rgb_crop, (self._default_res, w_n_), mode='bilinear' )
                mask_resize = F.interpolate( mask_crop, (self._default_res, w_n_), mode='nearest' )
                kc_new = torch.tensor([[w_n_/w_n,0,0],[0,self._default_res/h_n,0],[0,0,1]], device=kc.device, dtype=torch.float) @ kc_new # [3,3]
                
                pad_l = np.floor( (self._default_res - w_n_) / 2 )
                pad_r = np.ceil( (self._default_res - w_n_) / 2 )
                rgb_padding  = F.pad(rgb_resize, (int(pad_l),int(pad_r),0,0), mode='constant', value=0)
                mask_padding = F.pad(mask_resize, (int(pad_l),int(pad_r),0,0), mode='constant', value=0)
                kc_new = torch.tensor([[1,0,pad_l],[0,1,0],[0,0,1]], device=kc.device, dtype=torch.float) @ kc_new # [3,3]
            else: # pad u&d
                h_n_ = int(h_n/(w_n/self._default_res))
                rgb_resize  = F.interpolate( rgb_crop, (h_n_, self._default_res), mode='bilinear' )
                mask_resize = F.interpolate( mask_crop, (h_n_, self._default_res), mode='nearest' )
                kc_new = torch.tensor([[self._default_res/w_n,0,0],[0,h_n_/h_n,0],[0,0,1]], device=kc.device, dtype=torch.float) @ kc_new # [3,3]
                
                pad_u = np.floor( (self._default_res - h_n_) / 2 )
                pad_d = np.ceil( (self._default_res - h_n_) / 2 )
                rgb_padding  = F.pad(rgb_resize, (0,0,int(pad_u),int(pad_d)), mode='constant', value=0)
                mask_padding = F.pad(mask_resize, (0,0,int(pad_u),int(pad_d)), mode='constant', value=0)
                kc_new = torch.tensor([[1,0,0],[0,1,pad_u],[0,0,1]], device=kc.device, dtype=torch.float) @ kc_new # [3,3]
            
            # rotation 90 degree.
            # if self._is_vertical: # always rotate 90 degree here.
                # only full body or ( portrait and rank==1 ), rotate :
            if self._opts.rend_full_body or (not self._opts.rend_full_body and self._rank == 1):
                rgb_padding  = torch.rot90( rgb_padding, self._opts.rot_degree, [2,3])
                mask_padding = torch.rot90( mask_padding, self._opts.rot_degree, [2,3])
                if self._opts.rot_degree == 1:
                    kc_new       = torch.tensor([[0,1,0],[-1,0,self._default_res],[0,0,1]], device=kc.device, dtype=torch.float) @ kc_new # [3,3]
                else:
                    kc_new       = torch.tensor([[0,-1,self._default_res],[1,0,0],[0,0,1]], device=kc.device, dtype=torch.float) @ kc_new # [3,3]

            # copy data to buffers.
            rgbs_final[i:i+1].copy_(rgb_padding)
            masks_final[i:i+1].copy_(mask_padding)
            kc_final[i].copy_(kc_new)

        return rgbs_final, masks_final, kc_final

class DepthToColor(nn.Module):

    def __init__(self, res, dsize=1, dthresh=0.05):
        super(DepthToColor, self).__init__()
        self._w, self._h = res[0], res[1]
        self._dize = dsize
        self._dthresh = dthresh # default as 5cm
    
    def forward(self, depths, masks, kc, kd, rt_d2c):
        device_Id = int( str(depths.device).split(':')[1] )
        # approx. 0.13ms
        depths_final = VoxelEncoding.depth2color(depths, masks, kc, kd, rt_d2c, 
                                                    self._w, self._h, self._dize, self._dthresh,
                                                    device_Id)
        return depths_final


# Dataloader processor from raw loading data.
class RawDataProcessor:
    # 1. pre loading cam data.
    # 2. matting, clipping, padding, depth2color.

    def __init__(self, rank, body_drm, 
                 bg_rgb, input_rgbs, input_depths, resolution, 
                 shared_high_res_buf, shared_low_res_buf, shared_kc_buf,
                 barriers, num_gpus, opts):
        self._num_gpus = opts.num_gpus

        self.drm       = body_drm
        self.opts      = opts
        self.device    = 'cuda:0'
        self.rank      = rank
        self.num_gpus  = num_gpus
        self.shared_high_res_buf = shared_high_res_buf
        self.shared_low_res_buf  = shared_low_res_buf
        self.shared_kc_buf       = shared_kc_buf
        self.ns        = opts.num_views // num_gpus

        self.barriers  = barriers
        # pre-loading camera's data.
        self._cam_data = self.pre_load_cam_data( opts.cameras_xml_path, opts.num_distort_params )
        # funcs.
        self._matting  = BGMatting( opts.bgmatting_path ).to( self.device )
        self._undistort = Undistortion()
        self._crop_padding = CropPadding(opts, vertical=opts.is_vertical, rank=self.rank, resolution=resolution, bbox_thres=opts.bbox_thres)
        self._depth2color  = DepthToColor( [resolution, resolution] )
        
        self.resolution = resolution # default as 1k;
        self.depth_normalizer   = DepthNormalizer(opts, divided2=False)
        self.depth_normalizer_r = DepthNormalizer(opts)
        self.rgb_mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.rgb_std  = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        self._integrate_depths = integrate_depths.to( self.device )
        self.batch_size = self.opts.batch_size

        # preload data :
        self.batch_data_cur = {};
        self.batch_data = {};
        self.stream0 = torch.cuda.Stream(self.device)
        # pre loaded data.
        self._bg_rgbs = bg_rgb

        with torch.cuda.stream( self.stream0 ):
            self.preload( [bg_rgb, input_rgbs[0], input_depths[0]], self.rank )
            self.batch_data_cur = self.batch_data.copy()
            self.preload( [bg_rgb, input_rgbs[1], input_depths[1]], self.rank )

        torch.cuda.current_stream().wait_stream(self.stream0)
        self.sync_batch_data(self.batch_data_cur, self.rank)
        # get the point-cloud's center positions.
        self.target_center = self.get_target_center(self.batch_data_cur)
    
    def get_target_center(self, output_data):
        # using original depth to get center, orignal center may not exactly right.
        inputs_data = [ output_data['rgbs_ds'], output_data['depths_ds'], output_data['masks_ds'], \
                        output_data['ks'], output_data['rts'] ]
        return self._integrate_depths.get_target_center(inputs_data, self.batch_size)[0]

    @torch.no_grad()
    def depth_refinement(self, rgbs, depths, masks):
        b = rgbs.shape[0]
        depths_n, _, mids, dists = self.depth_normalizer( depths )
        rgbs_n = (rgbs - self.rgb_mean.type_as(rgbs)) / self.rgb_std.type_as(rgbs)
        data = torch.cat( [rgbs_n, depths_n, masks], dim=1 )

        r_depths = self.drm( data ).float()

        r_depths = ( r_depths * dists.view(b,1,1,1) + mids.view(b,1,1,1) ) * masks

        del rgbs_n, data, depths_n, mids, dists

        return r_depths

    @torch.no_grad()
    def depth_normalize(self, depths):
        b = depths.shape[0]
        depths_n, _, mids, dists = self.depth_normalizer_r( depths )
        return depths_n, mids, dists

    def sync_batch_data(self, batch_data, rank):
        # 1k, 512, 512, 3*3;
        rgbs, depths_ds, masks_ds, k_c = \
            batch_data['rgbs'], batch_data['depths_ds'], batch_data['masks_ds'], batch_data['ks']
        data_low_res = torch.cat([depths_ds, masks_ds], dim=1) # [Nv,3+1,h',w']
        
        # sync batch data : approx.14ms
        self.shared_high_res_buf[rank*self.ns:(rank+1)*self.ns] *= 0.0
        self.shared_high_res_buf[rank*self.ns:(rank+1)*self.ns] += rgbs.cpu() # data to cpu.

        self.shared_low_res_buf[rank*self.ns:(rank+1)*self.ns] *= 0.0
        self.shared_low_res_buf[rank*self.ns:(rank+1)*self.ns] += data_low_res.cpu() # data to cpu.

        self.shared_kc_buf[rank*self.ns:(rank+1)*self.ns] *= 0.0
        self.shared_kc_buf[rank*self.ns:(rank+1)*self.ns] += k_c.cpu()
        self.barriers.wait()
        
        # data copy to GPU.
        rgbs_         = self.shared_high_res_buf.to(self.device, non_blocking=True)
        data_low_res_ = self.shared_low_res_buf.to(self.device, non_blocking=True)
        kc            = self.shared_kc_buf.to(self.device, non_blocking=True)
        kc[:,:2]     *= 0.5
        # upsampling or downsampling.
        depths_ds, masks_ds = data_low_res_[:,:1], data_low_res_[:,1:] # sync low_res data.
        masks = F.interpolate(masks_ds, (self.resolution, self.resolution), mode='bilinear')
        rgbs_ds = F.interpolate(rgbs_, (self.resolution // 2, self.resolution // 2), mode='bilinear')
        rgbs_n = rgb_normalizer( rgbs_, masks )
        rgbs_ds_n = rgb_normalizer( rgbs_ds, masks_ds )
        
        # write to batch-data.
        self.batch_data_cur['rgbs']      = rgbs_
        self.batch_data_cur['rgbs_ds']   = rgbs_ds
        self.batch_data_cur['rgbs_n']    = rgbs_n
        self.batch_data_cur['rgbs_ds_n'] = rgbs_ds_n
        self.batch_data_cur['depths_ds'] = depths_ds
        self.batch_data_cur['masks_ds']  = masks_ds
        self.batch_data_cur['ks']        = kc
        
        self.batch_data_cur['depths_ds_n'], self.batch_data_cur['depths_mid'], self.batch_data_cur['depths_dist'] = self.depth_normalize( depths_ds )
        self.barriers.wait()

    def preload(self, data, rank):
        # undistort & matting, processing images.
        rgbs, depths, masks, k_c = self.pre_process_inputs( rank, data ) # half of the data.
        rt_c = self._cam_data['RT_color']

        # multi-gpu needed : rgbs(1k), rgbs_ds(512), rgbs_n(1k), rgbs_ds_n(512),
        # mask_ds(512), depth_ds(512);
        rgbs_ds   = F.interpolate(rgbs, (self.resolution // 2, self.resolution // 2), mode='bilinear')
        depths_ds = F.interpolate(depths, (self.resolution // 2, self.resolution // 2), mode='nearest')
        masks_ds  = F.interpolate(masks, (self.resolution // 2, self.resolution //2), mode='nearest')
        
        self.batch_data['depth_ori'] = depths_ds.clone()
        if self.opts.using_drm:
            depths_ds = self.depth_refinement( rgbs_ds, depths_ds, masks_ds ) # using normalized ds.
            
        torch.cuda.synchronize() # better for data aggregation.

        self.batch_data['rgbs']      = rgbs
        self.batch_data['depths_ds'] = depths_ds
        self.batch_data['masks_ds']  = masks_ds
        self.batch_data['ks']        = k_c
        
        self.batch_data['rts'] = rt_c
        self.batch_data['rs']  = rt_c[...,:3].contiguous()
        self.batch_data['ts']  = rt_c[...,-1:].contiguous()

    @torch.no_grad()
    def pre_process_inputs(self, rank, data):
        bg_rgbs, rgbs, depths = data
        # half of the data.
        # step 0. get the rgb, depths, bg data.
        # data : [N_views // 2, ...].
        bgs_h  = bg_rgbs[ rank*self.ns:(rank+1)*self.ns ].to(self.device, non_blocking=True)
        rgbs_h = rgbs[ rank*self.ns:(rank+1)*self.ns ].to(self.device, non_blocking=True)
        ds_h   = depths[ rank*self.ns:(rank+1)*self.ns ].to(self.device, non_blocking=True)

        # step 1. get params. k and dis, half of the data.
        k_color  = self._cam_data['K_color'][ rank*self.ns : (rank+1)*self.ns ]
        k_depth  = self._cam_data['K_depth'][ rank*self.ns : (rank+1)*self.ns ]
        dis_d    = self._cam_data['DIS_D'][ rank*self.ns : (rank+1)*self.ns ]
        dis_c    = self._cam_data['DIS_C'][ rank*self.ns : (rank+1)*self.ns ]
        RT_D2C   = self._cam_data['RT_D2C'][ rank*self.ns : (rank+1)*self.ns ]
        
        # approx. 8~11ms.
        # step 2. undistort using the cam's intrinsic and distortion coffs.
        if self.opts.is_undistort: # apply undistortion.
            dis_bgs, dis_rgbs, dis_ds = self._undistort(bgs_h, rgbs_h, ds_h, k_color, k_depth, dis_c, dis_d)
        else:
            dis_bgs, dis_rgbs, dis_ds = bgs_h.clone(), rgbs_h.clone(), ds_h.clone()
        # step 3. matting.
        matting_rgbs, masks = self._matting( [dis_rgbs, dis_bgs] )
        # step 4. crop, pad, rotate to 1k * 1k rgbs (vertical), adjust matrix k.
        rgbs_final, masks_final, k_c_final = self._crop_padding( matting_rgbs, masks, k_color )
        # step 5. depth2color warpping;
        depths_final = self._depth2color( dis_ds, masks_final, k_c_final, k_depth, RT_D2C )
        
        return rgbs_final, depths_final, masks_final, k_c_final
        
    # the camera data should be loaded before the process the input data.
    def pre_load_cam_data(self, cam_xml_path, num_dis_params):
        cam_data = parse_camera_parameters(cam_param_path = cam_xml_path, num_dis_params=num_dis_params)
        # parse cameras' data.
        K_depths   = cam_data['K_D'] # [N,3,3]
        K_rgbs     = cam_data['K_C'] # [N,3,3]
        RT_D2C     = cam_data['RT_D2C'] # [N,3,4]
        RT_depths  = cam_data['RT_D'] # [N,3,4]
        distort_ds = cam_data['Dis_D'] # [N,5+3]
        distort_cs = cam_data['Dis_C'] # [N,5+3]
        # depths of 512 resolution
        if num_dis_params == 8: # input depth of 512 resolution.
            K_depths[:,:2] *= 0.5
        
        # adjust R_color;
        len_cams   = K_depths.shape[0]
        RT_D2C_4x4 = np.stack( [np.eye(4)]*len_cams, axis=0 ) # [N,4,4]
        RT_D2C_4x4[:,:3,:] = RT_D2C
        RT_depths_4x4 = np.stack( [np.eye(4)]*len_cams, axis=0 ) # [N,4,4]
        RT_depths_4x4[:,:3,:] = RT_depths

        RT_depths_4x4 = np.linalg.inv(RT_depths_4x4) # [N,4,4]
        RT_rgbs = RT_D2C_4x4 @ RT_depths_4x4 # [N,4,4] @ [N,4,4]
        RT_rgbs = RT_rgbs[:,:3] # [N,3,4]
        RT_depths = RT_depths_4x4[:,:3] # [N,3,4]
        
        selected_idxs = self.opts.test_view_ids[:self.opts.num_views] # e.g., [4,6,7,0]'s data.
        # return the selected cams' params.
        
        return {
            'K_color': torch.tensor( K_rgbs ).float()[selected_idxs].to(self.device),
            'K_depth': torch.tensor( K_depths ).float()[selected_idxs].to(self.device),
            'RT_color': torch.tensor( RT_rgbs ).float()[selected_idxs].to(self.device),
            'RT_depth': torch.tensor( RT_depths ).float()[selected_idxs].to(self.device),
            'RT_D2C' : torch.tensor( RT_D2C ).float()[selected_idxs].to(self.device),
            'DIS_D' : torch.tensor( distort_ds ).float()[selected_idxs].to(self.device),
            'DIS_C' : torch.tensor( distort_cs ).float()[selected_idxs].to(self.device)
        }

    def next(self, nxt_data):
        # loading nxt_data.
        input_rgbs, input_depths = nxt_data
        # update data.
        b_cur = self.batch_data_cur.copy()
        torch.cuda.current_stream().wait_stream(self.stream0)
        # for nxt frame's data buffer.
        self.batch_data_cur = self.batch_data.copy()
        self.sync_batch_data( self.batch_data_cur, self.rank )
        b_nxt = self.batch_data_cur.copy()
        
        with torch.cuda.stream(self.stream0):
            self.preload( [self._bg_rgbs, input_rgbs, input_depths], self.rank )

        return b_cur, b_nxt


if __name__ == '__main__':
    from gui.RenTestOptions import RenTestOptions
    opts = RenTestOptions().parse()

    opts.matting_path     = './accelerated_models/matting_trt_parallel.pth'
    opts.cameras_xml_path = ''
    opts.num_gpus         = 2
    torch.set_num_threads(1)

    data_processor = RawDataProcessor( opts )