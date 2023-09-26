"""
    Depth denoising training dataset.
"""

import sys, os
import trimesh
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import cv2
import numpy as np
import random

from scipy.spatial.transform import Rotation as R
from models.data_utils import projection3D
from c_lib.AugDepth import DepthAug # depth augmentation.
import matplotlib.pyplot as plt
from models.utils import index2D
import time

import logging

"""
    The Depth-denoising training dataloader provide: 
    1. the original multi-view RGBs information.
    2. the original depth & mask images.
    3. the corresponding cameras' parameters.
    4. the argumented depth masp.
    The training dataset are from the Thuman 2.0 dataset.
"""

class RenTrainDataloader(object):

    def __init__(self, opts, phase='training'):
        super(RenTrainDataloader, self).__init__()
        self.dataset = RenTrainDataset(opts, phase)
        # logging.basicConfig(level=logging.INFO if torch.distributed.get_rank() in [-1, 0] else logging.WARN)
        # logging.info('Dataset [%s] was created' % type(self.dataset).__name__)
        
        # if opts.support_DDP: # using DDP dataloader.
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, 
        #                                                                     shuffle=not opts.serial_batches if phase == 'training' else False)
        #     self.dataloader = data.DataLoader(
        #         self.dataset,
        #         batch_size=opts.batch_size if phase == 'training' else 1,
        #         # shuffle=not opts.serial_batches,
        #         num_workers=int(opts.num_threads),
        #         pin_memory=opts.pin_memory,
        #         sampler=train_sampler
        #     )
        
        # else: # simple dataloader.
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=opts.batch_size if phase == 'training' else 1,
            shuffle=not opts.serial_batches if phase == 'training' else False,
            num_workers=int(opts.num_threads),
            pin_memory=opts.pin_memory
        )
        
    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for data in self.dataloader:
            yield data
    
    def get_iter(self):
        return iter(self)
    

class RenTrainDataset(object):
    
    def __init__(self, opts, phase='training'): # model phase: training, validating
        super(RenTrainDataset, self).__init__()
        self.opts     = opts
        self.is_train = opts.is_train
        self._phase   = phase
        
        self.num_views = opts.num_views
        self.root_dir  = opts.ren_data_root
        self.target_num_views = opts.target_num_views
        
        # get data dirs.
        self.OBJ    = opts.basic_obj_dir
        self.RGB    = os.path.join(self.root_dir, 'RENDER')
        self.MASK   = os.path.join(self.root_dir, 'MASK')
        self.DEPTH  = os.path.join(self.root_dir, 'DEPTH')

        self.CAM    = os.path.join(self.root_dir, 'PARAM')

        # other properties. e.g., default projective type: perspective.
        self.projection_mode = opts.project_mode if opts.project_mode is not None else 'perspective' # else ortho
        
        # default resolution, i.e., 512;
        self.IMG_SIZE = opts.img_size
        # the images' rendering list, reduce to 60 views, sampling views degrees.
        self.yaw_list   = list(range(0, 360, 6)) # total 120' yaw views, the seperate degree is 3.
        # self.yaw_list   = list(range(0, 8, 1)) # fit the rendering testing.
        self.real_order = [0,1,7,3,4,5,6,2] # this order
        self.pitch_list = [0]
        
        # get all the subjects.
        self.subjects = self.get_subjects()
        
        # PIL to Image.
        self.rgb_to_tensor = transforms.Compose([
            transforms.Resize([self.IMG_SIZE, self.IMG_SIZE]), # resize: to 512*512;
            transforms.ToTensor()
        ])
        # no image's augmentation.
        self.dot_pattern_  = cv2.imread('./data/kinect-pattern_3x3.png', 0).astype(np.float32)
        
    def augment_cam_params(self, RT, degree=1.0, dis=0.005):
        # augment the camera's parameters.
        rot_vec = R.from_matrix(RT[:, :3]).as_rotvec()
        rot_vec += np.random.randn(3) * np.pi / 180.0 * degree; # 1 degree error.
        RT[:, :3] = R.from_rotvec(rot_vec).as_matrix()
        RT[:, -1:] += np.random.randn(3,1) * dis # 5ms
        return RT

    def augment_depth_maps_new(self, depth, mask, K):
        K_new = K.clone(); K_new[0,0] = 320;
        aug_depth_map = np.copy(depth).astype(np.float32)
        blurred_depth_map = np.copy(depth).astype(np.float32) * 0.0
        # depths_blur = cv2.GaussianBlur(depth, (3,3), 1.5) # blur the depths maps. (No depth blurs.)
        DepthAug.depth_blur( depth.astype(np.float32), blurred_depth_map, 7, 0.03 )
        DepthAug.aug_depth(blurred_depth_map.astype(np.float32), mask.astype(np.float32), self.dot_pattern_, K_new.numpy().astype(np.float32), aug_depth_map,
                           self.opts.scale_factor, self.opts.baseline_m, self.opts.invalid_disp, self.opts.z_size, self.opts.holes_rate * 2.0, self.opts.sigma_d * 1.5)               
        return aug_depth_map
    
    def augment_depth_maps(self, depth, mask):
        # input depth : [h,w], mask : [h,w]
        # 1. erode borders.
        randv0 = np.random.rand()
        if_erode = True
        if randv0 > 0.6:
            self._erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        elif randv0 > 0.4 and randv0 <= 0.6:
            self._erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        elif randv0 > 0.2 and randv0 <= 0.4:
            self._erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        else:
            if_erode = False
            
        # 1.' add holes.
        mask_holes = np.copy(mask)
        # : get the mask regions.
        if np.random.rand() > 0.5:
            h_idx, w_idx = np.where(mask != 0) # get the non-empty pixels.
            # get random idx.
            valid_mask_size = h_idx.size
            holes_rate = 0.001 + np.random.rand() * (0.03 - 0.001) # [0.001,0.02]
            rand_idx = np.random.randint( 0, valid_mask_size, size=[ int(valid_mask_size * holes_rate) ] ) # [num_rays]
            sampled_x = w_idx[rand_idx] # get the sampled idx.
            sampled_y = h_idx[rand_idx] # get the sampled idx.
            for i in range(sampled_x.size): # add holes in each 
                idx_x = sampled_x[i]; idx_y = sampled_y[i];
                randv_ = np.random.rand()
                if randv_ > 0.7:
                    k_size = 5
                elif randv_ > 0.3 and randv_ <= 0.7:
                    k_size = 3
                else:
                    k_size = 1
                min_y = idx_y - k_size // 2 if idx_y - k_size // 2 >= 0 else 0
                max_y = idx_y + k_size // 2 + 1 if idx_y + k_size // 2 + 1 <= self.opts.load_size else self.opts.load_size
                min_x = idx_x - k_size // 2 if idx_x - k_size // 2 >= 0 else 0
                max_x = idx_x + k_size // 2 + 1 if idx_x + k_size // 2 + 1 <= self.opts.load_size else self.opts.load_size
                for j in range(min_y, max_y):
                    for k in range(min_x, max_x):
                        mask_holes[j,k] = 0 # assign with 0;
        
        # 2. noised depths.
        # step 1. add gaussian blur.
        randv = np.random.rand()
        if randv > 0.8:
            depth_ = np.copy(depth).astype(np.float32)
            DepthAug.depth_blur( depth_, depth * 0.0, 7, 0.03 )
        elif randv > 0.5 and randv <= 0.8:
            depth_ = np.copy(depth).astype(np.float32)
            DepthAug.depth_blur( depth_, depth * 0.0, 5, 0.03 )
        elif randv > 0.2 and randv <= 0.5:
            depth_ = np.copy(depth).astype(np.float32)
            DepthAug.depth_blur( depth_, depth * 0.0, 3, 0.03 )
            
        # step 2. add gaussian noises
        randv1 = np.random.rand()
        if randv1 > 0.7:
            depth += np.random.normal(0, 0.015, depth.shape) # 2cm depth noise, large noises.
        elif randv1 > 0.3 and randv1 <= 0.7:
            depth += np.random.normal(0, 0.0075, depth.shape) # 1cm depth noise
        else:
            depth += np.random.normal(0, 0.001, depth.shape) # 0.1cm depth noise, a little noises here.
        
        if if_erode:
            mask_eroded = cv2.erode(mask, self._erode_kernel)
            depth *= mask_eroded * mask_holes # masked the depth.
        else:
            depth *= mask * mask_holes

        return depth

    def process_vertices_data(self, obj_path):
        # for the input object data. Obtain the surface vertices data.
        target_obj   = trimesh.load_mesh(obj_path)
        vertices     = np.array(target_obj.vertices) # [N, 3]
        num_vertices = vertices.shape[0] # num of the vertices points.
        if num_vertices >= self.opts.max_vertices_num: # select first N's points.
            rand_idx = np.random.choice( list(range(num_vertices)), int(self.opts.max_vertices_num), replace=False )
            vertices = vertices[ rand_idx ] # num of the vertices.

        else: # random select M - N points, and append to the list.
            rand_idx = np.random.choice( list(range(num_vertices)), int(self.opts.max_vertices_num)-num_vertices, replace=True )
            verts_appended = vertices[rand_idx]
            # fill the vertices.
            vertices = np.concatenate([vertices, verts_appended], axis=0)
        
        return torch.from_numpy(vertices).float() # to float datatype.
    
    def get_subjects(self):
        # loading training & validating.
        # all_subjects = sorted(os.listdir(self.OCC_SAMPLE_FINE))
        # all_subjects = sorted(  os.listdir(self.MAIN_FACE) )
        all_subjects = sorted( os.listdir(self.RGB) )
        
        # validating subjects.
        val_subjects = np.loadtxt( os.path.join(self.root_dir, 'val_data.txt'), dtype=str )
        
        if len(val_subjects) == 0:
            return all_subjects
        
        if self.is_train:
            if self._phase == 'training':
                subjects = list( set(all_subjects) - set(val_subjects) ) # only keep the training dataset.
            else:
                subjects = list(val_subjects)
        else: # when inference.
            subjects = all_subjects
        
        random.shuffle(subjects)
        return subjects

    def __len__(self): # the dataset's size : num(object) * num(yaw_list), e.g., 120 * 420.
        if self._phase == 'training':
            return len(self.subjects) * len(self.yaw_list)
        else:
            return len(self.subjects)      
    
    def get_RGBD_cam_data(self, subject, view_ids, aug_depths=True, aug_cam=False, aug_img=True):
        rgbs_list   = []; depths_list = []; masks_list = []; 
        gt_depths_list = []; # groud-truth depths should by append in the dataset
        Ks_list = []; RTs_list = [];
        pid = 0;
        
        for i, vid in enumerate(view_ids):
            # get the rendered RGBD sequences.
            rgb_path   = os.path.join( self.RGB, subject, '%d_%d.jpg' % (vid, pid) )
            mask_path  = os.path.join( self.MASK, subject, '%d_%d.png' % (vid, pid) )
            depth_path = os.path.join( self.DEPTH, subject, '%d_%d.png' % (vid, pid) )
            param_path = os.path.join( self.CAM, subject, '%d_%d.npy' % (vid, pid) )
            
            render = Image.open(rgb_path).convert('RGB')
            mask   = Image.open(mask_path).convert('L')
            depth  = cv2.imread(depth_path, -1) / 10000 # to m unit.
            cam_param = np.load(param_path, allow_pickle=True) # K, RTS
            
            K, RT = cam_param.item().get('K'), cam_param.item().get('RT')
            
            # translate under images's pixels space, randomly shift operation.
            trans_intrinsic = np.identity(3)
            
            if self.is_train and aug_img: # augment the images.
                pad_size = int(0.1 * self.IMG_SIZE) # e.g., 51
                render = ImageOps.expand(render, pad_size, fill=0) # padding the target images.
                mask   = ImageOps.expand(mask, pad_size, fill=0) # padding the mask in the same way.
            
                w, h = render.size
                tw, th = self.IMG_SIZE, self.IMG_SIZE # target image's resolution.
                
                # padding depth.
                depth_expand = np.zeros([h,w], dtype=np.float32) # non-depth areas, default as 0.
                depth_expand[pad_size:-pad_size, pad_size:-pad_size] = depth
                
                if not self.opts.no_random_shift:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0; dy = 0;
                # [1, 0, -dx; 0, 1, -dy; 0, 0, 1]
                
                # when dx > 0, the obj move left, and crop's x move right.
                trans_intrinsic[0, 2] = -dx
                trans_intrinsic[1, 2] = -dy
                
                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy 

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))
                depth = depth_expand[y1:y1 + th, x1:x1 + tw]
            
            if aug_cam and np.random.rand() > 0.6: # augment the cameras' parameters.
                RT = self.augment_cam_params(RT, degree=0.1, dis=0.001) # 3mm, 0.3 degree err.
                
            # cam to tensor.
            K  = torch.as_tensor(trans_intrinsic @ K, dtype=torch.float32)
            RT = torch.as_tensor(RT, dtype=torch.float32)
            if self.IMG_SIZE == 2 * self.opts.load_size or self.opts.data_down_sample: # incase the pixel pos -> 0.5
                K[:2,:] *= 0.5;

            Ks_list.append(K); RTs_list.append(RT);
            # mask, low resolution.
            mask = transforms.Resize(self.IMG_SIZE, interpolation=InterpolationMode.NEAREST)(mask) # default as 512
            mask = transforms.ToTensor()(mask).float() # [1, H, W]
            
            # rgbd data, original resolution
            rgb = self.rgb_to_tensor(render)
            rgb = mask.expand_as(rgb) * rgb # [3, H, W]
            rgbs_list.append(rgb)
            mask[mask <= 0.5] = 0; mask[mask > 0.5] = 1;
            masks_list.append(mask)
            
            if self.opts.data_down_sample: # down-sampling the gt-depth's data;
                gt_depth = cv2.resize(depth, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_NEAREST) # [1, H, W]
            else:
                gt_depth = depth

            gt_depth = torch.Tensor(gt_depth).float()[None, ...] # [1, H, W]            
            gt_depth = mask.expand_as(gt_depth) * gt_depth # [1, H, W]
            gt_depths_list.append(gt_depth)
            
            if aug_depths and np.random.rand() > 0.1: # apply on numpy data, blur and gaussian and eroded noise, 90% add noises here.
                # step1. depth to 512 resolution.
                if self.IMG_SIZE == 2 * self.opts.load_size or self.opts.data_down_sample: # aug depths (with resolution 512)
                    depth = cv2.resize(depth, (self.opts.load_size, self.opts.load_size), interpolation=cv2.INTER_NEAREST)
                    mask  = cv2.resize(mask[0].numpy(), (self.opts.load_size, self.opts.load_size), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = mask[0].numpy()

                # only when 
                if np.random.rand() > 0.5: # random noises (blurring & gaussian noises).
                    aug_depth = self.augment_depth_maps(depth, mask)
                else: # kinect pattern's noises here, using c_libs
                    aug_depth = self.augment_depth_maps_new(depth, mask, K)
                
                aug_depth = mask * aug_depth # [1, H, W]
                # resize to 1K.
                aug_depth = cv2.resize(aug_depth, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                aug_depth = torch.Tensor(aug_depth).float()[None, ...] # [1, H, W]
                depths_list.append(aug_depth)
            else:
                depths_list.append(gt_depth)
        
        
        return rgbs_list, gt_depths_list, depths_list, masks_list, Ks_list, RTs_list
    
    def load_data(self, subject, num_views, target_num_views, yid, random_sample=False): # loading the RGBD images         
        # 1. get inputs' view data.
        # randomly select N views' input rgbd, and N views' target rgbd.
        if random_sample or len(self.yaw_list) == 8: # randomly choice N ids.
            # view_ids = np.random.choice( self.yaw_list, num_views, replace=False )
            # self.real_order = [0,1,7,3,4,5,6,2] # this order
            num_cameras = len(self.yaw_list) # default as 8, as the real rendering setting.
            main_vid = np.random.choice(self.yaw_list, 1, replace=False)[0] # get the main id.
            view_ids = [
                self.real_order[main_vid], 
                self.real_order[(main_vid + 2) % num_cameras], 
                self.real_order[(main_vid - 2) % num_cameras], 
                self.real_order[(main_vid + 4) % num_cameras]
            ]
        else: # randomly noise: 30 degree.
            start_id = self.yaw_list[yid];
            view_interval = 360 // num_views # default as 120 degree.
            view_ids = [
                ( start_id + offset * view_interval + np.random.choice(np.arange(-15, 15)) ) % 360 
                for offset in range(num_views)
            ]
        
        # 2.get RGBD images. aug_depths is set to False for ground-truth mesh' based rendering.
        self.rgbs_list, self.depths_gt_list, self.depths_list, self.masks_list, self.Ks_list, self.RTs_list = \
            self.get_RGBD_cam_data(subject, view_ids, aug_depths=self.opts.aug_depths, aug_cam=self.opts.aug_cam, aug_img=False)

        target_obj_path = os.path.join(self.OBJ, subject, subject+'.obj')
        vertices = self.process_vertices_data(target_obj_path) # [MAX_VERTICES, 3]
        
        basic_info = {
            'target_verts':  vertices, # [N_max_points, 3]
            'rgbs':   torch.stack(self.rgbs_list, dim=0), # [Nv, 3, h, w]
            'depths': torch.stack(self.depths_list, dim=0), # [Nv, 1, h, w]
            'masks':  torch.stack(self.masks_list, dim=0), # [Nv, 1, h, w]
            'gt_depths': torch.stack( self.depths_gt_list, dim=0 ), # [Nv,1,h,w] ground-truth depths.
            # cameras.
            'ks':  torch.stack(self.Ks_list, dim=0), # [Nv, 3, 3]
            'rts': torch.stack(self.RTs_list, dim=0), # [Nv, 3, 4]
            # options.
            'source_view_ids': torch.as_tensor(np.array(view_ids)) # [id0, id1, id2]
        }

        return basic_info
        
    def get_item(self, idx):
        # get idxs.
        sid = idx % len(self.subjects)
        tmp = idx // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)
        
        subject = self.subjects[sid]
        res = {
            'name': subject,
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'render_data': True
        }
        
        render_data = self.load_data(subject, self.num_views, self.target_num_views, yid)
        # update the rendering_data.
        res.update(render_data)
        return res

    def __getitem__(self, idx):
        return self.get_item(idx)

if __name__ == '__main__':
    from options.RenTrainOptions import RenTrainOptions
    
    import matplotlib.pyplot as plt
    
    opts = RenTrainOptions().parse()
    opts.num_views = 4
    train_dataset = RenTrainDataloader(opts, phase='training')

    for data in train_dataset:
        names  = data['name']
        depths = data['depths']
        gt_depths = data['gt_depths']
        rgbs   = data['rgbs']
        masks  = data['masks']
        verts  = data['target_verts']
        # print(depths.shape,rgbs.shape,masks.shape,ks.shape, rts.shape,names)

        from utils_render.util import save_points
        save_points('/home/yons/my_Rendering/SAILOR/checkpoints_rend/Rend_v2_27', verts[0].permute(1,0))
        print(verts.shape)
        plt.imshow(depths[0,0,0].cpu().numpy() * 10000)
        plt.show()
        plt.imshow(gt_depths[0,0,0].cpu().numpy() * 10000)
        plt.show()

        plt.imshow(masks[0,0,0].cpu().numpy() * 10000)
        plt.show()
        
        # plt.imshow(rgbs[0,0].cpu().numpy())
        # plt.show()
        
        plt.imshow(rgbs[0,0].cpu().numpy().transpose(1,2,0))
        plt.show()
        # print(ks, rts)
        
        # exit()