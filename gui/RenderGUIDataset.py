"""
    The Rendering GUI Dataset;
"""


import sys, os
from PIL import Image, ImageOps
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import cv2
import numpy as np

from upsampling.dataset.TrainDataset import RenTrainDataset

"""
    The GUI RenderTest Dataloader.
    1. provided RGBD images.
"""

class RenTestDataloader(object):
    
    def __init__(self, opts, phase='testing'):
        super(RenTestDataloader, self).__init__()
        self.dataset = RenTestDataset(opts, phase)
        # logging.basicConfig(level=logging.INFO if torch.distributed.get_rank() in [-1, 0] else logging.WARN)
        print('Dataset [%s] is creating ...' % opts.ren_data_root)
        
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(opts.num_threads), # as 0;
            pin_memory=opts.pin_memory
        )
        
    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for data in self.dataloader:
            yield data
    
    def get_iter(self):
        return iter(self)
    

class RenTestDataset(RenTrainDataset):
    
    def __init__(self, opts, phase='testing'):
        self.opts     = opts
        self.is_train = opts.is_train
        self._phase   = phase

        self.num_views = opts.num_views
        self.root_dir  = opts.ren_data_root
        self.target_num_views = opts.target_num_views
        
        # get data dirs.
        self.RGB   = os.path.join(self.root_dir, 'COLOR') # color dir.
        self.MASK  = os.path.join(self.root_dir, 'MASK')
        self.DEPTH = os.path.join(self.root_dir, 'DEPTH')
        self.PARAM = os.path.join(self.root_dir, 'PARAM')
        
        # other properties. e.g., default projective type: perspective.
        self.projection_mode = opts.project_mode if opts.project_mode is not None else 'perspective' # else ortho
        
        # default resolution, i.e., 512;
        self.IMG_SIZE = opts.img_size if opts.img_size else 512
        
        # rendering 120 degrees' views.
        self.yaw_list = list(range(0, 360, opts.rendering_seperate_angle))
        self.input_view_ids = opts.test_view_ids
        
        self.pitch_list = [0]
        
        # get the objects.
        self.subjects = self.get_subjects()
        
        # PIL to Image.
        self.rgb_to_tensor = transforms.Compose([
            transforms.Resize([self.IMG_SIZE, self.IMG_SIZE]),
            transforms.ToTensor()
        ])
    
    def get_subjects(self):
        if self.opts.test_subject is None and self.opts.real_data: # testing on real data.
            all_subjects = sorted( list(os.listdir(self.RGB)) )
        else:
            all_subjects = [self.opts.test_subject]

        return all_subjects
    
    def __len__(self):
        return len(self.subjects)

    def get_RGBD_cam_data(self, subject): # get the input N views' data.
        view_ids = self.input_view_ids
        rgbs_list = []; depths_list = []; masks_list = [];
        self._Ks_list = []; self._RTs_list = []; # used when obtaining target camera parameters.
        pid = self.pitch_list[0] # pid: 0
        
        for i, vid in enumerate(view_ids): # get the target data_path.
            rgb_path   = os.path.join(self.RGB,   subject, '%d.jpg' % (vid) if self.opts.real_data else '%d_%d.jpg' % (vid, pid))
            mask_path  = os.path.join(self.MASK,  subject, '%d.png' % (vid) if self.opts.real_data else '%d_%d.png' % (vid, pid))
            depth_path = os.path.join(self.DEPTH, subject, '%d.png' % (vid) if self.opts.real_data else '%d_%d.png' % (vid, pid))
            param_path = os.path.join(self.PARAM, subject, '%d.npy' % (vid) if self.opts.real_data else '%d_%d.npy' % (vid, pid))

            render = Image.open(rgb_path).convert('RGB')
            mask   = Image.open(mask_path).convert('L')
            depth  = cv2.imread(depth_path, -1) / 10000 # to m unit.
            cam_param = np.load(param_path, allow_pickle=True) # K, RTS
            
            K, RT = cam_param.item().get('K'), cam_param.item().get('RT')
            
            K  = torch.as_tensor(K, dtype=torch.float32)
            RT = torch.as_tensor(RT, dtype=torch.float32)
            self._Ks_list.append(K); self._RTs_list.append(RT);
            # resize the image's 
            mask = transforms.Resize(self.IMG_SIZE, interpolation=InterpolationMode.BILINEAR)(mask) # default as 512
            mask = transforms.ToTensor()(mask).float() # [1, H, W]
            mask[mask < 0.5] = 0; mask[mask >= 0.5] = 1;
            masks_list.append(mask)
            
            # rgbd data.
            rgb = self.rgb_to_tensor(render)
            rgb = mask.expand_as(rgb) * rgb # [3, H, W]
            rgbs_list.append(rgb)
            
            # if depth's resolution is smaller than the original images, then resize here;
            depth = cv2.resize(depth, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            depth = torch.Tensor(depth).float()[None, ...] # [1, H, W]
            depth = mask.expand_as(depth) * depth
            depths_list.append(depth)
            
        # fuse multi-view depth data, re-render the depth maps.
        depth_new_list = depths_list[:self.num_views]
        rgbs_list      = rgbs_list[:self.num_views]
        masks_list     = masks_list[:self.num_views]
        
        # when input size are largers, resize the matrix K;
        if self.IMG_SIZE == 2 * self.opts.load_size: # incase the pixel pos -> 0.5
            for i, k in enumerate(self._Ks_list): # this time.
                self._Ks_list[i][:2] *= 0.5
                
        RTs_list       = self._RTs_list[:self.num_views]
        Ks_list        = self._Ks_list[:self.num_views]
        self.Ks_list = Ks_list; self.RTs_list = RTs_list;
       
        return {
            'rgbs':   torch.stack(rgbs_list, dim=0), # [Nv, 3, h, w]
            'depths': torch.stack(depth_new_list, dim=0), # [Nv, 1, h, w]
            'masks':  torch.stack(masks_list, dim=0), # [Nv, 1, h, w]
            # cameras.
            'ks':  torch.stack(Ks_list, dim=0), # [Nv, 3, 3]
            'rts': torch.stack(RTs_list, dim=0), # [Nv, 3, 4]
            # options.
            'source_view_ids': torch.as_tensor(np.array(view_ids)) # [id0, id1, id2]
        }
    
    def get_item(self, idx):
        # get idxs, first vid then sid;
        sid = idx % len(self.subjects) # the object idx.
        
        subject = self.subjects[sid]
        
        res = {
            'name': subject,
            'sid': sid,
            'dir': self.opts.ren_data_root
        }
        # 1. get input n views' data.
        render_data = self.get_RGBD_cam_data(subject)
        res.update(render_data)
        
        return res

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    
if __name__ == '__main__':
    from gui.RenTestOptions import RenTestOptions
    
    opts = RenTestOptions().parse()
    test_dataset = RenTestDataloader(opts, phase='testing')

    for data in test_dataset:
        names = data['name']
        rgbs  = data['rgbs']
        depths  = data['depths']
        masks = data['masks']
        print(rgbs.shape, depths.shape, masks.shape)

    
        
        