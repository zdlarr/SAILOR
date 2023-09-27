"""
    The SRONet Test Dataset;
"""

import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
# for rendering mesh.
from c_lib.RenderUtil.dist import RenderUtils
import open3d as o3d
from SRONet.dataset.TrainDataset import RenTrainDataset
from PIL import Image
from torchvision.transforms import InterpolationMode

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

"""
    The RenderTest Dataloader.
    1. provided RGBD images.
"""

class RenTestDataloader(object):
    
    def __init__(self, opts, phase='testing'):
        super(RenTestDataloader, self).__init__()
        self.dataset = RenTestDataset(opts, phase)
    
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
        self.target_num_views = opts.target_num_views # default as 1;
        
        # get data dirs.
        if opts.real_data:
            self.RGB   = os.path.join(self.root_dir, 'COLOR')
            self.MASK  = os.path.join(self.root_dir, 'MASK')
            self.DEPTH = os.path.join(self.root_dir, 'DEPTH')
            self.PARAM = os.path.join(self.root_dir, 'PARAM')
        else: # not real captured data.
            self.RGB   = os.path.join(self.root_dir, 'RENDER')
        
        # other properties. e.g., default projective type: perspective.
        self.projection_mode = opts.project_mode if opts.project_mode is not None else 'perspective' # else ortho
        
        # default resolution, i.e., 512;
        self.IMG_SIZE = opts.img_size if opts.img_size else 512
        
        # the images' rendering list.
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
        # volume fusion scheme.
        self.volume_fusion = o3d.integration.ScalableTSDFVolume(
            voxel_length= 1.8 / 800, # suppose the box.1.8m
            sdf_trunc=0.01, # the truncate-psdf value. 1cm.
            color_type=o3d.integration.TSDFVolumeColorType.RGB8
        )
        
        self.FOCAL_X = opts.focal_x
    
    def get_subjects(self):
        if self.opts.test_subject is None and self.opts.real_data: # testing on real data.
            all_subjects = sorted( list(os.listdir(self.RGB)) )
        else:
            all_subjects = [self.opts.test_subject]

        return all_subjects
    
    def __len__(self):
        # for each subject, rendering along raw axis.
        if self.opts.rendering_static:
            return len(self.subjects) * len(self.yaw_list)
        else:
            return len(self.subjects)
    
    ########################  preprocessing, fusing mult-view depths. ########################
    def render_mesh(self, verts, faces, K, RT, tar_size=512):
        num_verts, num_faces = verts.shape[0], faces.shape[0]
        tri_uvs = np.ones([num_faces, 3, 2], dtype=np.float32) * 0.5
        tri_normals = np.ones([num_faces, 3, 3], dtype=np.float32)
        tex = np.ones([512, 512, 3], dtype=np.float32) * 0.5

        # to torch.
        vert_tr   = torch.from_numpy(verts[None]).float().cuda()
        face_tr   = torch.from_numpy(faces[None]).int().cuda()
        normal_tr = torch.from_numpy(tri_normals[None]).float().cuda()
        uv_tr     = torch.from_numpy(tri_uvs[None]).float().cuda()
        tex_tr    = torch.from_numpy(tex[None]).float().cuda()

        # output tensor.
        depth = torch.ones([1, tar_size, tar_size]).float().cuda() * 1000.
        rgb   = torch.zeros([1, tar_size, tar_size, 4]).float().cuda()
        mask  = torch.zeros([1, tar_size, tar_size]).int().cuda()

        cam_pos = (- RT[:3, :3].T @ RT[:3, -1:])[:, 0] # [3].
        K_, RT_ = torch.from_numpy(K[None]).float().cuda(), torch.from_numpy(RT[None]).float().cuda()
        light_dir = torch.from_numpy(cam_pos[None]).float().cuda(); 
        view_dir  = light_dir.clone()

        RenderUtils.render_mesh(vert_tr, face_tr, normal_tr, uv_tr, tex_tr,
                                K_, RT_, tar_size, tar_size,
                                0.3, 0.7, view_dir, light_dir, False, False,
                                depth, rgb, mask)

        rgb[..., -1] += (1-mask)
        rgb[..., :3] /= rgb[..., -1:]
        rgb *= mask[..., None]
        depth *= 1000 # to mm.
        depth *= mask # weighted by mask.
        
        return depth[0].cpu().numpy(), mask[0].cpu().numpy()
    
    def fuse_depth_maps(self, rgbs, depths, masks, Ks, RTs):
        # fuse the three-view's depths map to mesh, then project to depth maps.
        self.volume_fusion.reset()
        K_new = np.array([[self.FOCAL_X, 0.0, self.IMG_SIZE / 2.0 - 0.5], 
                          [0.0, self.FOCAL_X, self.IMG_SIZE / 2.0 - 0.5], 
                          [0.0, 0.0, 1.0]])
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.IMG_SIZE, self.IMG_SIZE, 
                                                      self.FOCAL_X, self.FOCAL_X, 
                                                      self.IMG_SIZE / 2.0 - 0.5, self.IMG_SIZE / 2.0 - 0.5)

        for i in range(len(depths)):
            rgb   = rgbs[i].permute(1, 2, 0).contiguous().numpy() * 255
            depth = depths[i][0].numpy()
            mask  = masks[i].numpy()
            RT    = np.eye(4)
            RT_   = RTs[i].contiguous().numpy() # [3, 4]
            K     = Ks[i].contiguous().numpy() # [3,3]

            # K_new @ res_mat @ RT
            res_mat = np.linalg.inv(K_new) @ K # [3,3]
            RT_new = res_mat @ RT_ # [3, 4]
            RT[:3, :] = RT_new

            color = o3d.geometry.Image(rgb.astype(np.uint8))
            depth = o3d.geometry.Image(depth.astype(np.float32))
            rgbd  = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1,
                                                                       depth_trunc=self.opts.z_size + self.opts.z_bbox_len, convert_rgb_to_intensity=False)
            self.volume_fusion.integrate(rgbd, intrinsic, RT.astype(np.float64))

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(self.volume_fusion.extract_triangle_mesh())
        # vis.run()
        # vis.destroy_window()

        mesh  = self.volume_fusion.extract_triangle_mesh()
        verts =  np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        target_center = verts.mean(axis=0)
        
        depth_new_list = []
        for i in range(len(depths)):
            K = Ks[i].contiguous().numpy() # [3,3]
            RT = RTs[i].contiguous().numpy() # [3, 4]
            mask = masks[i]
            depth_rend, _ = self.render_mesh(verts, faces, K, RT, tar_size=self.IMG_SIZE)
            depth_rend = torch.Tensor(depth_rend).float()[None, ...] / 1000.0 # to m unit.
            depth_rend = mask.expand_as(depth_rend) * depth_rend
            depth_new_list.append(depth_rend) # [1, h, w]
            
        return depth_new_list, target_center
    ######################### end ############################
    
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
            if depth.shape[0] == 2 * self.opts.img_size:
                K[:2,:] *= 0.5

            self._Ks_list.append(K); self._RTs_list.append(RT)
            # to target size (mask, rgb, depths.)
            mask = transforms.Resize(self.IMG_SIZE, interpolation=InterpolationMode.BILINEAR)(mask) # default as 512
            mask = transforms.ToTensor()(mask).float() # [1, H, W]
            mask[mask < 0.5] = 0; mask[mask >= 0.5] = 1;
            masks_list.append(mask)
            
            # rgbd data.
            rgb = self.rgb_to_tensor(render)
            rgb = mask.expand_as(rgb) * rgb # [3, H, W]
            rgbs_list.append(rgb)
            
            if not self.opts.real_data: # not using real data.
                depth = cv2.resize(depth, (self.opts.img_size, self.opts.load_size), interpolation=cv2.INTER_NEAREST)
                depth = self.augment_depth_maps_all(depth, mask[0].numpy(), K)

            # if depth's resolution is smaller than the original images, then resize here;
            depth = cv2.resize(depth, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            depth = torch.Tensor(depth).float()[None, ...] # [1, H, W]
            depth = mask.expand_as(depth) * depth
            depths_list.append(depth)
            
        # fuse multi-view depth data, re-render the depth maps.
        if self.opts.pre_fuse_depths: # pre-fusing the depth images.
            depth_new_list, self._target_center = self.fuse_depth_maps(rgbs_list, depths_list, masks_list, self._Ks_list, self._RTs_list)
            depth_new_list = depth_new_list[:self.num_views]
        else:            
            depth_new_list = depths_list[:self.num_views]
            
        rgbs_list      = rgbs_list[:self.num_views]
        masks_list     = masks_list[:self.num_views]
                
        RTs_list       = self._RTs_list[:self.num_views]
        Ks_list        = self._Ks_list[:self.num_views]
       
        return {
            'rgbs':   torch.stack(rgbs_list, dim=0), # [Nv, 3, h, w]
            'depths': torch.stack(depth_new_list, dim=0), # [Nv, 1, h, w]
            'masks':  torch.stack(masks_list, dim=0), # [Nv, 1, h, w]
            # cameras.
            'ks':  torch.stack(Ks_list, dim=0), # [Nv, 3, 3]
            'rts': torch.stack(RTs_list, dim=0), # [Nv, 3, 4]
            # options.
            'source_view_ids': torch.as_tensor(np.array(view_ids)[:self.num_views])
        }
        
    def generate_cam_Rt(self, center, direction, right, up):
        def normalize_vector(v):
            v_norm = np.linalg.norm(v)
            return v if v_norm == 0 else v / v_norm

        center = center.reshape([-1])
        direction = direction.reshape([-1])
        right = right.reshape([-1])
        up = up.reshape([-1])

        rot_mat = np.eye(3)
        s = right
        s = normalize_vector(s)
        rot_mat[0, :] = s
        u = up
        u = normalize_vector(u)
        rot_mat[1, :] = -u
        rot_mat[2, :] = normalize_vector(direction)
        trans = -np.dot(rot_mat, center) # x = R X + t, C = -R^T * t -> t = - R * C
        return rot_mat, trans
        
    def generate_cams(self, pitch, yaw, d, up=np.array([0.0, 1.0, 0.0]), target=[0,0,0]):
        # suppose pitch located in (-90, 90),  yaw located in (0, 360).
        angle_xz  = (np.pi / 180) * (yaw % 360)
        if pitch > 0:
            angle_y = (np.pi / 180) * pitch if pitch <= 90 else (np.pi / 180) * 90
        else:
            angle_y = (np.pi / 180) * pitch if pitch >= -90 else (np.pi / 180) * (-90)

        eye = np.asarray([d * np.cos(angle_y) * np.sin(angle_xz),
                          d * np.sin(angle_y),
                          d * np.cos(angle_y) * np.cos(angle_xz)]) + np.array(target, np.float64)

        up /= np.linalg.norm(up)

        fwd = np.asarray(target, np.float64) - eye
        fwd /= np.linalg.norm(fwd) # [1,3]
        
        left = np.cross(up, fwd)
        left /= np.linalg.norm(left)

        right = -left
        right /= np.linalg.norm(right)

        cam_R, cam_t = self.generate_cam_Rt(eye, fwd, right, up)

        return cam_R, cam_t, -fwd
        
    def preprocess_cam_data(self):
        # initialized intrinsic matrix.
        self._K = torch.tensor([[self.FOCAL_X, 0.0, self.IMG_SIZE / 2.0], 
                                [0.0, self.FOCAL_X, self.IMG_SIZE / 2.0], 
                                [0.0, 0.0, 1.0]], dtype=torch.float32)
        K_INV = torch.inverse(self._K)
        
        theta = []; cam_center = []
        # the ordered view lists.
        views_list = [0,2,3,1]
        for id, i in enumerate(views_list):
            # get the intrinsic and extrinsic matrix.
            k, rt = self._Ks_list[i], self._RTs_list[i]

            res_mat = K_INV @ k # [3,3]
            rt_ = res_mat @ rt # [3, 4]

            cam_pos = (- rt_[:, :3].T @ rt_[:3, -1:])[:, 0] # [3].
            
            cam_center.append( cam_pos )
            theta.append( id * (360 / 4) * np.pi / 180.0 )

        theta.append(2 * np.pi)
        cam_center.append(cam_center[0])
        thetas = np.array(theta).astype(np.float32)
        cam_datas = np.stack(cam_center, axis=0).astype(np.float32)
        self._cs = CubicSpline(thetas, cam_datas, bc_type='periodic')

        # xs = 2 * np.pi * np.linspace(0, 1, 100)
        # fig, ax = plt.subplots(figsize=(6.5, 4))
        # ax.plot(self._cs(xs)[:, 0], self._cs(xs)[:, 2], label='spline')
        # ax.axes.set_aspect('equal')
        # ax.legend(loc='center')
        # plt.show()

    def preprocess_cam_data_new(self):
        self._K = torch.tensor([[self.FOCAL_X, 0.0, self.IMG_SIZE / 2.0], 
                                [0.0, self.FOCAL_X, self.IMG_SIZE / 2.0], 
                                [0.0, 0.0, 1.0]], dtype=torch.float32)
        K_INV = torch.inverse(self._K)
        
        theta = []; cam_data = []
        # the ordered view lists.
        views_list = [0,2,3,1]
        for id, i in enumerate(views_list):
            # get the intrinsic and extrinsic matrix.
            k, rt = self._Ks_list[i], self._RTs_list[i]

            res_mat = K_INV @ k # [3,3]
            rt_ = res_mat @ rt # [3, 4]

            rt = rt_.view(-1).numpy() # [12]
            
            cam_data.append( np.concatenate( [rt],axis=0 ) ) # only interplation the cam extrinsic.
            theta.append( id * (360 / 4) * np.pi / 180.0 )

        theta.append(2 * np.pi)
        cam_data.append(cam_data[0])
        thetas = np.array(theta).astype(np.float32)
        cam_datas = np.stack(cam_data, axis=0).astype(np.float32)
        self._cs = CubicSpline(thetas, cam_datas, bc_type='periodic')

    def gen_spline_camera_data_new(self, vid):
        vid %= 360
        rt_new = self._cs( vid * np.pi / 180.0 )
        rt_new = torch.as_tensor(rt_new).view(3,4)
        
        tar_K  = self._K[None].float()
        tar_RT = rt_new[None].float()
         
        return {
            'target_ks':  tar_K, # [Nv, 3, 3]
            'target_rts': tar_RT, # [Nv, 3, 4]
            'target_view_ids': vid
        }
        
    def gen_spline_camera_data(self, vid):
        vid %= 360
        cam_center_new = self._cs( vid * np.pi / 180.0 )
        cam_target_distance = np.linalg.norm(cam_center_new - self._target_center)
        
        t_R, t_T, _ = self.generate_cams(0, 360 - vid, cam_target_distance, target=self._target_center)
        
        tar_K  = self._K[None].float()
        tar_RT = np.zeros([3,4])
        tar_RT[:3,:3] = t_R; tar_RT[:3,-1] = t_T;
        tar_RT = torch.as_tensor(tar_RT[None], dtype=torch.float32)
        
        return {
            'target_ks':  tar_K, # [Nv, 3, 3]
            'target_rts': tar_RT, # [Nv, 3, 4]
            'target_view_ids': vid
        }
    
    def get_item(self, idx):
        if self.opts.rendering_static: # order by the view ids.
            vid = idx % len(self.yaw_list)
            tmp = idx // len(self.yaw_list)
            sid = tmp % len(self.subjects)
        else: # order by the subject ids.
            sid = idx % len(self.subjects)
            tmp = idx // len(self.subjects)
            vid = tmp % len(self.yaw_list)
        
        subject = self.subjects[sid]
        
        res = {
            'name': subject,
            'sid': sid
        }
        # 1. get input n views' data.
        render_data = self.get_RGBD_cam_data(subject)
        res.update(render_data)

        # 2. get the target camera data (K & RTs.)
        if self.opts.rendering_static:
            if vid == 0:
                self.preprocess_cam_data_new() # preprocess camera data;

            target_vid = self.yaw_list[vid];
            target_cams = self.gen_spline_camera_data_new(target_vid)
        else:
            if sid == 0 and vid == 0: # only when first, process the frames;
                self.preprocess_cam_data_new() # preprocess camera data;

            target_vid = 360.0 * sid / len(self.subjects);
            target_cams = self.gen_spline_camera_data_new(target_vid)
            
        res.update(target_cams)
        
        return res

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    
if __name__ == '__main__':
    from SRONet.options.RenTestOptions import RenTestOptions
    
    import matplotlib.pyplot as plt
    
    opts = RenTestOptions().parse()
    test_dataset = RenTestDataloader(opts, phase='testing')

    for data in test_dataset:
        names = data['name']
        depths = data['depths']
        rgbs = data['rgbs']
        
        plt.imshow(rgbs[0,0].cpu().numpy().transpose(1,2,0))
        plt.show()
        plt.imshow(rgbs[0,1].cpu().numpy().transpose(1,2,0))
        plt.show()
        plt.imshow(rgbs[0,2].cpu().numpy().transpose(1,2,0))
        plt.show()
        plt.imshow(rgbs[0,3].cpu().numpy().transpose(1,2,0))
        plt.show()

        plt.imshow(depths[0,0,0].cpu().numpy() * 10000)
        plt.show()
        plt.imshow(depths[0,1,0].cpu().numpy() * 10000)
        plt.show()
        plt.imshow(depths[0,2,0].cpu().numpy() * 10000)
        plt.show()
        plt.imshow(depths[0,3,0].cpu().numpy() * 10000)
        plt.show()
    
        
        