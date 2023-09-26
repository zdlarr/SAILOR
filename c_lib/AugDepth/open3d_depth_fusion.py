# by zheng dong

import os
import numpy as np
import cv2
import open3d as o3d


class O3DepthFusion(object):

    def __init__(self, voxel_len=1 / 256, sdf_trunc=0.04, depth_trunc=2.5, write_able=True, view_able=True):
        super(O3DepthFusion, self).__init__()
        
        # build volumes;
        self.volume = o3d.integration.ScalableTSDFVolume(
            voxel_length=voxel_len,
            sdf_trunc=sdf_trunc,
            color_type=o3d.integration.TSDFVolumeColorType.RGB8
        )
        self.depth_trunc = depth_trunc
        self.is_write = write_able
        self.is_view = view_able
    
    def fusion(self, rgbs, depths, RTs, Ks, out_pd=False, output_path='./tsdf_fusion_scene.ply'):
        # fusion from RGBs, depths, RTs, Ks
        num_views = rgbs.shape[0]
        
        for i in range(num_views):
            rgb = rgbs[i]
            depth = depths[i]
            RT = np.eye(4)
            RT[:3, :] = RTs[i]; K = Ks[i]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb.shape[1], rgb.shape[0], K[0,0], K[1,1], rgb.shape[1] // 2 - 0.5, rgb.shape[0] // 2 - 0.5)
            
            color = o3d.geometry.Image(rgb.astype(np.uint8))
            depth = o3d.geometry.Image(depth.astype(np.float32))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1,
                                                                      depth_trunc=self.depth_trunc, convert_rgb_to_intensity=False)

            self.volume.integrate(rgbd, intrinsic, RT.astype(np.float64))

        if not out_pd:
            mesh = self.volume.extract_triangle_mesh()
        else:
            mesh = self.volume.extract_point_cloud()

        # mesh.filter_smooth_laplacian(3)
        if self.is_view:
            o3d.visualization.draw_geometries([mesh])
        
        if self.is_write:
            if not out_pd:
                o3d.io.write_triangle_mesh(output_path, mesh)
            else:
                o3d.io.write_point_cloud(output_path, mesh)


if __name__ == '__main__':

    o3d_fusion = O3DepthFusion(voxel_len=1 / 512.0, sdf_trunc=0.01, depth_trunc=3.0, write_able=True, view_able=False)
    
    base_dir  = './test_data'
    rgbs_path = os.path.join(base_dir, 'color')
    depths_path = os.path.join(base_dir, 'depth')
    cam_params = os.path.join(base_dir, 'param')
    masks_path = os.path.join(base_dir, 'mask')
    depth_path = './noise_depth_0_0.png'

    output_path = './noised_pds.ply'

    rgb_datas = []; depth_datas = []; RTs = []; Ks = []

    for i, rgb in enumerate(sorted(os.listdir(rgbs_path))):
        id = rgb.split('.')[0]
        
        rgb_path = os.path.join(rgbs_path, rgb)
        cam_path = os.path.join(cam_params, id + '.npy')
        mask_path = os.path.join(masks_path, id + '.png')
        
        rgb_data = cv2.imread(rgb_path)[..., ::-1]
        mask_data = cv2.imread(mask_path)[..., 0] / 255.0
        mask_data = cv2.resize(mask_data, (512,512), interpolation=cv2.INTER_NEAREST)
        rgb_data  = cv2.resize(rgb_data, (512,512), interpolation=cv2.INTER_LINEAR)
        rgb_data  = np.ones([512,512,3])

        rgb_data[..., 0] = 150;
        rgb_data[..., 1] = 150;
        rgb_data[..., 2] = 200;
        
        depth_data = cv2.imread(depth_path, -1) / 10000
        depth_data = cv2.resize(depth_data, (512, 512), interpolation=cv2.INTER_NEAREST)
        depth_data *= np.array(mask_data)
        
        cam_data = np.load(cam_path, allow_pickle=True)
        K, RT = cam_data.item().get('K'), cam_data.item().get('RT')
        K[:2] /= 2.0
        new_K = np.array([[320, 0, 256],[0, 320, 256],[0,0,1]], dtype=np.float32)
        new_RT = (np.linalg.inv(new_K) @ K) @ RT

        rgb_datas.append(rgb_data); depth_datas.append(depth_data);
        RTs.append(new_RT); Ks.append(new_K);

    rgb_datas = np.stack(rgb_datas, axis=0).astype(np.float32)
    depth_datas = np.stack(depth_datas, axis=0).astype(np.float32)
    RTs = np.stack(RTs, axis=0).astype(np.float32)
    Ks = np.stack(Ks, axis=0).astype(np.float32)

    o3d_fusion.fusion(rgb_datas, depth_datas, RTs, Ks, True, output_path)
