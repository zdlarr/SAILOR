import sys, os
import cv2
import numpy as np

import torch
import torch.nn as nn
sys.path.append(os.path.join(__file__, '../dist'))
import RenderUtils
import trimesh
from .render_data import generate_cam_Rt

class MeshRender(nn.Module):

    def __init__(self, batch_size=1, resolution=512, max_depth=100, 
                 ambient=0.3, light_stren=0.8, calc_spec=True, is_ortho=True, depth_normalize=False):
        super(MeshRender, self).__init__()
        assert torch.cuda.is_available(), 'cuda is not available.'
        self.batch_size = batch_size
        self.resolution = resolution
        self.ambient = ambient
        self.light_stren = light_stren
        self.render_spec = calc_spec
        self.proj_ortho = is_ortho
        self.max_depth = max_depth
        self.depth_normalize = depth_normalize

        # rgb, depths, masks,
        RGBs = torch.cuda.FloatTensor(self.batch_size, resolution, resolution, 4).fill_(0)
        depths = torch.cuda.FloatTensor(self.batch_size, resolution, resolution).fill_(max_depth)
        masks = torch.zeros_like(depths).int()
        self.register_buffer('RGBs', RGBs)
        self.register_buffer('depths', depths)
        self.register_buffer('masks', masks)
    
    def forward(self, properties):
        """
            verts: [B, N, 3].
            faces: [B, M, 3].
            normals: [B, M, 3, 3]
            uvs: [B, M, 3, 2].
            textures: [B, H, W, 3].
            Ks: [B, 3, 3].
            RTs: [B, 3, 4].
            view_dir: [B,3].
            light_dir: [B,3].
        """
        for x in properties:
            self.check_batch(x)
            self.check_input(x)
        
        self.clear_buffer()
        vs, fs, ns, uvs, texs, Ks, RTs, vd, ld = properties

        RenderUtils.render_mesh(vs, fs, ns, uvs, texs, Ks, RTs, 
                                self.resolution, self.resolution,
                                self.ambient, self.light_stren, 
                                vd, ld, self.render_spec, self.proj_ortho,
                                self.depths, self.RGBs, self.masks)
        
        # rgbs = self.RGBs[..., :3] / self.RGBs[..., -1:]
        self.RGBs[..., -1] += (1 - self.masks)
        rgbs = self.RGBs[..., :3] / self.RGBs[..., -1:]
        if self.depth_normalize:
            self.depths = (self.depths - torch.min(self.depths)) \
                        / (torch.max(self.depths) - torch.min(self.depths))

        return self.depths, rgbs, self.masks

    def clear_buffer(self):
        self.RGBs.fill_(0)
        self.depths.fill_(self.max_depth)
        self.masks.fill_(0)
    
    def check_input(self, x):
        if x.device == 'cpu':
            raise TypeError('Only support cuda tensors.')

    def check_batch(self, x):
        if x.shape[0] != self.batch_size:
            raise IndexError('Error batch size.')


class TorchObjLoader(object):
    """
        Load object with textures, and transform the items to torch tensor.
    """
    def __init__(self, batch_size=1):
        super(TorchObjLoader, self).__init__()
        assert torch.cuda.is_available(), 'TorchObject Loader only allow cuda tensor now.'
        self.__batch_size = batch_size
        self.__tex_img = None
        self.__mesh = None
        self.__tri_normals = None
        self.__tri_uvs = None
        self.__vs = None
        self.__fs = None
        self.__num_verts = 0
        self.__num_faces = 0

    def load(self, obj_path, tex_path=None):
        if tex_path is not None:
            self.__tex_img = cv2.imread(tex_path).astype(np.float32)[:,:, ::-1] / 255.0 # BGR2RGB
        # loading mesh.
        self.__mesh = trimesh.load(obj_path)
        self.__vs = self.__mesh.vertices
        self.__fs = self.__mesh.faces
        self.__uvs = self.__mesh.visual.uv
        self.__num_faces = self.__fs.shape[0]
        self.__num_verts = self.__vs.shape[0]
        # uvs.
        self.__tri_uvs = self.__uvs[self.__fs].reshape(self.__num_faces, 3, 2)
        self.__nols = self.__mesh.vertex_normals
        self.__tri_normals = self.__nols[self.__fs].reshape(self.__num_faces, 3, 3)
        return 

    @property
    def vertices(self):
        # with shape: [1, N, 3]
        return torch.from_numpy(np.stack([self.__vs]*self.__batch_size, axis=0)).float().cuda()

    @property
    def faces(self):
        # [1, M, 3] of int types.
        return torch.from_numpy(np.stack([self.__fs]*self.__batch_size, axis=0)).int().cuda()

    @property
    def normals(self):
        # [1, M, 3, 3] of float type.
        return torch.from_numpy(np.stack([self.__tri_normals]*self.__batch_size, axis=0)).float().cuda()

    @property
    def uvs(self):
        # [1, M, 3, 2] of float type.
        return torch.from_numpy(np.stack([self.__tri_uvs]*self.__batch_size, axis=0)).float().cuda()

    @property
    def texs(self):
        # [1, H, W, 3]
        if self.__tex_img is not None:
            return torch.from_numpy(np.stack([self.__tex_img]*self.__batch_size, axis=0)).float().cuda()
        return None
    
    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def n_verts(self):
        return self.__num_verts

    @property
    def n_faces(self):
        return self.__num_faces

def generate_sphere_cam(pitch, yaw, target=[0,0,0], dist=10, focal=5000, resolution=512):
    """
        Given yaw & pitch & dist value, generate a cam in the sphere surface.
        return: cam, view_dir.
    """
    # yaw, locate in [0, 360]
    # pitch, locate in [-90, 90]
    angle_yaw  = (np.pi / 180) * (yaw % 360)
    angle_pitch = (np.pi / 180) * pitch

    eye = np.asarray([dist * np.cos(angle_pitch) * np.sin(angle_yaw),
                      dist * np.sin(angle_pitch),
                      dist * np.cos(angle_pitch) * np.cos(angle_yaw)])
    
    left = np.cross([0, 1, 0], -eye)
    up = np.cross(-eye, left)
    up /= np.linalg.norm(up)

    fwd = np.asarray(target, np.float64) - eye
    fwd /= np.linalg.norm(fwd)

    right = -left
    right /= np.linalg.norm(right)

    RT = np.zeros([3,4])
    cam_R, cam_t = generate_cam_Rt(eye, fwd, right, up)
    RT[:3, :3] = cam_R; RT[:3, -1] = cam_t

    # intrinsic mat.
    K = np.eye(3)
    K[0, 0] = focal; K[1, 1] = focal
    K[0, 2] = resolution / 2; K[1, 2] = resolution / 2
    
    cam = {'RT': RT, 'K': K}
    
    return cam, -fwd

def test(render, obj_path, tex_path, cam_path):
    from render_utils import load_parameters
    import matplotlib.pyplot as plt
    vertices, faces, normals, uvs, texs, Ks, RTs, ambient, light_stren,  view_dirs, light_dirs, \
           depth, RGBs, masks, num_cams = load_parameters(obj_path, tex_path, cam_path)

    depths, rgbs, _ = render([vertices, faces, normals, uvs, texs, Ks, RTs, view_dirs, light_dirs])
    plt.imshow(rgbs[0].cpu().numpy())
    plt.show()

if __name__ == '__main__':
    render = MeshRender(max_depth=13, resolution=1024, depth_normalize=True)
    obj_path = "./render_people_dataset/Males_Vis_Models/Male_011/Male_011.obj"
    tex_path = "./render_people_dataset/Males_Vis_Models/Male_011/tex/Male_011.jpg"
    cam_path = "./render_people_dataset/image_data/Male_011/meta/cam_data.mat"

    obj_loader = TorchObjLoader(batch_size=10)
    obj_loader.load(obj_path, tex_path)
    test(render, obj_path, tex_path, cam_path)
