"""
    Mesh Renderer using our render-utils.
"""

import trimesh
import numpy as np
import torch
import os, sys
import matplotlib.pyplot as plt
import cv2

from c_lib.RenderUtil.dist import RenderUtils
from upsampling.dataset.TestDataset import RenTestDataloader
from upsampling.options.RenTestOptions import RenTestOptions

opts = RenTestOptions().parse()
opts.rendering_static = True
opts.ren_data_root    = './test_data/static_data0'
# opts.ren_data_root    = './test_data/static_data1'

test_dataset = RenTestDataloader(opts, phase=opts.phase)

obj_path = './checkpoints_rend/SAILOR/val_results/FRAME0001_epoch_test.obj'
obj_name = 'FRAME0001_epoch_test'
output_path = os.path.join('./checkpoints_rend/SAILOR/', obj_name)

mesh = trimesh.load(obj_path)

# get center position.
verts = mesh.vertices
x_mid = np.mean(verts[:, 0])
y_mid = np.mean(verts[:, 1])
z_mid = np.mean(verts[:, 2])
print('the center positions of the verts.', x_mid, y_mid, z_mid)

if not os.path.exists(output_path):
    os.mkdir(output_path)

def render_mesh(mesh_path, K, RT, tar_size=(512, 512), device='cuda:0', move_center=False, target=[0,0,0], tex=None):
    mesh = trimesh.load(mesh_path)
    # load_properties. verts [N, 3]
    verts, faces, normals = mesh.vertices, mesh.faces, mesh.vertex_normals
    if move_center:
        verts -= np.array(target)[None, :]
        
    num_verts, num_faces = verts.shape[0], faces.shape[0] # [N_verts, 3], [N_faces, 3];
    tri_uvs = np.ones([num_faces, 3, 2]) * 0.5 # suppose all vertices located in [0.5, 0.5]
    tri_normals = normals[faces].reshape(num_faces, 3, 3)
    tex = np.ones([512, 512, 3], dtype=np.float32) * 1.0 # the color is white here.
    
    # to torch.
    vert_tr   = torch.from_numpy(verts[None]).float().to(device)
    face_tr   = torch.from_numpy(faces[None]).int().to(device)
    normal_tr = torch.from_numpy(tri_normals[None]).float().to(device)
    uv_tr     = torch.from_numpy(tri_uvs[None]).float().to(device)
    tex_tr    = torch.from_numpy(tex[None]).float().to(device)
    
    # output tensor.
    depth = torch.ones([1, tar_size[0], tar_size[1]]).float().to(device) * 1000.
    rgb   = torch.zeros([1, tar_size[0], tar_size[1], 4]).float().to(device)
    mask  = torch.zeros([1, tar_size[0], tar_size[1]]).int().to(device)
    
    cam_pos = (- RT[:3, :3].T @ RT[:3, -1:])[:, 0] # [3].
    K_, RT_ = torch.from_numpy(K[None]).float().to(device), torch.from_numpy(RT[None]).float().to(device)
    light_dir = torch.from_numpy(cam_pos[None]).float().to(device); 
    light_dir = light_dir / torch.norm(light_dir, dim=1)[:, None]
    view_dir  = light_dir.clone()
    
    # rendering operations.
    RenderUtils.render_mesh(vert_tr, face_tr, normal_tr, uv_tr, tex_tr,
                            K_, RT_, tar_size[0], tar_size[1],
                            0.4, 0.45, view_dir, light_dir, True, False,
                            depth, rgb, mask)
    rgb[..., -1] += (1-mask)
    rgb[..., :3] /= rgb[..., -1:]
    rgb *= mask[..., None]
    rgb += (1 - mask)[..., None] * 0.0
    depth *= 1000 # to mm.
    depth *= mask # weighted by mask.
    return rgb[0].cpu().numpy(), depth[0].cpu().numpy(), mask[0].cpu().numpy()

for idx, data in enumerate(test_dataset):
    tar_k = data['target_ks'][0,0]
    tar_rt = data['target_rts'][0,0]
    tar_k[:2] *= 2.0 # to original k.

    rgb_rend, depth_rend, mask_rend = render_mesh(obj_path, tar_k.numpy(), tar_rt.numpy(), tar_size=(1024, 1024), device='cuda:0', 
                                                    move_center=False, target=[x_mid, y_mid + 0.1, z_mid])
    # print(rgb_rend.shape)
    # plt.imshow(rgb_rend[..., :3])
    # plt.show()
    rgb_rend *= 255
    # exit()
    cv2.imwrite(os.path.join(output_path, obj_name + '_' + str(idx).zfill(5) + '.png'), rgb_rend[...,:3].astype(np.uint8))
    print('fin, name:{}, deg:{}'.format(obj_name, idx))

output_video_path = os.path.join(output_path, obj_name + '.avi')
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videowriter = cv2.VideoWriter(output_video_path, fourcc, 25, (1024, 1024), True)
frames = sorted(os.listdir(output_path))
for frame in frames:
    f_path = os.path.join(output_path, frame)
    image = cv2.imread(f_path)
    videowriter.write(image)
    print(frame + " has been written!")

videowriter.release()

        
