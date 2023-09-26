import sys, os

import numpy as np
import cv2
import torch
import trimesh
import math

sys.path.append(os.path.join(__file__, '../dist'))
import RenderUtils
import scipy.io as sio
import time
import matplotlib.pyplot as plt

from mesh import load_obj_mesh

def generate_cam_Rt(center, direction, right, up):
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

def load_cams(cam_path):
    cams_data = sio.loadmat(cam_path, verify_compressed_data_integrity=True)
    cams_data = cams_data['cam'][0]
    num_cams = 1
    
    RTs = np.zeros([num_cams, 3,4])
    VDs = np.zeros([num_cams, 3])
    for i in range(num_cams):
        cam_param = cams_data[i]
        cam_R, cam_t = generate_cam_Rt(
        center=cam_param['center'][0, 0], right=cam_param['right'][0, 0],
        up=cam_param['up'][0, 0], direction=cam_param['direction'][0, 0])

        RTs[i, :3, :3] = cam_R
        RTs[i , :, -1] = cam_t
        VDs[i, :] = -cam_param['direction'][0,0]

    K = np.eye(3)
    Ks = np.stack([K]*num_cams, axis=0)
    Ks[:, 0,0] = 10000; Ks[:, 1,1] = 10000;
    Ks[:, 0, 2] = 512; Ks[:, 1,2] = 512;
    return Ks, RTs, VDs, num_cams

def render_mask(grids):
    im = np.zeros([512, 512])
    grids = grids.astype(np.int)
    # for i in range(grids.shape[0]):
        # im[grids[i,0], grids[i,1]] = 1
    im[grids[:, 1], grids[:,0]] = 1
    plt.imshow(im, cmap='gray')
    plt.show()

def render_gray(grays, num_cams):
    for i in range(num_cams):
        gray = grays[i]
        # depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) # 0~1
        plt.imshow(gray)
        plt.show()

def render_rgbs(rgbs, num_cams):
    for i in range(num_cams):
        rgb = rgbs[i]
        plt.imshow(rgb)
        plt.show()

def load_parameters(obj_path, tex_path, cam_path):
    text_img = cv2.imread(tex_path).astype(np.float32)[:,:, ::-1] / 255.0

    verts, faces, normals, tri_normals, uvs, tri_uvs = load_obj_mesh(obj_path, with_normal=True, with_texture=True)
    # print(verts.shape, faces.shape, tri_normals.shape, uvs.shape, tri_uvs.shape)

    # t0 = time.time()
    # mesh = trimesh.load(obj_path)

    # verts = mesh.vertices
    # faces = mesh.faces
    # uvs = mesh.visual.uv
    n_faces = faces.shape[0]
    n_verts = verts.shape[0]

    # # uv maps.
    tri_uvs = uvs[faces].reshape(n_faces, 3, 2)

    # normals = mesh.vertex_normals
    tri_normals = normals[faces].reshape(n_faces, 3, 3)
    # print(time.time() - t0)

    # t0 = time.time()
    # om_mesh = om.read_trimesh(obj_path, halfedge_tex_coord = True)
    # verts = om_mesh.points()
    # n_faces = om_mesh.n_faces()
    # n_verts = om_mesh.n_vertices()
    # faces = om_mesh.face_vertex_indices()
    # fh_indices = om_mesh.face_halfedge_indices()

    # he_uv = om_mesh.halfedge_texcoords2D()
    # tri_uvs = (he_uv[fh_indices]).reshape(n_faces, 3, 2)
    
    # om_mesh.request_face_normals()
    # om_mesh.request_vertex_normals()
    # om_mesh.update_normals()
    # vns = om_mesh.vertex_normals()
    # tri_normals = (vns[faces]).reshape(n_faces, 3, 3)
    # print(time.time() - t0)

    # cam's parameters.
    Ks, RTs, VDs, num_cams = load_cams(cam_path)

    # colors & lights.
    ambient = 0.4; light_stren = 0.9;
    light_dirs = np.array([0, 0, 1])

    # generate torch tensor for CUDA. e.g. batch = 2.
    vertices = torch.from_numpy(np.stack([verts]*num_cams,axis=0)).float().cuda()
    faces = torch.from_numpy(np.stack([faces]*num_cams,axis=0)).int().cuda()
    normals = torch.from_numpy(np.stack([tri_normals]*num_cams, axis=0)).float().cuda()
    uvs = torch.from_numpy(np.stack([tri_uvs]*num_cams, axis=0)).float().cuda()
    texs = torch.from_numpy(np.stack([text_img]*num_cams, axis=0)).float().cuda()
    # light_dirs = torch.from_numpy(np.stack([light_dirs]*num_cams, axis=0)).float().cuda()
    light_dirs = torch.from_numpy(VDs).float().cuda()

    Ks = torch.from_numpy(Ks).float().cuda()
    RTs = torch.from_numpy(RTs).float().cuda()
    view_dirs = torch.from_numpy(VDs).float().cuda()
    # output tensor.
    depth = torch.ones([num_cams, 1024, 1024]).float().cuda()
    depth *= 11.
    # grids = torch.zeros([num_cams, n_verts, 3]).float().cuda()
    # status = torch.zeros([num_cams, n_verts]).int().cuda()
    RGBs = torch.zeros([num_cams, 1024, 1024, 4]).float().cuda() # the last filter save the filter triangles' num
    masks = torch.zeros([num_cams, 1024, 1024]).int().cuda()

    return vertices, faces, normals, uvs, texs, Ks, RTs, ambient, light_stren,  view_dirs, light_dirs, \
           depth, RGBs, masks, num_cams

def test(obj_path, tex_path, cam_path):

    vertices, faces, normals, uvs, texs, Ks, RTs, ambient, light_stren,  view_dirs, light_dirs, \
           depth, RGBs, masks, num_cams = load_parameters(obj_path, tex_path, cam_path)

    t1 = time.time()
    RenderUtils.render_mesh(vertices, faces, normals, 
                            uvs, texs,
                            Ks, RTs,
                            1024, 1024, 
                            ambient, light_stren, view_dirs, light_dirs, False, False,
                            depth, RGBs, masks)
                            
    torch.cuda.synchronize()
    print((time.time() - t1) * 1e3, num_cams)
    # render_mask(grids[0].cpu().numpy())
    # render_gray(depth.cpu().numpy(), num_cams)
    RGBs[..., :3] /= RGBs[..., -1:]
    render_rgbs(RGBs[..., :3].cpu().numpy(), num_cams)
    # render_gray(masks.cpu().numpy(), num_cams)
    
def test_opencv(obj_path, tex_path, cam_path):
    import cv2

    vertices, faces, normals, uvs, texs, Ks, RTs, ambient, light_stren,  view_dirs, light_dirs, \
           depth, RGBs, masks, num_cams = load_parameters(obj_path, tex_path, cam_path)
    
    WNAME = 'render'
    cv2.namedWindow(WNAME, 0)
    cv2.resizeWindow(WNAME, 512, 512)
    global_pitch = 0; global_yaw = 0; global_dist = 200; global_focal = 4.5

    def render(new_Ks, new_RTs, new_view_dirs):
        RGBs = torch.zeros([1, 1024, 1024, 4]).float().cuda()
        depth = torch.ones([1, 1024, 1024]).float().cuda()
        depth *= 300.
        masks = torch.zeros([num_cams, 1024, 1024]).int().cuda()
        # t1 = time.time()
        RenderUtils.render_mesh(vertices, faces, normals, 
                                uvs, texs,
                                new_Ks, new_RTs,
                                1024, 1024,
                                ambient, light_stren, new_view_dirs, light_dirs, False, True,
                                depth, RGBs, masks)
        # torch.cuda.synchronize()
        # print((time.time() - t1) * 1e3)
        return RGBs, depth, masks
    
    def update_cam(pitch, yaw, d, focal):
        # suppose pitch located in (-90, 90),  yaw located in (0, 360).
        target = [0,0,0]
        angle_xz  = (math.pi / 180) * (yaw % 360)
        if pitch > 0:
            angle_y = (math.pi / 180) * pitch if pitch <= 90 else (math.pi / 180) * 90
        else:
            angle_y = (math.pi / 180) * pitch if pitch >= -90 else (math.pi / 180) * (-90)

        eye = np.asarray([d * math.cos(angle_y) * math.sin(angle_xz),
                          d * math.sin(angle_y),
                          d * math.cos(angle_y) * math.cos(angle_xz)])

        # calculate up vector.
        left = np.cross([0, 1, 0], -eye)
        up = np.cross(-eye, left)
        up /= np.linalg.norm(up)

        fwd = np.asarray(target, np.float64) - eye
        fwd /= np.linalg.norm(fwd)

        right = -left
        right /= np.linalg.norm(right)

        cam_R, cam_t = generate_cam_Rt(eye, fwd, right, up)
        
        RTs = np.zeros([num_cams, 3, 4])
        VDs = np.zeros([num_cams, 3])
        for i in range(num_cams):
            RTs[i, :3, :3] = cam_R
            RTs[i, :3, -1] = cam_t
            VDs[i, :3] = -fwd
        
        K = np.eye(3)
        Ks = np.stack([K]*num_cams, axis=0)
        Ks[:, 0,0] = focal; Ks[:, 1,1] = focal;
        Ks[:, 0, 2] = 512; Ks[:, 1,2] = 512;

        return Ks, RTs, VDs

    def update(w_name, pitch, yaw, dist, focal):
        new_Ks, new_RTs, new_view_dirs = update_cam(pitch, yaw, dist, focal)
        new_Ks = torch.from_numpy(new_Ks).float().cuda()
        new_RTs = torch.from_numpy(new_RTs).float().cuda()
        new_view_dirs = torch.from_numpy(new_view_dirs).float().cuda()

        RGBs, _, masks = render(new_Ks, new_RTs, new_view_dirs) # [B, H, W, ?]
        RGBs = RGBs * masks[..., None] + (1 - masks[..., None])
        RGBs[..., :3] /= RGBs[..., -1:]
        RGB = RGBs[0, ..., :3]

        # depths = (depths - torch.min(depths)) / (torch.max(depths) - torch.min(depths))
        cv2.imshow(WNAME, RGB.cpu().numpy()[..., ::-1])
        # cv2.imshow(WNAME, masks.float().cpu().numpy()[0])
        # cv2.imshow(WNAME, depths.cpu().numpy()[0])


    update(WNAME, global_pitch, global_yaw, global_dist, global_focal)
    while True:
        key_code = cv2.waitKey(1) # keyboard event.

        if key_code != -1:
            if chr(key_code) == 'a':                
                global_yaw += 1
            elif chr(key_code) == 'd':
                global_yaw -= 1
            elif chr(key_code) == 'w':
                global_pitch += 1
            elif chr(key_code) == 's':
                global_pitch -= 1
            elif chr(key_code) == 'l':
                global_dist -= 0.05
            elif chr(key_code) == 'o':
                global_dist += 0.05
            elif chr(key_code) == 'z':
                global_focal *= 1.05
            elif chr(key_code) == 'x':
                global_focal *= 0.95
            elif chr(key_code) == 'q':
                break

            update(WNAME, global_pitch, global_yaw, global_dist, global_focal)

    cv2.destroyWindow(WNAME)
    

if __name__ == '__main__':
    obj_path = "./render_people_dataset/Males_Vis_Models/Male_010/Male_010.obj"
    tex_path = "./render_people_dataset/Males_Vis_Models/Male_010/tex/Male_010.jpg"
    cam_path = "./render_people_dataset/image_data/Male_011/meta/cam_data.mat"

    test_opencv(obj_path, tex_path, cam_path)

    