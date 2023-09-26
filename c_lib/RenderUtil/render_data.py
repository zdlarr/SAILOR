import sys, os

import numpy as np
import cv2
import torch
import trimesh
import math

import scipy.io as sio
import time
import matplotlib.pyplot as plt

from c_lib.RenderUtil.mesh import load_obj_mesh

from tqdm import tqdm

img_size = 1024
n_focal = 690
cam_dis = 1.5

def load_parameters(obj_path, tex_path):
    text_img = cv2.imread(tex_path).astype(np.float32)[:,:, ::-1] / 255.0

    verts, faces, normals, tri_normals, uvs, tri_uvs = load_obj_mesh(obj_path, with_normal=True, with_texture=True)
    n_faces = faces.shape[0]
    n_verts = verts.shape[0]

    # uv maps.
    tri_uvs = uvs[faces].reshape(n_faces, 3, 2)

    # normals = mesh.vertex_normals
    tri_normals = normals[faces].reshape(n_faces, 3, 3)

    return verts, faces, tri_normals, tri_uvs, text_img


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


def generate_cams(pitch, yaw, d, focal, num_cams, im_size=512, target=[0,0,0]):
    # suppose pitch located in (-90, 90),  yaw located in (0, 360).
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
    Ks[:, 0, 2] = im_size // 2; Ks[:, 1,2] = im_size // 2;

    return Ks, RTs, VDs

def render(out_path, folder_name, subject_name, Ks, RTs, VDs, total_num_cams, yaw, pitch, num_cams=72, img_size=512):
    # set path for obj, prt
    mesh_file = os.path.join(folder_name, subject_name + '.obj')
    if not os.path.exists(mesh_file):
        print('ERROR: obj file does not exist!!', mesh_file)
        return

    # text_file = os.path.join(folder_name, 'tex', subject_name + '.jpg')
    text_file = os.path.join(folder_name, 'material0.jpeg')
    if not os.path.exists(text_file):
        print('ERROR: dif file does not exist!!', text_file)
        return

    os.makedirs(os.path.join(out_path, 'GEO', 'OBJ', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'PARAM', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'RENDER', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'MASK', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'DEPTH', subject_name),exist_ok=True)
    
    # copy obj file
    cmd = 'cp %s %s' % (mesh_file, os.path.join(out_path, 'GEO', 'OBJ', subject_name))
    print(cmd)
    # os.system(cmd)

    verts, faces, normals, uvs, tex = load_parameters(mesh_file, text_file)

    # generate torch tensor for CUDA. e.g. batch = 12, 36
    vertices = torch.from_numpy(np.stack([verts]*num_cams,axis=0)).float().cuda()
    faces = torch.from_numpy(np.stack([faces]*num_cams,axis=0)).int().cuda()
    normals = torch.from_numpy(np.stack([normals]*num_cams, axis=0)).float().cuda()
    uvs = torch.from_numpy(np.stack([uvs]*num_cams, axis=0)).float().cuda()
    texs = torch.from_numpy(np.stack([tex]*num_cams, axis=0)).float().cuda()

    ambient = 0.4 + np.random.rand() * 0.08; 
    light_stren = 0.25 + np.random.rand() * 0.1;

    # output tensor.
    depth = torch.ones([num_cams, img_size, img_size]).float().cuda()
    depth *= 1000.
    # grids = torch.zeros([num_cams, n_verts, 3]).float().cuda()
    # status = torch.zeros([num_cams, n_verts]).int().cuda()
    RGBs = torch.zeros([num_cams, img_size, img_size, 4]).float().cuda() # the last filter save the filter triangles' num
    masks = torch.zeros([num_cams, img_size, img_size]).int().cuda()

    WNAME = 'render'
    cv2.namedWindow(WNAME, 0)
    cv2.resizeWindow(WNAME, 1024, 1024)

    for i in tqdm(range(0, total_num_cams, num_cams)): # every 30 epoch, rendering.
        K_ = Ks[i:i+num_cams]; RT_ = RTs[i:i+num_cams]; VD_ = VDs[i:i+num_cams]

        depth = torch.ones_like(depth) * 1000
        RGBs = torch.zeros_like(RGBs)
        masks = torch.zeros_like(masks)

        light_dirs = torch.from_numpy(VD_).float().cuda()
        light_dirs += torch.randn_like(light_dirs) * 0.03
        light_dirs = light_dirs / torch.norm(light_dirs, dim=1)[:, None]

        K_ = torch.from_numpy(K_).float().cuda()
        RT_ = torch.from_numpy(RT_).float().cuda()
        view_dirs = torch.from_numpy(VD_).float().cuda()

        # render dataset, perspective projection here.
        RenderUtils.render_mesh(vertices, faces, normals, 
                                uvs, texs,
                                K_, RT_,
                                img_size, img_size,
                                ambient, light_stren, view_dirs, light_dirs, False, False,
                                depth, RGBs, masks)
        
        # RGBs = RGBs * masks[..., None]
        RGBs[..., -1] += (1 - masks)
        RGBs[..., :3] /= RGBs[..., -1:]
        RGBs *= masks[..., None]
        depth = depth * masks

        # save.
        for j in range(num_cams):
            y = yaw[(i + j) // len(pitch)]; p = pitch[(i + j) % len(pitch)]
            out_all_f = cv2.cvtColor(RGBs[j, ..., :3].cpu().numpy(), cv2.COLOR_RGB2BGR)
            out_mask = masks[j].cpu().numpy()
            out_depth = depth[j].cpu().numpy() * 10000
            
            cv2.imwrite(os.path.join(out_path, 'RENDER', subject_name,  '%d_%d.jpg'%(y,p)), 255.0*out_all_f)
            cv2.imwrite(os.path.join(out_path, 'MASK', subject_name, '%d_%d.png'%(y,p)), 255.0*out_mask)
            # cv2.imwrite(os.path.join(out_path, 'DEPTH', subject_name, '%d_%d.png'%(y,p)),255.0*out_depth)
            np.save(os.path.join(out_path, 'PARAM', subject_name, '%d_%d.npy'%(y,p)), {'K': K_[j].cpu().numpy(), 'RT': RT_[j].cpu().numpy()})
            # np.save(os.path.join(out_path, 'DEPTH', subject_name, '%d_%d.npy'%(y,p)), out_depth)
            cv2.imwrite(os.path.join(out_path, 'DEPTH', subject_name, '%d_%d.png'%(y,p)), out_depth.astype(np.uint16)) # the depth * 10000.

            # show image.
            out_depth = out_depth + (1 - out_mask) * 512.
            out_depth = (out_depth - np.min(out_depth)) / (np.max(out_depth) - np.min(out_depth))
            cv2.imshow(WNAME, out_all_f)
            # cv2.imshow(WNAME, out_depth)
            cv2.waitKey(1)

def load_real_cameras(basic_path=None, num_cameras=8):
    if basic_path is None:
        basic_path = '/home/ssd2t/dz/render_dataset_real768/cam'
    
    cams = []
    all_cam_paths = os.listdir(basic_path)
    for item in all_cam_paths:
        cam_item = []
        for i in range(num_cameras):
            # load from the cameras.
            cam_path = os.path.join(basic_path, item, str(i) + '.npy')
            cam_data = np.load(cam_path, allow_pickle=True)
            cam_K  = cam_data.item().get('K');
            cam_RT = cam_data.item().get('RT');
            cam_item.append({'K': cam_K, 'RT': cam_RT});

        cams.append(cam_item)
    return cams

def render_real_cameras(num_cams=8):
    output_dir = '/home/ssd2t/dz/render_dataset_fake'
    output_dir2 = '/home/ssd2t/dz/render_dataset_fake/MAIN_FACE2'
    base_dir = '/home/ssd2t/dz/THuman2.0'
    base_dir2 = '/home/ssd2t/dz/render_dataset2/MAIN_FACE'

    dir_list = [base_dir]
    # render the target mesh in the target 8 views.
    cams_data = load_real_cameras(num_cameras=num_cams); # totally 120's objects.
    Ks, RTs, VDs = np.stack([np.eye(3)]*8, axis=0), np.zeros([8, 3, 4]), np.zeros([8, 3])
    # view_ids = [0, 45, 90, 135, 180, 225, 270, 315];
    view_ids = [180, 225, 135, 315, 0, 45, 90, 270]
    
    for dir in dir_list:
        datas = sorted(os.listdir(dir))
        
        for k, subject_name in enumerate(datas[478:]):
            # random_cam_id = np.random.randint(0, len(cams_data))
            cam_select = cams_data[k % len(cams_data)]
            for i in range(num_cams):
                K  = cam_select[i]['K'];
                RT = cam_select[i]['RT'];
                cam_pos = (- RT[:3, :3].T @ RT[:3, -1:])[:, 0] # [3].
                Ks[i] = K; RTs[i] = RT; VDs[i] = cam_pos;

            input_folder = os.path.join(dir, subject_name)
            yaw=list(range(0, num_cams, 1));
            render(output_dir, input_folder, subject_name, Ks, RTs, VDs, num_cams, yaw, [0], num_cams=num_cams, img_size=img_size)
            
            # rendering front views:
            front_view_folder = os.path.join(base_dir2, subject_name)
            view_id_path = os.path.join(front_view_folder, 'view_id.npy')
            view_id = int(np.load(view_id_path, allow_pickle=True).item().get('view_id').split('_')[0])

            # the view_id is located in [0, 359].
            final_vid = -1;
            min_t = 9999;
            for t in range(num_cams):
                tmp = abs(view_id - view_ids[t])
                if tmp < min_t:
                    min_t = tmp
                    final_vid = t;


            # select the front_view id.
            assert final_vid != -1
            Ks_new = np.copy(Ks[final_vid])[None, ...]; RTs_new = np.copy(RTs[final_vid])[None, ...]; VDs_new = np.copy(VDs[final_vid])[None, ...];
            Ks_new *= 2;
            Ks_new[:, -1, -1] = 1.0;
            yaw_new=list(range(0,1, 1));
            render(output_dir2, input_folder, subject_name, Ks_new, RTs_new, VDs_new, 1, yaw_new, [0], num_cams=1, img_size=1536)


def main():
    output_dir = '/home/ssd2t/dz/render_dataset8'
    output_dir2 = '/home/ssd2t/dz/render_dataset8/MAIN_FACE'
    # base_dir = '/home/dz/my_Rendering/render_people_dataset'
    base_dir = '/home/ssd2t/dz/THuman2.0'
    base_dir2 = '/home/ssd2t/dz/render_dataset2/MAIN_FACE'
    
    # dir_list = [os.path.join(base_dir, 'Males_Vis_Models'),
    #                 os.path.join(base_dir, 'galadata'),
    #                 os.path.join(base_dir, 'normaldata'),
    #                 os.path.join(base_dir, 'Female_Vis_Models')]
    # main_face_dir = '/home/dz/my_Rendering/pifu_dataset/render_dataset2/MAIN_FACE'
    dir_list = [base_dir]

    # generate camera parameters.
    pitch=[0]; yaw=list(range(0,360, 1))
    total_num_cams = len(pitch) * len(yaw)
    Ks, RTs, VDs = np.stack([np.eye(3)]*total_num_cams, axis=0), np.zeros([total_num_cams, 3, 4]), np.zeros([total_num_cams, 3])
    

    for dir in dir_list:
        datas = sorted(os.listdir(dir))
        
        k = 0
        for y in yaw:
            for p in pitch:
                noise_cam_dis = cam_dis + (-0.1) + np.random.rand() * 0.03
                # noise_n_focal = n_focal + (-5) + np.random.rand() * 10
                K, RT, VD = generate_cams(p, y, d=noise_cam_dis, focal=n_focal, num_cams=1, im_size=img_size)
                Ks[k] = K[0]; RTs[k] = RT[0]; VDs[k] = VD[0]
                k += 1
        
        for subject_name in datas[478:]:
            # for main views
            # main_face_id_path = os.path.join(main_face_dir, subject_name, 'view_id.npy')
            # main_id = int(np.load(main_face_id_path, allow_pickle=True).item().get('view_id').split('_')[0])

            # Ks, RTs, VDs = np.stack([np.eye(3)]*1, axis=0), np.zeros([1, 3, 4]), np.zeros([1, 3])
            # K, RT, VD = generate_cams(0, main_id, d=1.8, focal=320 * 4, num_cams=1)

            input_folder = os.path.join(dir, subject_name)
            
            # yaw = [main_id]
            total_num_cams = len(pitch) * len(yaw)
            
            # when rendering all data.
            render(output_dir, input_folder, subject_name, Ks, RTs, VDs, total_num_cams, yaw, pitch, img_size=img_size)
            
            # rendering front view.
            front_view_folder = os.path.join(base_dir2, subject_name)
            view_id_path = os.path.join(front_view_folder, 'view_id.npy')
            view_id = int(np.load(view_id_path, allow_pickle=True).item().get('view_id').split('_')[0])

            # # get K and RT matrix.
            # front_param_path = os.path.join(output_dir, 'PARAM', subject_name, str(view_id) + '_0.npy')
            # front_face_param = np.load(front_param_path, allow_pickle=True)
            # face_K, face_RT = front_face_param.item().get('K'), front_face_param.item().get('RT')
            
            # rendering front view.
            # Ks_new = np.copy(Ks[view_id])[None, ...]; RTs_new = np.copy(RTs[view_id])[None, ...]; VDs_new = np.copy(VDs[view_id])[None, ...];
            # Ks_new *= 4;
            # Ks_new[:, -1, -1] = 1.0;
            # total_num_cams = 1;
            # yaw_new=list(range(0,1, 1));
            # render(output_dir2, input_folder, subject_name, Ks_new, RTs_new, VDs_new, total_num_cams, yaw_new, pitch, num_cams=1, img_size=img_size)

if __name__ == '__main__':
    main()
# if __name__ == '__main__':
#     render_real_cameras()
    