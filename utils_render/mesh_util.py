"""
    Utils for mesh reconstruction & object operations.
    # recon, save_mesh
"""

import numpy as np
import torch
import time
import mcubes

def eval_func(opts, model, num_views, points, device):
    points = np.expand_dims(points, axis=0) # [1, 3, N_Points.]
    if not opts.is_train: # especially for the front-face,
        points = np.concatenate([points] * num_views, axis=0) # [N_view * 1, 4, N_points]; {front_view; other 3 views};
    else:
        points = np.concatenate([points] * num_views, axis=0) # [N_view * 1, 4, N_points]; {front_view; other 3 views};

    samples = torch.from_numpy(points).to(device=device).float() # [N1, 3, N2].
    model.forward_query(samples) # [[B, 1, N_points] * Stages.] -> [1, N_points]
    pred = model.pred_occ[0] # select the first one.
    return pred.detach().cpu().numpy()


def batch_eval(opts, model, device, num_views, points, eval_func, num_samples=512 * 512 * 512):
    # points of shape: [3, N_points];
    num_pts = points.shape[-1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = \
            eval_func(opts, model, num_views, points[:, i * num_samples:i * num_samples + num_samples], device)
    
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(opts, model, num_views, points[:, num_batches * num_samples:], device)
    
    if opts.save_face_weights:
        model.geo_net.face_fusion_module.save_points_weight_()
    
    return sdf


def create_grids(resolution, bbox_min, bbox_max):
    res_x = resolution
    res_y = resolution
    res_z = resolution
    
    coords = np.mgrid[:res_x, :res_y, :res_z]
    coords = coords.reshape(3, -1)
    coord_matrix = np.eye(4)
    length = bbox_max - bbox_min
    coord_matrix[0, 0] = length[0] / res_x
    coord_matrix[1, 1] = length[1] / res_y
    coord_matrix[2, 2] = length[2] / res_z
    coord_matrix[0:3, 3] = bbox_min
    # to world coordinates.
    coords = coord_matrix[:3, :3] @ coords + coord_matrix[:3, -1:]
    coords = coords.reshape(3, res_x, res_y, res_z)
    return coords, coord_matrix


def eval_grid_octree(opts, model, device, num_views, coords, eval_func,
                     init_resolution=64, threshold=0.01,
                     num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf = np.zeros(resolution)

    dirty = np.ones(resolution, dtype=np.bool)
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        # subdivide the grid
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        #print('step size:', reso, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval(opts, model, device, num_views, points, eval_func, num_samples=num_samples)
        dirty[test_mask] = False

        # do interpolation
        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
                    # if center marked, return
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = sdf[x, y, z]
                    v1 = sdf[x, y, z + reso]
                    v2 = sdf[x, y + reso, z]
                    v3 = sdf[x, y + reso, z + reso]
                    v4 = sdf[x + reso, y, z]
                    v5 = sdf[x + reso, y, z + reso]
                    v6 = sdf[x + reso, y + reso, z]
                    v7 = sdf[x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    # this cell is all the same
                    if (v_max - v_min) < threshold:
                        sdf[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2

    return sdf.reshape(resolution)


def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix


def eval_grid(opt, model, device, num_views, coords, eval_func, num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf = batch_eval(opt, model, device, num_views, coords, eval_func, num_samples=num_samples)
    return sdf.reshape(resolution)

def reconstruction(opts, model, device, num_views, resolution, bbox_min, bbox_max, 
                   use_octree=True, num_sample_batch=10000):  
    coords, mat = create_grids(resolution, bbox_min, bbox_max)
    res_XYZ = (resolution, resolution, resolution)
    # step 1. filter features and get the returned filtered normal, depths.
    model.forward_filter()
    
    # step 2. eval the grids' occupancy in several batch.
    t1 = time.time()
    if 1:# use_octree:
        # init_resolution = 64
        # threshold = 0.05
        # # initialize the properties: SDF, dirty, grid_mask.
        # sdf = np.zeros(res_XYZ)
        # # dirty = np.ones(res_XYZ, dtype=np.bool)

        # notprocessed = np.zeros(res_XYZ, dtype=bool)
        # notprocessed[:-1, :-1, :-1] = True

        # grid_mask = np.zeros(res_XYZ, dtype=np.bool)

        # reso = res_XYZ[0] // init_resolution

        # while reso > 0:
        #     # subdivide the grid
        #     grid_mask[0:res_XYZ[0]:reso, 0:res_XYZ[1]:reso, 0:res_XYZ[2]:reso] = True
        #     # test samples in this iteration
        #     test_mask = np.logical_and(grid_mask, notprocessed)
        #     #print('step size:', reso, 'test sample size:', test_mask.sum())
        #     points = coords[:, test_mask]
        
        #     sdf[test_mask] = batch_eval(opts, model, device, num_views, points, eval_func, num_sample_batch)
        #     notprocessed[test_mask] = False

        #     # do interpolation
        #     if reso <= 1:
        #         break

        #     x_grid = np.arange(0, res_XYZ[0], reso)
        #     y_grid = np.arange(0, res_XYZ[1], reso)
        #     z_grid = np.arange(0, res_XYZ[2], reso)

        #     v = sdf[tuple(np.meshgrid(x_grid, y_grid, z_grid, indexing='ij'))]

        #     v0 = v[:-1, :-1, :-1]
        #     v1 = v[:-1, :-1, 1:]
        #     v2 = v[:-1, 1:, :-1]
        #     v3 = v[:-1, 1:, 1:]
        #     v4 = v[1:, :-1, :-1]
        #     v5 = v[1:, :-1, 1:]
        #     v6 = v[1:, 1:, :-1]
        #     v7 = v[1:, 1:, 1:]

        #     x_grid = x_grid[:-1] + reso//2
        #     y_grid = y_grid[:-1] + reso//2
        #     z_grid = z_grid[:-1] + reso//2

        #     nonprocessed_grid = notprocessed[tuple(np.meshgrid(x_grid, y_grid, z_grid, indexing='ij'))]

        #     v = np.stack([v0,v1,v2,v3,v4,v5,v6,v7], 0)
        #     v_min = v.min(0)
        #     v_max = v.max(0)
        #     v = 0.5*(v_min+v_max)

        #     skip_grid = np.logical_and(((v_max - v_min) < threshold), nonprocessed_grid)

        #     n_x = res_XYZ[0] // reso
        #     n_y = res_XYZ[1] // reso
        #     n_z = res_XYZ[2] // reso

        #     xs, ys, zs = np.where(skip_grid)
        #     for x, y, z in zip(xs*reso, ys*reso, zs*reso):
        #         sdf[x:(x+reso+1), y:(y+reso+1), z:(z+reso+1)] = v[x//reso,y//reso,z//reso]
        #         notprocessed[x:(x+reso+1), y:(y+reso+1), z:(z+reso+1)] = False

        #     reso //= 2
    
        # sdf = sdf.reshape(res_XYZ)
        
        sdf = eval_grid_octree(opts, model, device, num_views, coords, eval_func, num_samples=num_sample_batch)
        
    else:
        coords, mat = create_grid(resolution, resolution, resolution,
                              bbox_min, bbox_max, transform=None)
        sdf = eval_grid(opts, model, device, num_views, coords, eval_func, num_samples=num_sample_batch)

        # coords = coords.reshape([3, -1])
        # sdf = batch_eval(opts, model, device, points, eval_func, num_sample_batch)
        
    torch.cuda.synchronize()
    t2 = time.time()
    print('Time for predict sdf field: {}'.format(t2 - t1))

    # sdf = mcubes.smooth(sdf)
    verts, faces = mcubes.marching_cubes(sdf, 0.5)

    # verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5) # using marching cubes to get all vertices & faces.
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    return verts, faces, None, None


def save_obj_mesh(save_path, verts, faces, colors = None, inv_order = True):
    mcubes.export_obj(verts, faces, save_path)
    # file = open(save_path, 'w')

    # if colors is not None:
    #     for idx, v in enumerate(verts):
    #         c = colors[idx]
    #         file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    # else:
    #     for idx, v in enumerate(verts):
    #         file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
            
    # for f in faces:
    #     f_plus = f + 1
    #     if inv_order:
    #         file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    #     else:
    #         file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))

    # file.close()
