
import sys, os
import argparse
from upsampling.options.RenTrainOptions import RenTrainOptions


class RenTestOptions(RenTrainOptions):
    
    def __init__(self):
        RenTrainOptions.__init__(self)
        
    def parse(self):
        opts = RenTrainOptions.parse(self)
        opts.ren_data_root    = ''

        opts.gpu_ids          = [0]
        opts.is_val           = True
        opts.img_size         = 1024
        opts.is_train         = False

        # for geometries:
        opts.gen_mesh    = False
        opts.resolution  = 512
        opts.bbox_min    = [-0.8, -1, -0.8]
        opts.bbox_max    = [0.8, 1.0, 0.8]
        
        # for rendering.
        opts.focal_x          = 720.0
        opts.target_num_views = 1 # we only need one target view for rendering (OOM).
        opts.num_sampled_rays = 1024 * 32 # rendering in patches. 512 * 512 // (1024 * 4)
        
        opts.data_name        = 'SAILOR_test'
        opts.phase            = 'testing'
        
        opts.real_data        = True # whether testing on real data.
        opts.test_subject     = None
        opts.num_threads      = 0 # the num of the threads for dataset.
        opts.rendering_seperate_angle = 6 # seperate 60 degrees.
        opts.pre_fuse_depths  = True
        opts.support_post_fusion = True
        
        # sampling points, enough points here.
        opts.octree_rate               = 4
        opts.octree_level              = 2
        opts.instersect_start_level    = 0
        opts.volume_dim                = 128
        opts.num_max_hits_voxel        = 128
        
        # total sampling points.
        opts.rend_full_body = True
        if opts.rend_full_body:
            opts.num_sampled_points_coarse = 48 # full-body sampled points.
        else:
            opts.num_sampled_points_coarse = 32 # for portrait rendering.

        # control the tsdf volume's degrees.
        opts.tsdf_th_low               = 13.5
        opts.tsdf_th_high              = 13.5
        # suppose that, succ travel N's voxels
        opts.early_stop_th = -1
        opts.max_intersected_voxels = -1

        # camera params.
        opts.rendering_static = True
        
        opts.batch_size       = 1
        # default num of views.
        opts.num_views        = 4
        # input views are the first four views.
        opts.test_view_ids    = [4, 6, 7, 0, 3, 5, 2, 1] # the last N views are used to refine input depths.
        
        opts.epoch            = 'latest'
        
        return opts