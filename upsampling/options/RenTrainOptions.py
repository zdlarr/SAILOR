
import sys, os
import argparse
from options.GeoTrainOptions import GeoTrainOptions

class RenTrainOptions(GeoTrainOptions):

    def __init__(self):
        GeoTrainOptions.__init__(self)

    def initialize(self, parser):
        parser = GeoTrainOptions.initialize(self, parser)
        # basic options.
        parser.add_argument('--name_rend_net', type=str, default='BasicRenNet', help='# the name of the Rendering Net;')
        parser.add_argument('--ren_data_root', type=str, default='/home/ssd2t/dz/render_dataset8', help='# the root dir of the dataset.1K')
        parser.add_argument('--ren_real_root', type=str, default='/home/ssd2t/dz/render_dataset5', help='# the root dir of the dataset 1k.')
        
        parser.add_argument('--basic_obj_dir', type=str, default='/home/ssd2t/dz/THuman2.0')
        parser.add_argument('--render_model_dir', type=str, default='./checkpoints_rend', help='# to save the rendered results.')
        parser.add_argument('--lr_render', type=float, default=1e-4, help='# the learning rate for rendering net.')
        parser.add_argument('--num_epoch_render', type=int, default=200, help='# the learning rate for rendering net.')
        parser.add_argument('--freq_plot_render', type=int, default=10, help='# the frequency for plotting rendering info.')
        parser.add_argument('--freq_save_render', type=int, default=200, help='# the frequency for plotting rendering info.')
        parser.add_argument('--phase', type=str, default='training', help='# the phase for training or validating.')
        parser.add_argument('--img_size', type=int, default=1024, help='# size of images')
        parser.add_argument('--data_down_sample', type=bool, default=False, help='# whether downsample the images and the camera matrixs')
        
        # structure's options.
        parser.add_argument('--octree_level', type=int, default=2, help='# num of levels for the octree.')
        parser.add_argument('--octree_rate',  type=int, default=4, help='# the rate of the octree resolution decreasing.')
        parser.add_argument('--volume_dim',   type=int, default=128, help='# the volume dimension.')
        parser.add_argument('--num_max_hits_voxel', type=int, default=128, help='# the number of the assigned max hitted voxels.')
        parser.add_argument('--freq_encoding_degree', type=int, default=4, help='# the degree of frequency encoding')
        parser.add_argument('--sh_encoding_degree', type=int, default=4, help='# the degree of sh encoding')
        # this threshold value is used to clip the psdf values into the range [-x, x]
        parser.add_argument('--tsdf_th_low', type=float, default=13.5, help='the low value of the tsdf fusion, for the insided voxels')
        parser.add_argument('--tsdf_th_high', type=float, default=13.5, help='the high value of the tsdf fusion.')

        # rays options.
        parser.add_argument('--instersect_start_level', type=int, default=0, help='# the level for start intersection.')
        parser.add_argument('--early_stop_th', type=int, default=-1, help='# the early stop threshold, -1: no early stopping.')
        parser.add_argument('--max_intersected_voxels', type=int, default=-1, help='# to control the max intersected voxels')
        
        # parameters for depth normalization.
        parser.add_argument('--valid_depth_min', type=float, default=0.2, help='# the valid min depth for Rendering.')
        parser.add_argument('--valid_depth_max', type=float, default=4.0, help='# the valid min depth for Rendering.')
        
        # mesh properties, for extracting vertices num. ~60w;
        parser.add_argument('--max_vertices_num', type=int, default=100000, help='# the max num of the mesh veretices')
        
        # features options.
        parser.add_argument('--inputs_dim_point', type=int, default=3, help='# the num of the inputs points dim.')
        parser.add_argument('--im_feats_dim', type=int, default=32, help='# the dim of the image feature.')
        parser.add_argument('--depth_feats_dim', type=int, default=32, help='# the dim of the depth feature.')
        parser.add_argument('--rgb_feats_dim', type=int, default=32, help='# the dim of the rgb feature.')
        parser.add_argument('--point_feats_dim', type=int, default=64, help='# the dim of the point feature.')
        # parser.add_argument('--point_feats_dim', type=int, default=128, help='# the dim of the point feature.')

        # mlp feature dim.
        # the feat_MLP : [64+2+..., 128, 128, 128, 128] [128, ..., 4]
        parser.add_argument('--tcnn', action='store_true', help='# whether using tcnn as mlp.')
        parser.add_argument('--feat_net_hidden_dim', type=int, default=64, help='# the dim of the mlp.')
        # parser.add_argument('--feat_net_hidden_dim', type=int, default=128, help='# the dim of the mlp.')
        parser.add_argument('--nerf_net_hidden_dim', type=int, default=128, help='# the dim of the mlp.')
        parser.add_argument('--sigma_net_hidden_dim', type=int, default=64, help='# the dim of the mlp.')
        parser.add_argument('--color_net_hidden_dim', type=int, default=64, help='# the dim of the mlp.')
        # parser.add_argument('--num_layers_feat_net', type=int, default=2-1, help='# the dim of the mlp.') # original nerf MLP.
        parser.add_argument('--num_layers_feat_net', type=int, default=3-1, help='# the dim of the mlp.')
        parser.add_argument('--num_layers_sigma_net', type=int, default=2-1, help='# the dim of the mlp.')
        parser.add_argument('--num_layers_color_net', type=int, default=3-1, help='# the dim of the mlp.')
        parser.add_argument('--num_layers_nerf_net', type=int, default=2-1, help='# the dim of the mlp.')

        # options for training.
        parser.add_argument('--support_DDP', type=bool, default=True, help='# whether using DDP as training stragegy.')
        parser.add_argument('--aug_cam', type=bool, default=False, help='# whether using camera param augmentation.')
        parser.add_argument('--aug_depths', type=bool, default=True, help='# whether using depth augmentation for training images.')
        parser.add_argument('--num_sampled_rays', type=int, default=96 ** 2, help='# the sampled rays num per view. batch size = 2')
        # parser.add_argument('--num_sampled_rays', type=int, default=156 ** 2, help='# the sampled rays num per view. batch size = 1')
        parser.add_argument('--num_sampled_points', type=int, default=64+32+32, help='# the sampled points num for per ray.')
        # new sampling approach for nerf-based rendering.
        parser.add_argument('--num_sampled_points_coarse', type=int, default=48, help='# the sampled points num for per ray.')
        parser.add_argument('--num_sampled_points_fine', type=int, default=0, help='# the sampled points num for per ray.')
        
        parser.add_argument('--target_num_views', type=int, default=2, help='# num of the target sampled views for training.')
        parser.add_argument('--using_trunc_exp_sigma', type=bool, default=False, help='# whether using the exp sigma terms.')
        parser.add_argument('--using_sdf', type=bool, default=False, help='# whether using the SDF processing.')
        parser.add_argument('--using_nvsnerf', type=bool, default=False, help='# whether using the NVS-Nerf module.')
        parser.add_argument('--using_transnerf', type=bool, default=True, help='# whether using the Trans-Nerf module.')
        parser.add_argument('--encode_rgbd', type=bool, default=True, help='# whether encoding the rgbd values.')
        parser.add_argument('--dim_inputs', type=int, default=4, help='# the encoding feature dim for the encoding nets.')
        parser.add_argument('--user_dists', type=float, default=1.0, help='# the distance for nomarlizing depths.')

        parser.add_argument('--lr_gamma', type=float, default=0.98, help='# the exp decreasing rate for the optimizer. lr * gamma**k')
        parser.add_argument('--lam_depth', type=float, default=1.0, help='# the loss weight for the depth.')
        parser.add_argument('--lam_rgb', type=float, default=1.0, help='# the loss weight for the rgbs.')
        parser.add_argument('--lam_normal', type=float, default=0.0, help='# the loss weight for the normals.')
        parser.add_argument('--lam_udf', type=float, default=0.6, help='# the loss weight for the udfs.')
        parser.add_argument('--lam_alpha', type=float, default=0.0, help='# the loss weight for the alpha maps.')
        parser.add_argument('--lam_reg', type=float, default=0.001, help='# the loss weight for the regularization.')
        parser.add_argument('--lam_ek', type=float, default=0.0, help='# the loss weight for the eikonal loss.')

        # options for density function. & other properties.
        parser.add_argument('--s_min', type=float, default=10.0, help='# the minimum s for logistic density.')
        parser.add_argument('--s_init', type=float, default=0.1, help='# the initial s for logistic density.')
        # for laplace function properties.
        parser.add_argument('--beta_init', type=float, default=0.0, help='# the initial beat for laplace density.')
        parser.add_argument('--beta_min', type=float, default=1.0/1000.0, help='# the initial s for laplace density.')
        parser.add_argument('--beta_max', type=float, default=1.0/100.0, help='# the initial s for laplace density.')
        parser.add_argument('--dilate_ksize', type=int, default=5, help='# the dilated kernel size.')
        
        parser.add_argument('--only_train_color', type=bool, default=False, help='# whether only training colornet.')
        parser.add_argument('--only_train_depth_refine', type=bool, default=False, help='# whether only training depth refinement.')
        parser.add_argument('--enable_norm_reg_loss', type=bool, default=False, help='# whether enabling training normal regularization loss')
        parser.add_argument('--neighb_offset_render', type=float, default=0.005, help='# the offset for regularization loss.')
        parser.add_argument('--neighb_offset_real', type=float, default=0.003, help='# the offset for regularization loss. as 6mm')
        
        #options for points sampling.
        parser.add_argument('--ray_patch_sampling', type=bool, default=True, help='# whether sampling rays in patch.')
        parser.add_argument('--ray_bbox_sampling', type=bool, default=False, help='# whether sampling rays in bbox.')
        parser.add_argument('--pifu_sampled_points', type=int, default=8000, help='the points sampled for training pifu.')
        
        #whether adopting unet or upsampling pipeline.
        parser.add_argument('--adopting_unet', type=bool, default=False, help='# using unet to refine the nerf results.')
        parser.add_argument('--adopting_upsampling', type=bool, default=True, help='# using ray upsampling to refine results.')
        parser.add_argument('--up_scale_rate', type=int, default=1, help='# the upscaling rate.')
        parser.add_argument('--rend_full_body', type=bool, default=True, help='# whether rendering full body')
        
        #whether adopting the depth_refine.
        parser.add_argument('--support_depth_refine_render', type=bool, default=True, help='# support the depth refinement module.')
        parser.add_argument('--train_included_depth_refine', type=bool, default=True, help='# training under the depth refinement module.')
        parser.add_argument('--no_refined_depth_finetune', type=bool, default=False, help='# training under the depth refinement module.')
        parser.add_argument('--fstime_load_DRM', type=bool, default=False, help='# whether the first time to load DRM')
        
        # vgg parameters.
        parser.add_argument('--vgg_weights', type=int, nargs='+', default=[1.0, 1.0, 1.0], help='vgg net weights')
        # parser.add_argument('--vgg_weights', type=int, nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0], help='vgg net weights')
        # parser.add_argument('--vgg_indices', type=int, nargs='+', default=[2, 7, 12, 21, 30], help='used vgg layers, for simple three-layers loss.')
        parser.add_argument('--vgg_indices', type=int, nargs='+', default=[2, 7, 12], help='used vgg layers, for simple three-layers loss.')
        parser.add_argument('--lam_vgg', type=float, default=0.01, help='the vgg loss weight.')
        
        # options for testing rays:
        parser.add_argument('--test_rays_ui', action='store_true', help='# whether testing rays gui.')

        # other properties, important to fixed the sampling seed.
        parser.add_argument('--rand_seed', type=int, default=3130, help='# the random seed for cpu,gpu.')

        return parser

