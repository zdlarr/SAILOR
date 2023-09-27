
import sys, os
import argparse

class GeoTrainOptions(object):

    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):
        
        # parsing items, basic options.
        parser.add_argument('--name', type=str, default='test', help='# name of our model.')
        parser.add_argument('--data_root', type=str, default='/home/ssd2t/dz/render_dataset2', help='# the root dir of the dataset.')
        parser.add_argument('--real_data_root', type=str, default='/home/ssd2t/dz/render_dataset3', help='# the root dir of the real dataset.')
        parser.add_argument('--model_dir', type=str, default='./checkpoints', help='# the model dir.')
        parser.add_argument('--save_dir', type=str, default='./results', help='# the save dir.')
        # options for training.
        parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='[0], [0,1].., [-1]:CPU')
        parser.add_argument('--is_train', type=bool, default=True, help='# whether is training.')
        parser.add_argument('--is_val', type=bool, default=False, help='# whether is validating.')
        parser.add_argument('--continue_train', action='store_true', default=False, help='# whether continue training.')
        parser.add_argument('--num_epoch', type=int, default=100, help='# num of training epoches.')
        parser.add_argument('--epoch', type=str, default='latest', help='# the epoch loaded for network.')
        parser.add_argument('--resume_epoch', type=int, default=-1, help='# the resume epoch for training.')
        parser.add_argument('--freq_plot', type=int, default=20, help='# the frequency of print logs.')
        parser.add_argument('--freq_save_ply', type=int, default=50, help='# epoch for saving the ply file.')
        parser.add_argument('--freq_save_latest', type=int, default=200, help='# epoch for saving the latest model.')
        parser.add_argument('--freq_save_epoch', type=int, default=4, help='# epoch for saving the iter model.')
        parser.add_argument('--freq_val', type=int, default=1, help='# epoch for validating.')
        parser.add_argument('--finetune_epoch', type=int, default=80, help='# epoch for finetuning our model.')
        parser.add_argument('--only_val', type=bool, default=False, help='# whether only validation.')
        parser.add_argument('--net_init_gain', type=float, default=0.02, help='# the frequency for plotting rendering info.')
        parser.add_argument('--default_init_params', type=bool, default=True, help='# init the network using default method.')

        # options for validating.
        parser.add_argument('--check_num_val', type=bool, default=True, help='# whether checking number val.')
        parser.add_argument('--check_gen_mesh', type=bool, default=True, help='# whether generating mesh during validating.')
        parser.add_argument('--random_choice_mesh_val', type=bool, default=True, help='# whether randomly choice the mesh for generating mesh.')
        parser.add_argument('--num_gen_mesh', type=int, default=1, help='# num of the generated mesh when validating.')
        parser.add_argument('--resolution', type=int, default=256, help='# the resolution of the generated mesh.')

        # dataset's options.
        parser.add_argument('--local_rank', default=-1)
        parser.add_argument('--batch_size', type=int, default=4, help='# num of our batch size.')
        parser.add_argument('--num_threads', type=int, default=8, help='# num of the threadings.')
        parser.add_argument('--serial_batches', action='store_true', help='# num of the batches.')
        parser.add_argument('--pin_memory', default=True, action='store_true', help='pin_memory')
        parser.add_argument('--num_views', type=int, default=3, help='# num of the deafult views.')
        parser.add_argument('--project_mode', type=str, default='perspective', help='# the type of the projection mode.')
        parser.add_argument('--bbox_min', type=int, nargs='+', default=[-1.28, -1.28, -1.28])
        parser.add_argument('--bbox_max', type=int, nargs='+', default=[1.28, 1.28, 1.28])
        parser.add_argument('--load_size', type=int, default=512, help='# the loading size.')
        parser.add_argument('--face_load_size', type=int, default=512, help='# the loading facial size.')
        parser.add_argument('--num_sample_geo', type=int, default=4000, help='# sampling number of the points for geo.')
        parser.add_argument('--num_sample_face', type=int, default=4000, help='# sampling number of the points for face.')
        parser.add_argument('--max_depth', type=float, default=1000, help='# the default max depth.')
        parser.add_argument('--z_size', type=float, default=1.5, help='# z normalization factor: distance from cam to object center.')
        parser.add_argument('--z_bbox_len', type=float, default=1, help='# the length for z normalization;')
        parser.add_argument('--depth_normalize', type=str, default='adaptive', help='# normalization method for depth, exp or linear or adaptive')
        # augmentaion for dataset.
        parser.add_argument('--no_random_shift', action='store_true', default=False, help='# check if no randomly shift applied.')
        parser.add_argument('--aug_alstd', type=float, default=0.0, help='# augmentation pca lighting alpha std')
        parser.add_argument('--aug_bri', type=float, default=0.04, help='# augmentation brightness')
        parser.add_argument('--aug_con', type=float, default=0.04, help='# augmentation contrast')
        parser.add_argument('--aug_sat', type=float, default=0.04, help='# augmentation saturation')
        parser.add_argument('--aug_hue', type=float, default=0.04, help='# augmentation hue')
        parser.add_argument('--aug_blur', type=float, default=1.8, help='# augmentation blur')
        parser.add_argument('--aug_blur_face', type=float, default=0, help='# augmentation blur for face;')

        # net's basic options.
        parser.add_argument('--net_init_type', type=str, default='normal', help='# default network init type:normal, xavier, kaiming, orthogonal as soon;')
        parser.add_argument('--optimizer_type', type=str, default='adam', help='# the optimizer: adam, rmsprop')
        parser.add_argument('--loss_type', type=str, default='bce', help='# type of criterion method, e.g, bce, mse;')
        parser.add_argument('--lr', type=float, default=1e-3, help='# the learning rate.')
        parser.add_argument('--lr_fine_tune', type=float, default=1e-4, help='# the learning rate for fine-tuning model.')
        parser.add_argument('--beta1', type=float, default=0.99, help='# the beta1 parameter for ADAM.')
        parser.add_argument('--beta2', type=float, default=0.999, help='# the beta2 parameter for ADAM.')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='# the weight_decay parameter for ADAM.')
        
        # geo net's options.
        # rgbd input, defaults using rgbd 4 channels' data.
        parser.add_argument('--name_geo_net', type=str, default='TransGeoNet', help='# the name of the Geometry Net. GeoNet, GeoNetPSDF, GeoNetwFace, TransGeoNet;')
        parser.add_argument('--type_input', type=str, default='rgbd', help='# input type: depth_only, rgb, rgbd')
        parser.add_argument('--save_depths_normals_npy', type=bool, default=False, help='whether to save the depths & normals.')
        parser.add_argument('--save_depths_ply', type=bool, default=False, help='whether to save the depths ply.')
        parser.add_argument('--type_filter', type=str, default='hrnet', help='# the type of the feature filter. e.g, hourglass, hrnet')
        parser.add_argument('--truncated_z', type=float, default=0.01, help='# the truncated z as 0.01m. default as (0.01 / (128 / 100) = 0078125')
        parser.add_argument('--truncated_refined_depth', type=bool, default=False, help='# whether truncated refined depth.')
        # Depth thres is the parameters used to control the depth mask maps.
        parser.add_argument('--depth_edge_thres', type=float, default=0.06, help='# the depth discontinous boundary mask threshold')
        parser.add_argument('--face_depth_edge_thres', type=float, default=0.04, help='# the depth discontinous face boundary mask threshold')
        parser.add_argument('--using_visual_hull', type=bool, default=True, help='# whether using visual hull to build mesh.')
        # truncate_z: 1m -> [-1,1]; truncated values as 2 cm here.
        # the truncated values: 0.04; when inputting with normal vector, no truncated_z here.
        # for face filter's option.
        parser.add_argument('--type_filter_face', type=str, default='hourglass', help='# the type of the filter for face. e.g., hourglass, hrnet.')
        parser.add_argument('--face_border_th', type=float, default=0.1, help='# the border threshold for face detector.')
        parser.add_argument('--load_pifu_face_hourglass', type=bool, default=True, help='# whether load the facial pifu model.')
        parser.add_argument('--pifu_face', type=bool, default=True, help='# whether using the facial pifu model.')
        parser.add_argument('--feature_woface', type=bool, default=False, help='# whether using the facial pifu model.')
        parser.add_argument('--save_face_weights', type=bool, default=False, help='# whether saving facial fusion weights.')
        
        # for surface smooth loss;
        parser.add_argument('--perturb_surface_pts_body', type=float, default=0.004, help='# the weight of the surface smooth weights, default as 1cm -> 2mm')
        parser.add_argument('--perturb_surface_pts_face', type=float, default=0.001, help='# the weight of the surface smooth weights, default as 1mm')
        parser.add_argument('--reg_weights', type=float, default=800, help='the weight of the smooth loss. The smooth loss is only used to update fusioned MLP.')

        # hourglass's options.
        parser.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        parser.add_argument('--hourglass_dim', type=int, default=256, help='256 | 512')
        parser.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
        parser.add_argument('--num_stack', type=int, default=4, help='# the number of the stack hourglass.')
        #### the norm for the existed filters;
        parser.add_argument('--filter_norm', type=str, default='group', help='# the norm type of the filter net.')
        parser.add_argument('--mlp_norm', type=str, default='group', help='# the norm type of the MLP, !! now we dont use norm in MLP.')
        # MLP's parsing options.
        parser.add_argument('--filter_channels', type=int, nargs='+', default=[128 + 4] + [128, 128, 128, 128, 128] * 2 + [1], help='# the channels of the decoder MLP.')
        parser.add_argument('--mlp_res_layers', type=int, nargs='+', default=range(1, 12, 1), help='# mlp residual layers for concat features.')
        parser.add_argument('--merge_layer', type=int, default=5, help='# the layer that merged the multi-view features.')
        # Face mlp's parsing options.
        parser.add_argument('--face_filter_channels', type=int, nargs='+', default=[128 + 64 + 4] + [128, 128, 128, 128] * 2 + [1], help='# the channels of the face decoder MLP, 64 + 32 for merging')
        parser.add_argument('--face_mlp_res_layers', type=int, nargs='+', default=range(1, 10, 1), help='# residual layers for the face mlp.')
        parser.add_argument('--face_merge_layer', type=int, default=4, help='# the layer that merged the multi-view feats for faces.')
        parser.add_argument('--face_scale', type=float, default=1.0, help='# the face resolution scale for training.')
        parser.add_argument('--only_update_face', type=bool, default=False, help='# whether only update face part.')
        parser.add_argument('--only_optimize_pifu', type=bool, default=False, help='# whether only update pifu part.')
        parser.add_argument('--only_optimize_drm', type=bool, default=False, help='# whether only update drm part.')
        parser.add_argument('--only_optimize_mlp', type=bool, default=False, help='# whether only update mlp part.')
        parser.add_argument('--only_optimize_body', type=bool, default=False, help='# whether only update body parts.')
        parser.add_argument('--only_optimize_drm_body', type=bool, default=False, help='# whether only update body & drm parts.')
        parser.add_argument('--no_update_swinir', type=bool, default=False, help='# whether only update swinir.')
        
        parser.add_argument('--update_drm', type=bool, default=False, help='# whether update the depth refine module.')
        parser.add_argument('--drm_basic_num_channels', type=int, default=32, help='# the basic num of the feature channel.')
        # merge options.
        parser.add_argument('--merge_method', type=str, default='distance', help='# method for merging features. visible, distance, mean.')
        parser.add_argument('--encoding_type', type=str, default='hrnet', help='# Image Encoding methods.')

        # depth to normal map.
        parser.add_argument('--depth2normal_type', type=str, default='pcdifference', help='# the type of depth to normal method. pcdifference or imdifference')
        parser.add_argument('--normal_map_scale', type=float, default=256.0, help='# the scale factor of the normal map (multiplied on normal).')
        
        # depth refinement & depth augmentation.
        parser.add_argument('--scale_factor', type=float, default=21.4, help='# the scale_factor of the depth aug.')
        parser.add_argument('--baseline_m', type=float, default=0.025, help='# the base_line of the kinect camera.')
        parser.add_argument('--invalid_disp', type=float, default=999999999.9, help='# the invalid value of the disparity value.')
        parser.add_argument('--holes_rate', type=float, default=0.005, help='# the rates for adding holes.')
        parser.add_argument('--sigma_d', type=float, default=0.01, help='# the sigma for the depth.')
        parser.add_argument('--support_depth_refine', type=bool, default=True, help='# check if support depth refinement.')
        parser.add_argument('--smooth_loss_available', type=bool, default=False, help='# check if support smooth loss.')

        # feature aggregation modules.
        parser.add_argument('--att_input_dim', type=int, default=84, help='# num of the attention input feature dim.')
        parser.add_argument('--att_num_heads', type=int, default=8, help='# num of the attention heads.')
        parser.add_argument('--att_num_layers', type=int, default=6, help='# num of the transformer layers')

        # 2D feature to 3D.
        parser.add_argument('--vol_dim', type=int, default=64, help='# the dim of the volume.')
        parser.add_argument('--volume_ft_dim', type=int, default=64, help='# the feature dim of the 3D volume.')

        #### the 3rdparties options:
        parser.add_argument('--face_detector_dir', type=str, default='./thirdparties/RetinaFace/', help='# The root dir for face detection.')
        parser.add_argument('--face_th', type=float, default=0.7, help='# threshold for the face detection.')
        parser.add_argument('--part_seg_config_path', type=str, default='/home/yons/my_Rendering/DensePoseFnL/configs/densepose_rcnn_R_101_FPN_s1x.yaml', help='# the path of the config for partseg.')
        parser.add_argument('--part_seg_model_path', type=str, default='/home/yons/my_Rendering/DensePoseFnL/models/model_final_ad63b5.pkl', help='# the path of the partseg model.')
        
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            self.parser = parser
            
        assert self.parser is not None, 'not arg parser initialized.'
        
        return self.parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    
    def parse(self):
        opts = self.gather_options()
        return opts