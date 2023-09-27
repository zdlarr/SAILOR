
import time
import torch
from torch2trt import torch2trt,TRTModule
from upsampling.options.RenTrainOptions import RenTrainOptions
import utils_render.util as util
from depth_denoising.net import FeatureNet

opts = RenTrainOptions().parse()
opts.test_tensorRT = False

if not opts.test_tensorRT:
    human_nerf_path = './checkpoints_rend/SAILOR/latest_model_BasicRenNet.pth'
    filter_2d_high_res = FeatureNet().to('cuda:0').eval()
    
    state_dict = torch.load(human_nerf_path)
    state_dicts_ = {}
    filter_2d_dict = filter_2d_high_res.state_dict()
    
    for k, v in state_dict.items():
        module_names = k.split('.')
        if module_names[0] == 'filter_2d_high_res':
            key_ = '.'.join(module_names[1:])
            state_dicts_[key_] = v
    filter_2d_dict.update(state_dicts_)
    filter_2d_high_res.load_state_dict( filter_2d_dict, strict=True )
    print('fin load high-res unet model..')
    filter_2d_high_res_trt = TRTModule()
    x = torch.randn([4, 3, 1024, 1024], requires_grad=False).to('cuda:0')
    
    print('transform to tensorrt model...')
    filter_2d_high_res_trt = torch2trt(filter_2d_high_res, [x], max_batch_size=4, fp16_mode=True)
    util.make_dir('./accelerated_models')
    torch.save(filter_2d_high_res_trt.state_dict(), './accelerated_models/high_res_unet_trt.pth')
else:
    filter_2d_high_res_trt = TRTModule()
    x = torch.randn([4, 3, 1024, 1024], requires_grad=False).to('cuda:0')
    filter_2d_high_res_trt.load_state_dict(torch.load('./accelerated_models/high_res_unet_trt.pth'))

    # test inference time.
    t0 = time.time()

    for _ in range(500):
        output = filter_2d_high_res_trt(x)

    t1 = time.time()
    print(output.shape, (t1 - t0) / 500.0)