
import time
import torch
from torch2trt import torch2trt,TRTModule
from options.RenTrainOptions import RenTrainOptions
import utils_render.util as util
from depth_denoising.net import HRNetUNet

opts = RenTrainOptions().parse()
opts.test_tensorRT = False
opts.num_gpus = 2

# batch_size = 2 here.
if not opts.test_tensorRT:
    human_nerf_path = './checkpoints_rend/SAILOR/latest_model_BasicRenNet.pth'
    filter_2d = HRNetUNet(opts, 'cuda:0').to('cuda:0').eval()
    
    state_dict = torch.load(human_nerf_path)
    state_dicts_ = {}
    filter_2d_dict = filter_2d.state_dict()

    for k, v in state_dict.items():
        module_names = k.split('.')
        if module_names[0] == 'filter_2d':
            key_ = '.'.join(module_names[1:])
            state_dicts_[key_] = v
    filter_2d_dict.update(state_dicts_)
    filter_2d.load_state_dict( state_dicts_, strict=True )
    print('fin load hrnetunet model..')
    # process half of the data.

    print('transform to tensorrt model...')
    util.make_dir('./accelerated_models')
    filter_2d_trt = TRTModule()
    if opts.num_gpus == 2: # the batch-size is 2 here.
        x = torch.randn([2, 3, 512, 512], requires_grad=False).to('cuda:0')
        y = torch.randn([2, 1, 512, 512], requires_grad=False).to('cuda:0')
        filter_2d_trt = torch2trt(filter_2d, [x,y], max_batch_size=2, fp16_mode=True)
        torch.save(filter_2d_trt.state_dict(), './accelerated_models/hrunet_trt_parallel.pth')
    elif opts.num_gpus == 1: # the batch-size is 4 here.
        x = torch.randn([4, 3, 512, 512], requires_grad=False).to('cuda:0')
        y = torch.randn([4, 1, 512, 512], requires_grad=False).to('cuda:0')
        filter_2d_trt = torch2trt(filter_2d, [x,y], max_batch_size=4, fp16_mode=True)
        torch.save(filter_2d_trt.state_dict(), './accelerated_models/hrunet_trt_parallel_big.pth')
    
else:
    filter_2d_trt = TRTModule()
    x = torch.randn([2, 3, 512, 512], requires_grad=False).to('cuda:0')
    y = torch.randn([2, 1, 512, 512], requires_grad=False).to('cuda:0')
    filter_2d_trt.load_state_dict(torch.load('./accelerated_models/hrunet_trt_parallel.pth'))

    # test inference time.
    t0 = time.time()

    for _ in range(500):
        output0, output1 = filter_2d_trt(x,y)

    t1 = time.time()
    print(output0.shape, output1.shape, (t1 - t0) / 500.0)