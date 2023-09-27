
import time
import torch
from torch2trt import torch2trt,TRTModule
from options.RenTrainOptions import RenTrainOptions
import utils_render.util as util
from depth_denoising.net import HRNetUNet

opts = RenTrainOptions().parse()
opts.test_tensorRT = False

# batch_size = 2 here.
if not opts.test_tensorRT:
    human_nerf_path = './checkpoints_rend/SAILOR/latest_model_BasicRenNet.pth'
    filter_2d = HRNetUNet(opts, 'cuda:0').to('cuda:0').eval()
    
    filter_2d.load_state_dict(torch.load(human_nerf_path), strict=False)
    # process half of the data.
    filter_2d_trt = TRTModule()
    x = torch.randn([2, 3, 512, 512], requires_grad=False).to('cuda:0')
    y = torch.randn([2, 1, 512, 512], requires_grad=False).to('cuda:0')
    
    filter_2d_trt = torch2trt(filter_2d, [x,y], max_batch_size=2, fp16_mode=True, strict_type_constraints=False)
    util.make_dir('./accelerated_models')
    torch.save(filter_2d_trt.state_dict(), './accelerated_models/hrunet_trt_parallel.pth')
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