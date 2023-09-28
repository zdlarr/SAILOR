
import time
import torch
from torch2trt import torch2trt,TRTModule
from options.RenTrainOptions import RenTrainOptions
import utils_render.util as util
# two depth-refinement network.
from depth_denoising.net import BodyDRM2, BodyDRM3, DepthRefineModule

opts = RenTrainOptions().parse()
opts.test_tensorRT = False
opts.num_gpus = 2

# the network based on hrnet structure,
# batch_size = 2 here.
if not opts.test_tensorRT:
    # original version.
    drm_path = './checkpoints_rend/SAILOR/latest_model_BodyDRM2.pth'
    bodydrm = BodyDRM2(opts, 'cuda:0').to('cuda:0').eval()
    
    # a lighter version.
    # drm_path = './checkpoints_rend/SAILOR/latest_model_BodyDRM3.pth'
    # bodydrm = BodyDRM3(opts, 'cuda:0').to('cuda:0').eval()
    
    bodydrm.load_state_dict(torch.load(drm_path), strict=False)
    bodydrm_trt = TRTModule()
    util.make_dir('./accelerated_models')
    print('transform to tensorrt model...')
    if opts.num_gpus == 2: # the batch-size is 2 here.
        x = torch.randn([2,5, 512, 512], requires_grad=False).to('cuda:0')
        bodydrm_trt = torch2trt(bodydrm, [x], max_batch_size=2, fp16_mode=True)

        torch.save(bodydrm_trt.state_dict(), './accelerated_models/depth_refine_trt_parallel_v0.pth')
        # torch.save(bodydrm_trt.state_dict(), './accelerated_models/depth_refine_trt_parallel_v1.pth')
    elif opts.num_gpus == 1:
        x = torch.randn([4,5, 512, 512], requires_grad=False).to('cuda:0')
        bodydrm_trt = torch2trt(bodydrm, [x], max_batch_size=4, fp16_mode=True)

        torch.save(bodydrm_trt.state_dict(), './accelerated_models/depth_refine_trt_parallel_v0_big.pth')
        # torch.save(bodydrm_trt.state_dict(), './accelerated_models/depth_refine_trt_parallel_v1_big.pth')
        
else:
    bodydrm_trt = TRTModule()
    x = torch.randn([2,5, 512, 512], requires_grad=False).to('cuda:0')
    bodydrm_trt.load_state_dict(torch.load('./accelerated_models/depth_refine_trt_parallel_v0.pth'))
    # bodydrm_trt.load_state_dict(torch.load('./accelerated_models/depth_refine_trt_parallel_v1.pth'))

    # test inference time.
    t0 = time.time()

    for _ in range(500):
        output = bodydrm_trt(x)

    t1 = time.time()
    print(output.shape, (t1 - t0) / 500.0)