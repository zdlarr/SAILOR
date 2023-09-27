'''
    Inference code for static rendering of performer.
'''
import sys, os
import warnings
import numpy as np
from PIL import Image
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import utils_render.util as util
from upsampling.dataset.TestDataset import RenTestDataloader
from upsampling.options.RenTestOptions import RenTestOptions
from upsampling.SRONetUp import SRONetUp
from utils_render.utils_render import gen_mesh
import torch.nn.functional as F

warnings.filterwarnings('ignore')

opts = RenTestOptions().parse()
opts.gen_mesh = True

if torch.cuda.is_available():
    os.environ["OMP_NUM_THREADS"] = "1"
    
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
else:
    deivce = torch.device('cpu')
    raise('Only support GPU testing now...')

#### 1. create the model dirs. ####
opts.model_dir = os.path.join(opts.render_model_dir, opts.name)
util.make_dir(opts.model_dir)
opts.rendering_static = True
opts.output_dir = os.path.join('./upsampling/results/', opts.data_name)
util.make_dir(opts.output_dir)
opts.ren_data_root    = './test_data/static_data0'
# opts.ren_data_root    = './test_data/static_data1'
# opts.ren_data_root    = './test_data/static_data2'

#### 2. create the dataset. ####
test_dataset     = RenTestDataloader(opts, phase=opts.phase)
num_test_dataset = len(test_dataset)
print('Number of the test pairs = {} for each epoch'.format(num_test_dataset))

#### 3. create the models.
model = SRONetUp(opts, device)
model.setup_nets()
model.set_eval()
model.set_val(True)

for idx, data in enumerate(test_dataset):
    name = data['name'][0]
    vid  = data['target_view_ids'][0]

    print( 'Rendering for data: {}, vid: {}'.format(name, vid) )
    model.set_input(data)

    if vid == 0:
        model.forward_build()
        if opts.gen_mesh:
            print( 'generate mesh ...' )
            gen_mesh(opts, model, data, device, epoch='_test')

    model.forward_query()
    rendered_result = model.get_current_visuals()
    
    rgb   = rendered_result['_rgbs'].view(opts.batch_size, opts.load_size, opts.load_size, 4, 3).detach().cpu()
    depth = rendered_result['_depths'].view(opts.batch_size, 1, opts.load_size, opts.load_size).detach().cpu()
    rgb   = rgb.view(-1, opts.load_size * opts.load_size, 12).permute(0,2,1) # [B*Nv, 12, N_p]
    rgb   = rgb[:,[0,3,6,9,1,4,7,10,2,5,8,11]]
    rgb   = F.fold(rgb, (opts.img_size, opts.img_size), (2,2), stride=2).permute(0,2,3,1)[0].numpy()

    mask = torch.zeros_like(depth)
    mask[(depth > opts.valid_depth_min) & (depth < opts.valid_depth_max)] = 1.0
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = F.interpolate(mask, scale_factor=2, mode='bilinear')[0,0].numpy()
    mask = cv2.erode(mask, erode_kernel) # erode the mat.
    mask = cv2.GaussianBlur(mask, (5,5), 0) # [H,W]
    rgb *= mask[..., None]

    # make output dir.
    output_dir = os.path.join( opts.output_dir, name )
    util.make_dir(output_dir)
    # output data
    save_rgb_path = os.path.join( output_dir, 'frame_' + name + '_' + str(vid.item()).zfill(3) + '_rgb.png' )
    Image.fromarray( (rgb * 255).astype(np.uint8) ).save(save_rgb_path)

# write to videos.
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# write video.
for name in os.listdir(opts.output_dir):
    output_dir = os.path.join( opts.output_dir, name )
    output_video_path = os.path.join(output_dir, 'rgbs_{}.avi'.format(name))
    videowriter = cv2.VideoWriter(output_video_path, fourcc, 25, (1024, 1024), True)
    frames = sorted(os.listdir(output_dir))
    for frame in frames:
        f_path = os.path.join(output_dir, frame)
        image = cv2.imread(f_path)
        videowriter.write(image)
        print(frame + " has been written!")

    videowriter.release()
    print('fin video:', name)