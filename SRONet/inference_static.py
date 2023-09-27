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
from SRONet.dataset.TestDataset import RenTestDataloader
from SRONet.options.RenTestOptions import RenTestOptions
from SRONet.SRONet import SRONet
from utils_render.utils_render import gen_mesh

warnings.filterwarnings('ignore')

opts = RenTestOptions().parse()
opts.gen_mesh = False

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
util.make_dir('./SRONet/results/')
opts.output_dir = os.path.join('./SRONet/results/', opts.data_name)
util.make_dir(opts.output_dir)
opts.ren_data_root    = './test_data/static_data0'

#### 2. create the dataset. ####
test_dataset     = RenTestDataloader(opts, phase=opts.phase)
num_test_dataset = len(test_dataset)
print('Number of the test pairs = {} for each epoch'.format(num_test_dataset))

#### 3. create the models.
model = SRONet(opts, device)
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

    rgb   = rendered_result['_rgbs'].view(opts.batch_size, opts.img_size, opts.img_size, 3).detach().cpu()
    depth = rendered_result['_depths'].view(opts.batch_size, opts.img_size, opts.img_size).detach().cpu()
    
    # make output dir.
    output_dir = os.path.join( opts.output_dir, name )
    util.make_dir(output_dir)
    # output data
    save_rgb_path = os.path.join( output_dir, 'frame_' + name + '_' + str(vid.item()).zfill(3) + '_rgb.png' )
    Image.fromarray( (rgb[0].numpy() * 255).astype(np.uint8) ).save(save_rgb_path)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# write video.
for name in os.listdir(opts.output_dir):
    output_dir = os.path.join( opts.output_dir, name )
    output_video_path = os.path.join(output_dir, 'rgbs_{}.avi'.format(name))
    videowriter = cv2.VideoWriter(output_video_path, fourcc, 25, (512, 512), True)
    frames = sorted(os.listdir(output_dir))
    for frame in frames:
        f_path = os.path.join(output_dir, frame)
        image = cv2.imread(f_path)
        videowriter.write(image)
        print(frame + " has been written!")

    videowriter.release()
    print('fin video:', name)