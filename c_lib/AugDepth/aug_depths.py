# by zheng dong 

import numpy as np
import cv2
from c_lib.AugDepth import DepthAug
import matplotlib.pyplot as plt

def augment_depth_maps_v0(depth, mask):
    dot_pattern  = cv2.imread('./c_lib/AugDepth/data/kinect-pattern_3x3.png', 0).astype(np.float32)
    K = np.array([[320,0,256],[0,320,256],[0,0,1]]).astype(np.float32)
    aug_depth_map = np.copy(depth).astype(np.float32)
    depths_blur = cv2.GaussianBlur(depth, (3,3), 1.0)
    # last three params: z_size, holes_rate, sigma_d
    DepthAug.aug_depth(depths_blur.astype(np.float32), mask.astype(np.float32), dot_pattern, K, aug_depth_map,
                       21.4, 0.025, 999999999.9, 1.4, 0.004, 0.004)
    return aug_depth_map


def augment_depth_maps_v1(depth, mask):
        # input depth : [h,w], mask : [h,w]
        # 1. erode borders.
        img_size = depth.shape[0]
        randv0 = np.random.rand()
        if_erode = True
        if randv0 > 0.6:
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        elif randv0 > 0.4 and randv0 <= 0.6:
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        elif randv0 > 0.2 and randv0 <= 0.4:
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        else:
            if_erode = False
            
        # 1. add holes.
        mask_holes = np.copy(mask)
        # : get the mask regions.
        if np.random.rand() > 0.5:
            h_idx, w_idx = np.where(mask != 0) # get the non-empty pixels.
            # get random idx.
            valid_mask_size = h_idx.size
            holes_rate = 0.001 + np.random.rand() * (0.03 - 0.001) # [0.001,0.02]
            rand_idx = np.random.randint( 0, valid_mask_size, size=[ int(valid_mask_size * holes_rate) ] ) # [num_rays]
            sampled_x = w_idx[rand_idx] # get the sampled idx.
            sampled_y = h_idx[rand_idx] # get the sampled idx.
            for i in range(sampled_x.size): # add holes in each 
                idx_x = sampled_x[i]; idx_y = sampled_y[i];
                randv_ = np.random.rand()
                if randv_ > 0.7:
                    k_size = 5
                elif randv_ > 0.3 and randv_ <= 0.7:
                    k_size = 3
                else:
                    k_size = 1
                min_y = idx_y - k_size // 2 if idx_y - k_size // 2 >= 0 else 0
                max_y = idx_y + k_size // 2 + 1 if idx_y + k_size // 2 + 1 <= img_size else img_size
                min_x = idx_x - k_size // 2 if idx_x - k_size // 2 >= 0 else 0
                max_x = idx_x + k_size // 2 + 1 if idx_x + k_size // 2 + 1 <= img_size else img_size
                for j in range(min_y, max_y):
                    for k in range(min_x, max_x):
                        mask_holes[j,k] = 0 # assign with 0;
        
        # 2. noised depths.
        # step 1. add gaussian blur.
        randv = np.random.rand()
        if randv > 0.8:
            depth = cv2.GaussianBlur(depth, (7,7), 0) # gaussian blur may effect the borders.
        elif randv > 0.5 and randv <= 0.8:
            depth = cv2.GaussianBlur(depth, (5,5), 0) # gaussian blur may effect the borders.
        elif randv > 0.2 and randv <= 0.5:
            depth = cv2.GaussianBlur(depth, (3,3), 0) # gaussian blur may effect the borders.
        # step 2. add gaussian noises
        randv1 = np.random.rand()
        if randv1 > 0.7:
            depth += np.random.normal(0, 0.015, depth.shape) # 2cm depth noise, large noises.
        elif randv1 > 0.3 and randv1 <= 0.7:
            depth += np.random.normal(0, 0.0075, depth.shape) # 1cm depth noise
        else:
            depth += np.random.normal(0, 0.001, depth.shape) # 0.1cm depth noise, a little noises here.
        
        if if_erode:
            mask_eroded = cv2.erode(mask, erode_kernel)
            depth *= mask_eroded * mask_holes # masked the depth.
        else:
            depth *= mask * mask_holes
        
        return depth
    
def augment_depth_maps(depth, mask):
    if np.random.rand() > 0.5:
        return augment_depth_maps_v0(depth, mask)
    else:
        return augment_depth_maps_v1(depth, mask)
    

if __name__ == '__main__':
    
    # resize to 512 * 512
    depth = cv2.imread('./c_lib/AugDepth/test_data/depth/0_0.png', -1) / 10000.0
    depth = cv2.resize(depth, (512,512), interpolation=cv2.INTER_NEAREST)
    mask = cv2.imread('./c_lib/AugDepth/test_data/mask/0_0.png', -1) / 256
    mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_NEAREST)

    aug_depth = augment_depth_maps_v0(depth, mask)
    # aug_depth = augment_depth_maps_v1(depth, mask)
    print(np.max(aug_depth))
    plt.imshow(aug_depth * 10000)
    plt.savefig('./c_lib/AugDepth/noise_depth_0_0_plt.png')
    plt.show()
    cv2.imwrite('./c_lib/AugDepth/noise_depth_0_0.png', (aug_depth * 10000).astype(np.uint16))

