
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils
from models.Filters.HRNetEncoder import HRNetV2_W18_small_v2_light, HRNetV2_W18_small_v2_light_color, HRNetV2_W18_small_v2_light_depth, HRNetV2_W18_small_v2_light_v2
from models.Filters.HRNetEncoder import HRNetV2_W18_small_v2, HRNetV2_W18_small_v2_renew
from models.Filters.HGFilters import HGFilter

class DSRFeatureExtractionModel(nn.Module):
    
    def __init__(self, n_input_dim=3, n_output_dim=32):
        super(DSRFeatureExtractionModel, self).__init__()
        kernel_size = 3
        padding     = 1
        process_seq = nn.Sequential(
            nn.Conv2d(n_input_dim, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_output_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.add_module("featuring", process_seq)
    
    def forward(self, rgbs):
        # the inputs are feed into the pipeline.
        # rgbds : [B,3,H,W] and [B, 1, H, W] -> [B,8,H,W] and [B,4,H,W]. encoding resolution 1K.
        x_features = self.featuring( rgbs )
        x = torch.cat( [x_features, rgbs], dim=1 ) # 8+4 (rgbd features);
        return x

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)
    
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm2d):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class ResUnet(nn.Module):
    def __init__(self, channel=3, filters=[16, 32, 64, 128]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        # self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        # self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        # self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        # self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        # self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[2]+filters[3], 32, 1, 1)
        )

    def forward(self, x):
        # Encode
        # B, S, C, H, W = x.shape
        # x = x.view(B*S, C, H, W)
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        # x6 = self.up_residual_conv1(x5)

        # x6 = self.upsample_2(x6)
        # x7 = torch.cat([x6, x2], dim=1)

        # x8 = self.up_residual_conv2(x7)

        # x8 = self.upsample_3(x8)
        # x9 = torch.cat([x8, x1], dim=1)

        # x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x5)
        # output = output.view(B, S, 32, H//4, W//4)
        return output
    

class FeatureNet(nn.Module):
    def __init__(self, n_input_dim=3, norm_act=nn.BatchNorm2d):
        super(FeatureNet, self).__init__()
        self.conv0 = nn.Sequential(
                        ConvBnReLU(n_input_dim, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))
        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))
        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))
        feat0 = self._upsample_add(feat1, self.lat0(conv0))
        feat1 = self.smooth1(feat1)
        feat0 = self.smooth0(feat0)
        return feat2, feat1, feat0


class BasicConvs(nn.Module):
    def __init__(self, n_input_dim=1, n_output_dim=32):
        super(BasicConvs, self).__init__()
        # fully convolutional network, similar to the options in NeuralVolume.
        # [B, 64, H/4, W/4];
        self.conv = nn.Sequential( # fully convolutional net.
            nn.Conv2d(n_input_dim, 32, 5, 1, 2), nn.ReLU(), # B, 16, 1k, 1k
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),  # B, 32, 512, 512
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), # B, 32, 512, 512
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),  # B, 64, 256, 256
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(), # B, 64, 256, 256
            nn.Conv2d(128, n_output_dim, 3, 1, 1), nn.ReLU() # B, 64, 256, 256
            # nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class ImageEncoder(nn.Module):

    def __init__(self, opts, device, dim_inputs=1, output_dim=32, encoding_type='convs', encoded_data='rgb'):
        super(ImageEncoder, self).__init__()
        self.opts = opts
        self.device = device
        
        # only support the RGBD or RGB now;
        # if opts.type_input == 'rgbd':
        #     self._dim_inputs = 4
        # elif opts.type_input == 'rgb':
        #     self._dim_inputs = 3
        # else:
        #     raise Exception('Not supported input dim')
        self._dim_inputs  = dim_inputs
        self._dim_outputs = output_dim
        
        if encoding_type == 'convs':
            # self.conv = DSRFeatureExtractionModel(n_input_dim=self._dim_inputs, n_output_dim=self._dim_outputs).to(self.device)
            self.conv = FeatureNet(n_input_dim=self._dim_inputs).to(self.device)
        elif encoding_type == 'hrnet':
            if encoded_data == 'rgb':
                self.conv = HRNetV2_W18_small_v2_light_color(opts=opts, n_input_dim=self._dim_inputs,
                                                        n_output_dim=self._dim_outputs, last_layer=False).to(self.device) # Light HRNet w18.
            elif encoded_data == 'depth':
                self.conv = HRNetV2_W18_small_v2_light_depth(opts=opts, n_input_dim=self._dim_inputs,
                                                        n_output_dim=self._dim_outputs, last_layer=False).to(self.device) # Light HRNet w18.
            elif encoded_data == 'rgbd':
                self.conv = HRNetV2_W18_small_v2(opts=opts, n_input_dim=self._dim_inputs,
                                                 n_output_dim=self._dim_outputs, last_layer=False).to(self.device) # Light HRNet w18.
            else:
                raise Exception('Not supported encoding data.')
        elif encoding_type == 'hourglass':
            self.conv = HGFilter(opt=opts, nc_input=self._dim_inputs).to(self.device)
        else:
            raise Exception('Not supported encoding method.')

    def forward(self, xs):
        # xs : input RGB images : [B * N, 3, H, W]
        feats = self.conv(xs) # [B*N, C, H', W'] e.g., H' = H / 4, W' = W / 4;
        return feats


if __name__ == '__main__':
    from options.GeoTrainOptions import GeoTrainOptions
    from options.RenTrainOptions import RenTrainOptions
    import time
    
    from torch2trt import torch2trt
    opts = RenTrainOptions().parse()
    encoder = ImageEncoder(opts, 'cuda:0', dim_inputs=4, encoding_type='hrnet', encoded_data='rgbd').eval().cuda()
    x = torch.randn([4, 4, 512, 512]).to('cuda:0') # Batch = 1; for 3 views.
    # encoder = FeatureNet(n_input_dim=4).cuda().eval()
    # encoder_trt = torch2trt(encoder, [x], max_batch_size=4, fp16_mode=False) # transform the encoder to tensorrt.
    # print(encoder_trt)
    # x = torch.randn([1, 3, 1024, 1024]).to('cuda:0') # Batch = 3.
    t = 0; error = 0;
    for _ in range(0,20):
        with torch.no_grad():
            t1 = time.time()
            output = encoder(x)
            torch.cuda.synchronize()
            t2 = time.time()
            t += t2 - t1
            # output2 = encoder(x)
        # ft0 = output[0].view(-1, 3, 16, 256, 256)
        # print(t2 - t1, output[0].shape, output[-1].shape, ft0.shape)
        # for i in range(len(output)):
        #     print(output[i].shape)
        # err = torch.mean(output[2] - output2[2])
        # error += err
        # print(output.shape)
        
    print(t / 20, error / 20) # 12ms 4 * 1k rgb here.
        