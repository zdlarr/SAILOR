"""
    Classic Surface Classifier
    - full layer MLP construction.
    - MLP with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks import get_norm_layer
from models.utils import SirenLayer, DenseLayer

class SurfaceClassifier(nn.Module):

    def __init__(self, filter_channels, num_views=1, 
                       merge_layer=-1,
                       res_layers=[],
                       act_func=nn.LeakyReLU(),
                       use_siren = False,
                       last_op=None):

        super(SurfaceClassifier, self).__init__()

        self.filters     = nn.ModuleList()
        self.num_views   = num_views
        self.res_layers  = res_layers
        # self.norm        = norm
        self.norms       = []
        # notice that layers after the merged layers can not be added the resblocks.
        self.merge_layer = merge_layer if merge_layer > 0 else len(filter_channels) // 2 - 1
        self.act_func    = act_func
        self.last_op     = last_op

        for l in range(0, len(filter_channels) - 1):
            if l in self.res_layers:
                in_channel = filter_channels[l] + filter_channels[0]
            else:
                in_channel = filter_channels[l]
                
            if use_siren:
                self.filters.append(
                    SirenLayer(in_channel, filter_channels[l+1], 1, is_first=(l==0), 
                               activation=True if l<len(filter_channels) - 2 else False)
                )
            else:
                self.filters.append(
                    DenseLayer(in_channel, filter_channels[l+1], 1, 
                               activation=self.act_func if l<len(filter_channels) - 2 else None)
                )
    
    def forward0(self, features):
        # extract features of the input image.
        # features: [B * N_v, C, N_points]
        # output: [B * N_v, C', N_points]
        y        = features
        input_ft = features
        feat_geo = None

        for i, f in enumerate(self.filters):
            y = f(y if i not in self.res_layers else torch.cat([y, input_ft], dim=1))

            # if i != len(self.filters) - 1:
            #     y = self.act_func(y)

            if i == self.merge_layer: # the merged layer.
                feat_geo = y
                break

        # no gradient canceling.
        return feat_geo

    def forward1(self, features):
        # feed the last feature into network.
        y = features
        input_ft = features

        for i in range(self.merge_layer + 1, len(self.filters)):
            f = self.filters[i]
            y = f(y if i not in self.res_layers else torch.cat([y, input_ft], dim=1))
            
            # if i != len(self.filters) - 1:
            #     y = self.act_func(y)
            
        if self.last_op is not None:
            y = self.last_op(y)
        
        return y

    def forward(self, features):
        y        = features
        input_ft = features
        feat_geo = None

        for i, f in enumerate(self.filters):
            y = f(y if i not in self.res_layers else torch.cat([y, input_ft], dim=1))

            # if i != len(self.filters) - 1:
            #      y = self.act_func(y)

            if i == self.merge_layer: # the merged layer.
                feat_geo = y.clone()
                
        if self.last_op is not None:
            y = self.last_op(y)
        
        return y, feat_geo


if __name__ == '__main__':
    classifier = SurfaceClassifier(filter_channels=[257] + [128, 128, 128, 128, 128] * 2 + [1], 
                                   res_layers=range(1, 12, 1),
                                   last_op=nn.Sigmoid(), use_siren=True)
    
    t = torch.randn([20, 257, 1000])
    out, feat_geo = classifier(t)
    # feat_geo = classifier.forward0(t) # [B, 128, 1000];
    # out = classifier.forward1(feat_geo) # [B, 1, 1000]
    # out = classifier.forward(feat_geo)
    print(classifier)
    print(out.shape)
