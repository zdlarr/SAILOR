import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, pe=False, no_residual=True, activation=None, last_op=None, geometric_init=True):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        self.filter_channels = filter_channels
        self.last_op = last_op
        self.acti = activation if activation is not None else nn.ReLU()
        # whether using geometric_init method to initialize the MLPs.
        self._geometric_init = geometric_init
        self._bias = 0.6
        self._pe  = pe
        
        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Linear(
                    filter_channels[l],
                    filter_channels[l + 1]
                ))
                lin = self.init_weights(self.filters[l], l, filter_channels[l], filter_channels[l+1])
                self.add_module( "linear%d" % l, lin )
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    in_dim  = filter_channels[l] + filter_channels[0]
                    out_dim = filter_channels[l + 1]
                else:
                    in_dim  = filter_channels[l]
                    out_dim = filter_channels[l + 1]
                # [C, 128], [128 + C, 128], [128 + C, 128], .. [C] -> 64 + 39 + 1
                self.filters.append( nn.Linear( in_dim, out_dim ) )

                lin = self.init_weights(self.filters[l], l, in_dim, out_dim)
                self.add_module( "linear%d" % l, lin )
                
    def init_weights(self, layer, l, in_dim, out_dim):
        if self._geometric_init:
            if l == len(self.filter_channels) - 2:
                torch.nn.init.normal_( layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001 )
                torch.nn.init.constant_( layer.bias, -self._bias )
            elif self._pe and l == 0:
                torch.nn.init.constant_( layer.bias, 0.0 )
                torch.nn.init.constant_( layer.weight[:, 3:], 0.0 ) # for original positions.
                torch.nn.init.normal_( layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            elif (not self.no_residual) and self._pe and l != 0 and l != len(self.filter_channels) - 2: 
                torch.nn.init.constant_( layer.bias, 0.0 )
                torch.nn.init.normal_( layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                torch.nn.init.constant_( layer.weight[:, -(self.filter_channels[0] - 3):], 0.0) # support pe.
            else:
                torch.nn.init.constant_( layer.bias, 0.0)
                torch.nn.init.normal_( layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
        else:
            torch.nn.init.constant_( layer.bias, 0.0)
            # torch.nn.init.normal_( layer.weight, 0.0, 0.02)
            torch.nn.init.kaiming_normal_( layer.weight )
                
        # layer = nn.utils.weight_norm(layer)
        # print( 'initialize network {}'.format(type(self).__name__) )
        
        return layer

    def forward(self, feature):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, _ in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['linear' + str(i)](y) # the linear layer.
            else:
                if self._geometric_init:
                    y = self._modules['linear' + str(i)]( y if i == 0 else torch.cat([y, tmpy], -1) / np.sqrt(2))
                else:
                    y = self._modules['linear' + str(i)]( y if i == 0 else torch.cat([y, tmpy], -1) )
            if i != len(self.filters) - 1:
                y = self.acti(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)

        return y
