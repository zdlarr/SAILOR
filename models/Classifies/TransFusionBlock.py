"""
    Views fusion block based on transformer.
    1. Fusion Block based on Transformer.
    2. Multi-head attention weights based on visibility map (calculated by depth map: z - depth.)
    3. Parameters: head_num, the GT feature fusion tensor: [B, Num_views, Num_points]
    4. Input feature shape: [B, N_v, C, N_points] - fused -> [B, C, N_points] where C : [C_2D, z, pixel_valid, view_direction]
 """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks import get_norm_layer


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, num_head, num_in_feature, num_val_feature, num_inner_feature, bias=True, 
                 dropout=0.0, activation=nn.ReLU(inplace=True)):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_head = num_head
        self.inner_dim = num_inner_feature * num_head
        self.scale = num_inner_feature
        self.activation = activation

        # 1x1 linear layers, bias are needed for the multi-head-attention
        self.query = nn.Linear(num_in_feature, self.inner_dim, bias=bias)
        self.key   = nn.Linear(num_in_feature, self.inner_dim, bias=bias)
        self.value = nn.Linear(num_val_feature, self.inner_dim, bias=bias)

        self.out_linear = nn.Sequential(
            nn.Linear(self.inner_dim, num_val_feature, bias=bias),
            nn.Dropout(dropout)
        )

    def reshape_to_batches(self, x):
        # x: [B, N_pixels or N_points, inner_dim]. inner_dim = in_f * n_head.
        batch_size, _, inner_dim, n_head = *x.shape, self.num_head
        sub_dim = inner_dim // n_head
        x_view = x.reshape(batch_size, -1, n_head, sub_dim) \
                  .permute(0, 2, 1, 3) \
                  .reshape(batch_size * n_head, -1, sub_dim) \
         # [B*n_head, N_pixels or N_points., sub_dim.]
        return x_view
    
    def reshape_from_batches(self, x):
        # [Bxnhead, N_pixels, sub_dim]
        _, n_pixels, sub_dim = x.shape
        n_head = self.num_head
        x_view = x.reshape(-1, n_head, n_pixels, sub_dim)\
                  .permute(0, 2, 1, 3) \
                  .reshape(-1, n_pixels, sub_dim * n_head) # [B, N, C]
        return x_view

    def scale_dot_production(self, q, k, v):
        # q, k, v: [B*n_head, N_pixels, sub_dim]
        dk = q.shape[-1]
        assert dk == self.scale
        scores = q.matmul(k.transpose(-2, -1)) / dk ** 0.5 # (\Phi_q ^ T * \Phi_k) / \sqrt(d_k)
        attention = F.softmax(scores, dim=-1) # [B * head, N_pixels, N_pixels]; the values are located in [0, 1];
        return attention.matmul(v), attention # [B * n_head, N_pixels, sub_dim]

    def forward(self, feature):
        # feature: [B, N_pixel, C], for view fusion, N_pixel <-> N_views.
        # Get Key, Query, Value;
        q, k, v = self.query(feature), self.key(feature), self.value(feature) # [B, N_pixels, inner_dim].

        # activate the feature volumem, calculate k,q,v
        if self.activation is not None:
            q, k, v = self.activation(q), self.activation(k), self.activation(v) 
    
        # reshape operation: [B, N_pixels, inner_dim] -> [B * n_head, N_pixels, sub_dim]
        q, k, v = self.reshape_to_batches(q), self.reshape_to_batches(k), self.reshape_to_batches(v) # [B x n_head, N_pixel, sub_dim]
        # attention map: [B x n_head, N_pixels, N_pixels] x [B * n_head, N_pixels, sub_dim] -> [B * n_head, N_pixels, sub_dim];
        y, attention_map = self.scale_dot_production(q, k, v) # [B x nhead, N_pixels, sub_dim]; [B x nhead, N_pixels, N_pixels]
        y = self.reshape_from_batches(y) # [B, N_pixel, inner_dim x n_head]
        y = self.out_linear(y) # [B, N_pixels, C]

        if self.activation is not None:
            y = self.activation(y) # [B, N, C] # add attention to each view.
            
        return y, attention_map


# multi-head attention based transformer.
class TransformerEncoderBlock(nn.Module):

    def __init__(self, input_dim, opts, device, head_num=None):
        super(TransformerEncoderBlock, self).__init__()
        self.opts = opts
        self.device = device
        self.num_views = opts.num_views
        
        self.num_heads = opts.att_num_heads
        if head_num is not None:
            self.num_heads = head_num

        self.num_in_feats = input_dim
        self.num_inner_feat = self.num_in_feats * (self.num_heads // 2) // self.num_heads # e.g., 66 * 4 // 8 = 33;
        self.activation = nn.LeakyReLU()

        self.attention_layer = MultiHeadSelfAttention(self.num_heads, self.num_in_feats, self.num_in_feats, self.num_inner_feat, activation=self.activation)
        self.layer_norm = nn.LayerNorm(self.num_in_feats, eps=1e-6)
        
        # the final output layer.
        self.out_linear = nn.Linear(self.num_in_feats, self.num_in_feats, bias=True)
    
    def forward(self, x):
        # input features: [B * N_v, C, N_p]
        n_feat, n_points = x.shape[-2:]
        # step 1. transform x -> [B, N_v, C, N_p] -> [B * N_p, N_v, C];
        pt_feat = x.reshape(-1, self.num_views, n_feat, n_points)
        pt_feat = pt_feat.permute(0, 3, 1, 2).reshape(-1, self.num_views, n_feat)
        
        # step 2. self attention
        # pass attention layer; 
        att_output, attention_map = self.attention_layer(pt_feat) # [B * N_p, N_v, C];
        att_output += pt_feat # [B * N_p, N_v, C];
        att_output = self.layer_norm(att_output) # [B * N_p, N_v, C];

        # step 3. the last mlp layer. [B * N_p, N_v, C]; norm(input + mlp(input))
        mlp_output = self.activation(self.out_linear(att_output)) + att_output
        mlp_output = self.layer_norm(mlp_output)
        
        # step 4. reshape the last output linear layer;
        mlp_output = mlp_output.reshape(-1, n_points, self.num_views, n_feat)
        mlp_output = mlp_output.permute(0, 2, 3, 1).reshape(-1, n_feat, n_points) # [B, N_v, C, N_p]

        return mlp_output, attention_map


class TransformerEncoder(nn.Module):
    
    def __init__(self, input_dim, opts, device, head_num=None):
        super(TransformerEncoder, self).__init__()
        self.num_layers = opts.att_num_layers
        self.num_views  = opts.num_views

        self.att_modules = []
        for i in range(self.num_layers):
            self.att_modules.append(TransformerEncoderBlock(input_dim, opts, device, head_num))

        self.att_modules = nn.Sequential(*self.att_modules)

    def forward(self, x):
        # x: [B * N_v, C, N_p]
        b = x.shape[0] // self.num_views
        n_points = x.shape[-1]
        feat = x.clone()
        
        att_maps = []
        for m in self.att_modules:
            feat, att_map  = m(feat)
            att_maps.append(att_map)
            
        return feat, att_maps
        
if __name__ == '__main__':
    from options.GeoTrainOptions import GeoTrainOptions
    import matplotlib.pyplot as plt
    opts = GeoTrainOptions().parse()

    x = torch.randn([5 * 3, 128, 260000]).to('cuda:0')
    encoder = TransformerEncoder(opts, device='cuda:0').to('cuda:0')
    out, att_maps = encoder(x)
    print(out.shape)