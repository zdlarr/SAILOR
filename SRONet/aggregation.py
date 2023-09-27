"""
    The aggregation module to aggregate img features from N's views.
    aggregate features: considering projected depth values in other views.
    SelfAtt. HydraAtt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, temp, atten_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temp     = temp
        self.drop_out = nn.Dropout(atten_dropout)
        
    def forward(self, q, k, v):
        """
        q: [b, len_q, d_k];
        k: [b, len_v, d_k];
        v: [b, len_v, d_v];
        return : [b, len_q, d_v]
        """
        atten  = torch.matmul( q / self.temp, k.transpose(2,3) )
        # print(atten.shape) # [B, n_head, N, N];
        atten  = self.drop_out( F.softmax(atten, dim=-1) ) # Attention maps;
        output = torch.matmul( atten, v ) # [B, n_head, N, d_v];
        
        return output, atten


class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_head, d_model, d_v, d_k, dropout=0.1, norm=False):
        super(MultiHeadAttention, self).__init__()
        
        self._n_head = n_head
        self._d_k    = d_k
        self._d_v    = d_v
        
        self._w_qs  = nn.Linear(d_model, n_head * d_k, bias=False)
        self._w_ks  = nn.Linear(d_model, n_head * d_k, bias=False)
        self._w_vs  = nn.Linear(d_model, n_head * d_k, bias=False)
        self.fc     = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.atten = ScaledDotProductAttention(temp=d_k ** 0.5)
        
        self.dropout = nn.Dropout(dropout)
        self.norm    = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, q, k, v):
        # q : [B, N_q, d_model]
        # k : [B, N_k, d_model]
        # v : [B, N_k, d_model]
        batch_size, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1];
        d_k, d_v, n_head = self._d_k, self._d_v, self._n_head;
        
        res = q
        q = self._w_qs(q).view(batch_size, len_q, n_head, d_k);
        k = self._w_ks(k).view(batch_size, len_k, n_head, d_k);
        v = self._w_vs(v).view(batch_size, len_v, n_head, d_v); # [B, Len, n_head, d_v];

        # Transpose for attention dot product:
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # [bs, n_head, lq, dv];
        # No accelaration in this attention process.
        q, attn = self.atten(q, k, v)

        # Transpose to move the head dimension back: bs x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b sx lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout( self.fc(q) )
        q += res
        # q = self.bn(q.permute((0, 2, 1))).permute((0, 2, 1))
        # q = self.layer_norm(q)
        
        if self.norm:
            q = self.layer_norm(q)

        return q, attn


class HydraAttention(nn.Module):
    '''
    The full-heads attention modules.
    '''
    def __init__(self, n_inputs, d_inner, drop_out=0.1):
        super(HydraAttention, self).__init__()
        # e.g., d_inner = 64, n_inputs = 37;
        self._w_qs = nn.Linear(n_inputs, d_inner, bias=False)
        self._w_ks = nn.Linear(n_inputs, d_inner, bias=False)
        self._w_vs = nn.Linear(n_inputs, d_inner, bias=False)
        self._fc   = nn.Linear(d_inner, n_inputs, bias=False)
        # the hydra attention layer.

        self._drop_out = nn.Dropout(drop_out) # the dropout approach.
        # no normalization here.

    def forward(self, q, k, v):
        # reference to : Hydra Attention: Efficient Attention with Many Heads;
        # q : [B, N_q, n_inputs]
        # k : [B, N_k, n_inputs]
        # k : [B, N_v, n_inputs]
        
        # [B, Token(N_views), N_fts], the len_q ,len_k, len_v are num of views;
        res = v
        q = self._w_qs(q) # [B, len_q, N_fts]
        k = self._w_ks(k) # [B, len_k, N_fts]
        v = self._w_vs(v) # [B, len_v, N_fts]

        # hydra attention: sim(Q, K) @ V -> (\phi(Q) \phi(K)^T) @ V -> \phi(Q) (\phi(K)^T  V)
        # cosine, distance;
        # q = q / q.norm(dim=-1, keepdim=True)
        # k = k / k.norm(dim=-1, keepdim=True)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # step 1. K * V aggregate information among views.
        kv = (k * v).sum(dim=-2, keepdim=True) # [B, 1, F]
        # step 2. Q * (K * V) output the features.
        out = q * kv # [B, L, F] * [B, 1, F]
        # out += res
        
        out = self._drop_out( self._fc(out) )
        out += res # attenetion features.

        return out


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1, norm=False):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.bn = nn.BatchNorm1d(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        if self.norm:
        # x = self.bn(x.permute((0, 2, 1))).permute((0, 2, 1))
            x = self.layer_norm(x)
        return x
    

class HydraForward(nn.Module):
    
    def __init__(self, n_inputs, d_inner, drop_out=0.1, norm=False):
        super().__init__()
        self.w_1 = nn.Linear(n_inputs, d_inner)
        self.w_2 = nn.Linear(d_inner, n_inputs)
        # no drop out for rendering;
        self.drop_out = nn.Dropout(drop_out)
        
    def forward(self, x):
        # The feed forward layers.
        res = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.drop_out(x)
        x += res

        return x


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, norm=False):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, norm=norm)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, norm=norm)
    
    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class HydraEncoderLayer(nn.Module):

    def __init__(self, n_inputs, d_inner):
        super(HydraEncoderLayer, self).__init__()
        self.hydra_attn = HydraAttention(n_inputs, d_inner)
        self.pos_ffn    = HydraForward(n_inputs, d_inner)

    def forward(self, enc_input):
        enc_output = self.hydra_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        
        return enc_output
    
class Encoder(nn.Module):
    
    def __init__(self, n_layers, n_heads, d_k, d_v, d_models, d_inner, dropout=0.1, norm=False):
        super(Encoder, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_models, d_inner, n_heads, d_k, d_v, dropout=dropout, norm=norm)
            for _ in range(n_layers)])
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_models, eps=1e-6)
        
    def forward(self, feats_embedding, return_attns=False):
        """
        :param feats_embedding: (bs, num_views, dim)
        :param return_attns:
        :return:
        """
        enc_slf_attn_list = []
        # enc_output = self.dropout(self.position_enc(feats_embedding))
        enc_output = feats_embedding
        if self.norm: # only when applying the normalization, layer-norm;
            enc_output = self.layer_norm(enc_output)
            
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, norm=False):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, norm=norm)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, norm=norm)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, norm=norm)

    def forward(self, dec_input, enc_output):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, norm=False):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, norm=norm)
            for _ in range(n_layers)])
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, dec_input, enc_output, return_attns=False):
        """
        :param dec_input: (bs, 1, dim)
        :param enc_output:  (bs, num_views, dim)
        :param return_attns:
        :return:
        """
        dec_output = dec_input
        # dec_output = self.dropout(self.position_enc(dec_output))
        if self.norm:
            dec_output = self.layer_norm(dec_output)
        dec_slf_attn_list, dec_enc_attn_list = [], []
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class HydraEncoder(nn.Module):
    '''
    The encoder trick with hydra attention tricks.
    '''
    def __init__(self, n_layers, n_input, d_inner):
        super(HydraEncoder, self).__init__()
        # the layers stacked.
        self._layer_stack = nn.ModuleList([
            HydraEncoderLayer(n_input, d_inner) for _ in range(n_layers)
        ])

    def forward(self, feats_embedding):
        # feats embedding : (Bs=B*N_rays*N_ps, N_views, Dim=37)
        # returned attention results.
        enc_output = feats_embedding
        
        for enc_layer in self._layer_stack:
            enc_output = enc_layer(enc_output)
        
        return enc_output

class Transformer(nn.Module):
    
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()
        self.encoder = Encoder(d_model=d_model, d_inner=d_inner, n_layers=n_layers,
                               n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.decoder = Decoder(d_model=d_model, d_inner=d_inner, n_layers=n_layers,
                               n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_feats, trg_feats):
        """
        :param src_feats: (bs, num_views, dim)
        :param trg_feats: (bs, 1, dim)
        :return: fused feats: (bs, 1, dim)
        """
        enc_output, *_ = self.encoder(src_feats)
        dec_output, *_ = self.decoder(trg_feats, enc_output)
        return dec_output



if __name__ == '__main__':
    # encoder = Encoder(n_layers=2, n_heads=8, d_models=256, d_v=32, d_k=32, d_inner=256).cuda();
    encoder = HydraEncoder(2, 37, 32).cuda()
    feats_embedding = torch.randn([1000, 4, 37]).cuda();
    
    q = encoder(feats_embedding)[:,0]
    print(q.shape)
    