"""
AdapTFormer 及其所有依赖的主要类和函数完整代码汇总
"""
# ========== 依赖包导入 ==========
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.utils import weight_norm

# ========== layers/Autoformer_EncDec.py ==========
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# ========== layers/Embed.py ==========
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
class DecompEmbedding(nn.Module):
    def __init__(self, c_in, d_model, moving_avg, enc_in, individual=False):
        super(DecompEmbedding, self).__init__()
        self.seq_len = c_in
        self.d_model = d_model
        self.decompsition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = enc_in
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.d_model))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.d_model))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.d_model, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.d_model, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.d_model)
            self.Linear_Trend = nn.Linear(self.seq_len, self.d_model)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.d_model, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.d_model, self.seq_len]))
    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.d_model],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.d_model],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)
    def forward(self, x_enc):
        return self.encoder(x_enc)
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = None  # 简化
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x)  # + self.temporal_embedding(x_mark)
        return self.dropout(x)

# ========== layers/SelfAttention_Family.py ==========
class AdaptiveAttentionWeightFC(nn.Module):
    def __init__(self, d_head, enc_in, n_heads):
        super(AdaptiveAttentionWeightFC, self).__init__()
        self.fc = nn.Linear(d_head, enc_in)
        self.n_heads = n_heads
    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        weights = self.fc(x).mean(dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None, adaptive_weights=None):
        B, V, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        scores = torch.einsum("bvhe,bshe->bhvs", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.zeros(B, V, S, dtype=torch.bool, device=queries.device)
            scores.masked_fill_(attn_mask, -np.inf)
        if adaptive_weights is not None:
            scores = scores * adaptive_weights
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhvs,bshd->bvhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, enc_in, use_adaptive_weight, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.use_adaptive_weight = use_adaptive_weight
        if self.use_adaptive_weight:
            self.adaptive_weight = AdaptiveAttentionWeightFC(d_head=d_keys, enc_in=enc_in, n_heads=n_heads)
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, V, _ = queries.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, V, H, -1)
        keys = self.key_projection(keys).view(B, V, H, -1)
        values = self.value_projection(values).view(B, V, H, -1)
        if self.use_adaptive_weight:
            adaptive_weights = self.adaptive_weight(queries)
            adaptive_weights = adaptive_weights.unsqueeze(2)
        else:
            adaptive_weights = None
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
            adaptive_weights=adaptive_weights
        )
        out = out.view(B, V, -1)
        return self.out_projection(out), attn

# ========== layers/StandardNorm.py ==========
class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()
    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x
    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

# ========== normalization/RevIN.py ==========
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()
    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x
    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

# ========== AdapTFormer 主模型 ==========
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding = DecompEmbedding(configs.seq_len, configs.d_model, configs.moving_avg, configs.enc_in)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads, configs.enc_in, configs.use_adaptive_weight),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.use_RevIN = configs.use_RevIN
        if self.use_RevIN:
            self.revin = RevIN(configs.enc_in)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_RevIN:
            x_enc = self.revin(x_enc, 'norm')
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc)
        enc_out = enc_out.permute(0, 2, 1)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        if self.use_RevIN:
            dec_out = self.revin(dec_out, 'denorm')
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out, None
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, attentions = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if not self.output_attention:
                attentions = None
            return dec_out[:, -self.pred_len:, :], attentions
        return None

# ========== 示例配置和用法 ==========
class DummyConfigs:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.pred_len = 24
        self.output_attention = False
        self.d_model = 64
        self.moving_avg = 25
        self.enc_in = 7
        self.factor = 5
        self.dropout = 0.1
        self.n_heads = 4
        self.use_adaptive_weight = False
        self.e_layers = 2
        self.d_ff = 128
        self.activation = 'gelu'
        self.use_RevIN = True

if __name__ == '__main__':
    configs = DummyConfigs()
    model = Model(configs)
    print(model)
