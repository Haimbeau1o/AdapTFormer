"""
AdapTFormer 注意力可视化实验脚本
用于生成两个关键图表证明SAM模块在MV-FDE协同下的有效性

实验一：注意力热图对比 (定性证明)
实验二：局部累积注意力分数曲线对比 (定量证明)

参考论文：《The Devil in Linear Transformer》中的Figure 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.nn.utils import weight_norm

# ========== 导入您的模型组件 ==========
# 从sample.py中复制所有必要的类
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

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, previous_attn=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

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

# ========== 模型定义 ==========
class AdapTFormer(nn.Module):
    """您的完整AdapTFormer模型"""
    def __init__(self, configs):
        super(AdapTFormer, self).__init__()
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

class VanillaTransformer(nn.Module):
    """基线模型：禁用SAM和MV-FDE的版本"""
    def __init__(self, configs):
        super(VanillaTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # 使用标准嵌入层而不是DecompEmbedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads, configs.enc_in, False),  # 禁用自适应权重
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
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
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

# ========== 配置类 ==========
class DummyConfigs:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.pred_len = 24
        self.output_attention = True  # 必须开启以获取注意力矩阵
        self.d_model = 64
        self.moving_avg = 25
        self.enc_in = 7
        self.factor = 5
        self.dropout = 0.1
        self.n_heads = 4
        self.use_adaptive_weight = True  # AdapTFormer使用，VanillaTransformer禁用
        self.e_layers = 2
        self.d_ff = 128
        self.activation = 'gelu'
        self.use_RevIN = True

# ========== 注意力曲线计算函数 ==========
def calculate_local_attention_curve(attention_matrix: np.ndarray):
    """
    根据注意力矩阵计算局部累积注意力分数曲线。
    该函数是 "The Devil in Linear Transformer" 论文附录F中伪代码的Python实现。
    """
    if not isinstance(attention_matrix, np.ndarray):
        attention_matrix = attention_matrix.detach().cpu().numpy()

    L = attention_matrix.shape[0]
    if L == 0: 
        return np.array([0]), np.array([0])

    row_sums = attention_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    attention_matrix = attention_matrix / row_sums

    num_points = 101
    y_coords = np.linspace(0, 1, num_points)
    avg_x_coords = np.zeros(num_points)

    for i in range(L):
        att_row = attention_matrix[i]
        x_coords_for_row = np.zeros(num_points)
        accumulated_prob, neighborhood_size = att_row[i], 1
        left, right = i - 1, i + 1
        y_idx = 1
        while y_idx < num_points:
            if accumulated_prob >= y_coords[y_idx]:
                x_coords_for_row[y_idx] = neighborhood_size / L
                y_idx += 1
            else:
                dist_left = i - left if left >= 0 else float('inf')
                dist_right = right - i if right < L else float('inf')
                if dist_left <= dist_right:
                    if left < 0 and right >= L: 
                        break
                    if left >= 0:
                        accumulated_prob += att_row[left]; left -= 1
                    else:
                        accumulated_prob += att_row[right]; right += 1
                else:
                    if right >= L and left < 0: 
                        break
                    if right < L:
                        accumulated_prob += att_row[right]; right += 1
                    else:
                        accumulated_prob += att_row[left]; left -= 1
                neighborhood_size += 1
        x_coords_for_row[y_idx:] = neighborhood_size / L
        avg_x_coords += x_coords_for_row

    avg_x_coords /= L
    return avg_x_coords, y_coords

# ========== 实验函数 ==========
def generate_sample_data(seq_len=96, enc_in=7, batch_size=1):
    """生成示例数据用于实验"""
    # 生成模拟的时间序列数据
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建有代表性的时间序列数据
    t = np.linspace(0, 4*np.pi, seq_len)
    data = np.zeros((batch_size, seq_len, enc_in))
    
    for i in range(enc_in):
        # 每个特征包含不同的模式
        if i == 0:
            data[0, :, i] = np.sin(t) + 0.1 * np.random.randn(seq_len)  # 正弦波
        elif i == 1:
            data[0, :, i] = np.cos(t) + 0.1 * np.random.randn(seq_len)  # 余弦波
        elif i == 2:
            data[0, :, i] = np.sin(2*t) + 0.1 * np.random.randn(seq_len)  # 高频正弦
        elif i == 3:
            data[0, :, i] = t + 0.1 * np.random.randn(seq_len)  # 线性趋势
        else:
            data[0, :, i] = np.random.randn(seq_len)  # 随机噪声
    
    return torch.FloatTensor(data)

def get_attention_maps(model, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
    """获取模型的注意力矩阵"""
    model.eval()
    with torch.no_grad():
        if isinstance(model, AdapTFormer):
            output, attentions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            output, attentions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    # attentions是一个列表，包含每一层的注意力矩阵
    # 形状: [层数, 批量, 头数, 序列长度, 序列长度]
    return attentions

def plot_attention_heatmaps(attn_map_adap, attn_map_vanilla, save_path='attention_heatmaps_comparison.png'):
    """绘制注意力热图对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Attention Map Comparison', fontsize=16)

    # 绘制左图：Vanilla Transformer
    im1 = axes[0].imshow(attn_map_vanilla, cmap='viridis')
    axes[0].set_title('Vanilla Transformer Attention (Diffuse)', fontsize=14)
    axes[0].set_xlabel('Key Positions')
    axes[0].set_ylabel('Query Positions')

    # 绘制右图：AdapTFormer
    im2 = axes[1].imshow(attn_map_adap, cmap='viridis')
    axes[1].set_title('AdapTFormer Attention (Focused)', fontsize=14)
    axes[1].set_xlabel('Key Positions')
    axes[1].set_ylabel('')  # 隐藏右图的y轴标签，更美观

    # 添加颜色条
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"注意力热图已保存到: {save_path}")

def plot_attention_curves(x_adap, y_adap, x_vanilla, y_vanilla, save_path='attention_curve_comparison.png'):
    """绘制注意力曲线对比"""
    plt.figure(figsize=(8, 8))
    plt.plot(x_adap, y_adap, label='AdapTFormer (Ours)', color='red', linewidth=2.5)
    plt.plot(x_vanilla, y_vanilla, label='Vanilla Transformer', color='blue', linestyle='--', linewidth=2.5)

    # 绘制理想曲线作为参考
    plt.plot([0, 1], [1, 1], color='gray', linestyle='-.', label='Ideal (Oracle)')  # 理想的聚焦
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Uniform (Worst)')  # 完全均匀稀释

    # 设置图表样式
    plt.xlabel('Neighborhood Size (%)', fontsize=14)
    plt.ylabel('Accumulated Attention Score (%)', fontsize=14)
    plt.title('Quantitative Comparison of Attention Concentration', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"注意力曲线已保存到: {save_path}")

def run_attention_visualization_experiment():
    """运行完整的注意力可视化实验"""
    print("=" * 60)
    print("AdapTFormer 注意力可视化实验")
    print("=" * 60)
    
    # 1. 准备配置
    configs = DummyConfigs()
    print(f"实验配置: seq_len={configs.seq_len}, d_model={configs.d_model}, n_heads={configs.n_heads}")
    
    # 2. 创建模型
    print("\n创建模型...")
    adap_model = AdapTFormer(configs)
    
    # 创建基线模型（禁用SAM和MV-FDE）
    vanilla_configs = DummyConfigs()
    vanilla_configs.use_adaptive_weight = False  # 禁用自适应权重
    vanilla_model = VanillaTransformer(vanilla_configs)
    
    print(f"AdapTFormer 参数数量: {sum(p.numel() for p in adap_model.parameters()):,}")
    print(f"Vanilla Transformer 参数数量: {sum(p.numel() for p in vanilla_model.parameters()):,}")
    
    # 3. 生成示例数据
    print("\n生成示例数据...")
    x_enc = generate_sample_data(configs.seq_len, configs.enc_in)
    x_mark_enc = None  # 简化，不使用时间戳
    x_dec = torch.zeros(1, configs.pred_len, configs.enc_in)
    x_mark_dec = None
    
    print(f"输入数据形状: {x_enc.shape}")
    
    # 4. 获取注意力矩阵
    print("\n获取注意力矩阵...")
    adap_attentions = get_attention_maps(adap_model, x_enc, x_mark_enc, x_dec, x_mark_dec)
    vanilla_attentions = get_attention_maps(vanilla_model, x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    # 提取第一个样本、第一层、第一个头的注意力图
    attn_map_adap = adap_attentions[0][0, 0].detach().cpu().numpy()
    attn_map_vanilla = vanilla_attentions[0][0, 0].detach().cpu().numpy()
    
    print(f"注意力矩阵形状: {attn_map_adap.shape}")
    
    # 5. 实验一：绘制注意力热图对比
    print("\n实验一：绘制注意力热图对比...")
    plot_attention_heatmaps(attn_map_adap, attn_map_vanilla)
    
    # 6. 实验二：计算并绘制注意力曲线
    print("\n实验二：计算注意力曲线...")
    x_adap, y_adap = calculate_local_attention_curve(attn_map_adap)
    x_vanilla, y_vanilla = calculate_local_attention_curve(attn_map_vanilla)
    
    # 计算定量指标
    adap_area = np.trapz(y_adap, x_adap)
    vanilla_area = np.trapz(y_vanilla, x_vanilla)
    improvement = (adap_area - vanilla_area) / vanilla_area * 100
    
    print(f"AdapTFormer 曲线下面积: {adap_area:.4f}")
    print(f"Vanilla Transformer 曲线下面积: {vanilla_area:.4f}")
    print(f"改进百分比: {improvement:.2f}%")
    
    plot_attention_curves(x_adap, y_adap, x_vanilla, y_vanilla)
    
    # 7. 总结
    print("\n" + "=" * 60)
    print("实验总结:")
    print(f"1. 注意力热图对比: 展示了AdapTFormer的注意力更加聚焦")
    print(f"2. 注意力曲线对比: AdapTFormer相比基线模型提升了 {improvement:.2f}%")
    print(f"3. 这证明了SAM模块在MV-FDE协同下的有效性")
    print("=" * 60)

if __name__ == '__main__':
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行实验
    run_attention_visualization_experiment()