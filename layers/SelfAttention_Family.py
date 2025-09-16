import torch
import torch.nn as nn
import numpy as np
from math import sqrt, exp
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import torch.nn.functional as F
from scipy.stats import gaussian_kde

# 注意力自适应权重（全连接层）
class AdaptiveAttentionWeightFC(nn.Module):
    def __init__(self, d_head, enc_in, n_heads):
        super(AdaptiveAttentionWeightFC, self).__init__()
        # 定义一个全连接层，输入是每个头的嵌入维度 d_head，输出是变量数 variate_num
        self.fc = nn.Linear(d_head, enc_in)
        self.n_heads = n_heads

    def forward(self, x):
        # x 的形状为 [batch_size, variate_num, n_heads, d_head]
        # 调整为 [batch_size, n_heads, variate_num, d_head] 以匹配多头维度
        x = x.permute(0, 2, 1, 3)  # [batch_size, n_heads, variate_num, d_head]

        # 在每个头的维度上使用全连接层生成权重，得到 [batch_size, n_heads, variate_num]
        weights = self.fc(x).mean(dim=-1)  # 对 d_head 维度取均值

        # 归一化，确保每个变量的权重总和为 1，形状为 [batch_size, n_heads, variate_num]
        weights = F.softmax(weights, dim=-1)

        return weights


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class originalFullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)


        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


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
        scale = self.scale or 1. / sqrt(E)

        # 计算原始注意力得分
        scores = torch.einsum("bvhe,bshe->bhvs", queries, keys)  # [B, H, V, V]

        # 应用掩码（如果存在）
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, V, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 应用自适应权重
        if adaptive_weights is not None:
            scores = scores * adaptive_weights  # 形状 [B, H, V, V] * [B, H, 1, V]

        # 计算最终注意力得分
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhvs,bshd->bvhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None



# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, sigma=0.0):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)
#         # self.sigma = sigma  # Sigma for Gaussian Prior
#         # # # 初始化自注意力权重和高斯权重
#         # self.a_weights = nn.Parameter(torch.ones(1))  # 自注意力权重
#
#
#     def forward(self, queries, keys, values, attn_mask, previous_attn=None, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)
#
#         # # 将 a_weights 和 g_weights 移动到与 queries 相同的设备
#         # a_weights = self.a_weights.to(queries.device)
#         # g_weights = self.g_weights.to(queries.device)
#
#         # # 假设 a_weights 和 g_weights 是固定的权重值
#         # a_weights = 1.0
#         # g_weights = 0.2  # 高斯先验权重
#
#         # 自注意力得分计算
#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)
#         if self.mask_flag and attn_mask is not None:
#             scores = scores.masked_fill(attn_mask == 0, -float('inf'))
#
#         # 应用SoftMax归一化得分
#         attention_weights = torch.softmax(scale * scores, dim=-1)
#
#
#         # if previous_attn is not None:
#         #     attention_weights = self.a_weights * attention_weights + (1-self.a_weights) * previous_attn
#
#
#         attention_weights = self.dropout(attention_weights)
#
#         # Vs = torch.einsum("bhls,bshd->blhd", attention_weights, values)  # 自注意力输出
#         V = torch.einsum("bhls,bshd->blhd", attention_weights, values)  # 自注意力输出
#
#         # # 高斯权重计算
#         # gaussian_weights = torch.zeros(B, L, S, device=queries.device)
#         # for i in range(L):
#         #     for j in range(S):
#         #         gaussian_weights[:, i, j] = exp(-(abs(i - j) ** 2) / (2 * self.sigma ** 2))
#         #
#         # # 在计算完成后扩展维度，使其符合注意力头的数量H
#         # gaussian_weights = gaussian_weights.unsqueeze(1).expand(-1, H, -1, -1)  # [B, H, L, S]
#         #
#         # # 高斯权重归一化
#         # gaussian_weights = gaussian_weights / gaussian_weights.sum(dim=-1, keepdim=True)
#         # gaussian_weights = self.dropout(gaussian_weights)
#         #
#         # Vg = torch.einsum("bhls,bshd->blhd", gaussian_weights, values)  # 高斯注意力输出
#
#
#         # # # 融合操作，这里使用加权和
#         # V = a_weights * Vs + g_weights * Vg  # 融合后的输出
#
#
#         if self.output_attention:
#             return V.contiguous(), attention_weights
#         else:
#             return V.contiguous(), None









class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()
#
#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)
#
#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads
#
#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads
#
#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)
#
#
#
#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta
#         )
#         out = out.view(B, L, -1)
#
#
#         return self.out_projection(out), attn

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

        # 初始化自适应权重模块
        if self.use_adaptive_weight:
            self.adaptive_weight = AdaptiveAttentionWeightFC(d_head=d_keys, enc_in=enc_in, n_heads=n_heads)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, V, _ = queries.shape  # V 是 variate_num
        H = self.n_heads

        queries = self.query_projection(queries).view(B, V, H, -1)
        keys = self.key_projection(keys).view(B, V, H, -1)
        values = self.value_projection(values).view(B, V, H, -1)

        # 计算自适应权重
        if self.use_adaptive_weight:
            adaptive_weights = self.adaptive_weight(queries)  # [B, H, V]
            adaptive_weights = adaptive_weights.unsqueeze(2)  # [B, H, 1, V] 以适配 scores 的维度
        else:
            adaptive_weights = None  # 如果未启用，则不使用自适应权重

        # 将自适应权重传递到 `FullAttention`
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



# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, variate_num, use_adaptive_weight=False, d_keys=None, d_values=None):
#         super(AttentionLayer, self).__init__()
#
#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)
#
#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads
#         self.d_head = d_keys
#         self.use_adaptive_weight = use_adaptive_weight
#
#         # 如果启用自适应权重，则初始化自适应权重模块
#         if self.use_adaptive_weight:
#             self.adaptive_weight = AdaptiveAttentionWeightFC(d_head=d_keys, variate_num=variate_num, n_heads=n_heads)
#
#     def forward(self, queries, keys, values, attn_mask=None, previous_attn=None, tau=None, delta=None):
#         B, V, _ = queries.shape  # V 是 variate_num
#         H = self.n_heads
#
#         # 将 queries, keys, values 投影到多头的形状
#         queries = self.query_projection(queries).view(B, V, H, self.d_head)  # [B, V, H, d_head]
#         keys = self.key_projection(keys).view(B, V, H, self.d_head)          # [B, V, H, d_head]
#         values = self.value_projection(values).view(B, V, H, self.d_head)    # [B, V, H, d_head]
#
#         # 计算自适应权重（如果启用）
#         if self.use_adaptive_weight:
#             adaptive_weights = self.adaptive_weight(queries)  # [B, H, V]
#             adaptive_weights = adaptive_weights.unsqueeze(2)  # [B, H, 1, V] 以适配 scores 的维度
#         else:
#             adaptive_weights = None  # 如果未启用，则不使用自适应权重
#
#         # 将自适应权重传递到 `FullAttention`
#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask=attn_mask,
#             tau=tau,
#             delta=delta,
#             adaptive_weights=adaptive_weights
#         )
#
#         out = out.view(B, V, -1)
#         return self.out_projection(out), attn

# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         """
#         初始化注意力层
#
#         参数:
#             attention (nn.Module): 注意力机制模块
#             d_model (int): 模型的维度
#             n_heads (int): 注意力头的数量
#             d_keys (int): 键的维度，默认为 d_model // n_heads
#             d_values (int): 值的维度，默认为 d_model // n_heads
#         """
#         super(AttentionLayer, self).__init__()
#
#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)
#
#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads
#
#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         """
#         前向传播
#
#         参数:
#             queries (Tensor): 查询张量，形状为 (B, L, d_model)
#             keys (Tensor): 键张量，形状为 (B, S, d_model)
#             values (Tensor): 值张量，形状为 (B, S, d_model)
#             attn_mask (Tensor): 注意力掩码，形状为 (B, L, S)
#             tau (float): 温度参数，用于调整注意力分布的集中度
#             delta (float): 偏移参数，用于调整注意力分布的位置
#
#         返回:
#             out (Tensor): 输出张量，形状为 (B, L, d_model)
#             attn (Tensor): 注意力权重，形状为 (B, H, L, S)
#         """
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads
#
#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)
#
#         # # 计算皮尔森相关系数
#         # # queries, keys shape: [B, L/S, H, d_keys] -> 计算每个 head 的相关性
#         # theta, phi = self.compute_pearson_correlation(queries, keys)
#
#         # 计算注意力权重，传递 theta 和 phi
#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta,
#             # theta=theta,
#             # phi=phi
#         )
#         out = out.view(B, L, -1)
#
#         return self.out_projection(out), attn


    # def compute_pearson_correlation(self, queries, keys):
    #     """
    #     计算 queries 和 keys 之间的皮尔森相关系数，作为 theta 和 phi。
    #     queries, keys 形状为 [B, L, H, D]
    #     B 是 batch, L 是变量数, H 是头数, D 是每个头的特征维度 (序列长度 / H)
    #     返回：
    #     - theta: 相关性的绝对值，用作相关性权重
    #     - phi: 相关性的方向，用作偏移量
    #     """
    #     B, L, H, D = queries.shape
    #
    #     # 重塑张量以便于计算
    #     queries = queries.transpose(1, 2).reshape(B * H, L, D)
    #     keys = keys.transpose(1, 2).reshape(B * H, L, D)
    #
    #     # 计算每个变量的均值，沿着最后一个维度 D 计算
    #     mean_q = torch.mean(queries, dim=-1, keepdim=True)
    #     mean_k = torch.mean(keys, dim=-1, keepdim=True)
    #
    #     # 计算标准差
    #     std_q = torch.std(queries, dim=-1, keepdim=True)
    #     std_k = torch.std(keys, dim=-1, keepdim=True)
    #
    #     # 计算协方差
    #     cov = torch.mean((queries.unsqueeze(2) - mean_q.unsqueeze(2)) *
    #                      (keys.unsqueeze(1) - mean_k.unsqueeze(1)), dim=-1)
    #
    #     # 计算皮尔森相关系数
    #     pearson_corr = cov / (std_q * std_k.transpose(1, 2) + 1e-8)
    #
    #     # 重塑回原始维度 [B, H, L, L]
    #     pearson_corr = pearson_corr.view(B, H, L, L)
    #
    #     theta = torch.abs(pearson_corr)
    #     phi = pearson_corr
    #
    #     return theta, phi




class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
