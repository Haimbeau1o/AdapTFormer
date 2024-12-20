import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import  DecompEmbedding
import numpy as np
from normalization.RevIN import RevIN
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize






class Model(nn.Module):
    """
    Code link:
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DecompEmbedding(configs.seq_len, configs.d_model, configs.moving_avg, configs.enc_in)

        # Encoder
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


        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)



        # normalization
        self.use_RevIN = configs.use_RevIN
        if self.use_RevIN:
            self.revin = RevIN(configs.enc_in)



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        对输入的时间序列数据进行预测

        参数:
        x_enc: 编码输入数据，形状为 [batch_size, input_length, num_features]
        x_mark_enc: 编码输入数据的时间戳，形状为 [batch_size, input_length, num_features]
        x_dec: 解码输入数据，形状为 [batch_size, output_length, num_features]
        x_mark_dec: 解码输入数据的时间戳，形状为 [batch_size, output_length, num_features]

        返回:
        dec_out: 预测的输出数据，形状为 [batch_size, output_length, num_features]
        """

        if self.use_RevIN:
            x_enc = self.revin(x_enc, 'norm')
        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc)

        enc_out = enc_out.permute(0, 2, 1)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        if self.use_RevIN:
            dec_out = self.revin(dec_out, 'denorm')
        # return dec_out, attns
        # 判断是否输出注意力
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out, None





    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, attentions = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 如果不需要注意力权重，返回 None
            if not self.output_attention:
                attentions = None
            return dec_out[:, -self.pred_len:, :], attentions  # [B, L, D]
        return None





