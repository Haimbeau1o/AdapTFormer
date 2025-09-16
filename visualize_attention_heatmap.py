import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sample import Model, DummyConfigs, DataEmbedding_wo_pos

# 1. 读取数据
csv_path = 'dataset/electricity/electricity.csv'
df = pd.read_csv(csv_path)
col_name = 'MT_020'
start_idx = 1000
end_idx = 1400
raw_series = df[col_name].values[start_idx:end_idx]

# 2. 转为模型输入格式 [Batch, Seq_Len, Channels]
input_tensor = torch.from_numpy(raw_series).float().unsqueeze(0).unsqueeze(-1)

# 3. 构造配置
class VanillaConfigs(DummyConfigs):
    def __init__(self):
        super().__init__()
        self.model_type = 'Vanilla'
        self.use_adaptive_weight = False
        self.use_RevIN = False
        self.seq_len = end_idx - start_idx
        self.output_attention = True

class AdapTFormerConfigs(DummyConfigs):
    def __init__(self):
        super().__init__()
        self.model_type = 'AdapTFormer'
        self.use_adaptive_weight = True
        self.use_RevIN = False
        self.seq_len = end_idx - start_idx
        self.output_attention = True

# 4. Patch Model 以支持 Vanilla/AdapTFormer 切换
import types
from sample import DecompEmbedding

def patch_model_for_vanilla(model, configs):
    # 替换 embedding 层为 DataEmbedding_wo_pos
    model.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model)
    # 禁用 adaptive weight
    for layer in model.encoder.layers:
        layer.self_attention.use_adaptive_weight = False
        if hasattr(layer.self_attention, 'adaptive_weight'):
            delattr(layer.self_attention, 'adaptive_weight')

# 5. 构造模型
configs_vanilla = VanillaConfigs()
model_vanilla = Model(configs_vanilla)
patch_model_for_vanilla(model_vanilla, configs_vanilla)
model_vanilla.eval()

configs_adapt = AdapTFormerConfigs()
model_adapt = Model(configs_adapt)
model_adapt.eval()

# 6. 前向推理，获取注意力分数
with torch.no_grad():
    _, attns_vanilla = model_vanilla(input_tensor, None, None, None)
    _, attns_adapt = model_adapt(input_tensor, None, None, None)

# 7. 选取同一层/头/样本
layer_idx = 0
head_idx = 0
attn_map_vanilla = attns_vanilla[layer_idx][0, head_idx].cpu().numpy()
attn_map_adapt = attns_adapt[layer_idx][0, head_idx].cpu().numpy()

# 8. 绘制热图对比
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(attn_map_vanilla, aspect='auto', cmap='viridis')
axes[0].set_title('Vanilla Transformer Attention')
axes[1].imshow(attn_map_adapt, aspect='auto', cmap='viridis')
axes[1].set_title('AdapTFormer Attention')
for ax in axes:
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
plt.tight_layout()
plt.show() 