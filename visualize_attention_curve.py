import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sample import Model, DummyConfigs, DataEmbedding_wo_pos

# 附录F的累积注意力分数曲线实现
def calculate_local_attention_curve(attention_matrix: np.ndarray):
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
        accumulated_prob = att_row[i]
        neighborhood_size = 1
        left, right = i - 1, i + 1
        y_idx = 0
        while y_idx < num_points:
            if accumulated_prob >= y_coords[y_idx]:
                x_coords_for_row[y_idx] = neighborhood_size / L
                y_idx += 1
            else:
                dist_left = i - left if left >= 0 else float('inf')
                dist_right = right - i if right < L else float('inf')
                if dist_left <= dist_right:
                    if left >= 0:
                        accumulated_prob += att_row[left]
                        neighborhood_size += 1
                        left -= 1
                    elif right < L:
                        accumulated_prob += att_row[right]
                        neighborhood_size += 1
                        right += 1
                    else:
                        break
                else:
                    if right < L:
                        accumulated_prob += att_row[right]
                        neighborhood_size += 1
                        right += 1
                    elif left >= 0:
                        accumulated_prob += att_row[left]
                        neighborhood_size += 1
                        left -= 1
                    else:
                        break
        x_coords_for_row[y_idx:] = neighborhood_size / L
        avg_x_coords += x_coords_for_row
    avg_x_coords /= L
    return avg_x_coords, y_coords

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
from sample import DecompEmbedding

def patch_model_for_vanilla(model, configs):
    model.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model)
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

# 8. 计算累积注意力分数曲线
x_vanilla, y_vanilla = calculate_local_attention_curve(attn_map_vanilla)
x_adapt, y_adapt = calculate_local_attention_curve(attn_map_adapt)

# 9. 绘制曲线对比
plt.figure(figsize=(8, 6))
plt.plot(x_vanilla * 100, y_vanilla * 100, label='Vanilla Transformer', color='orange')
plt.plot(x_adapt * 100, y_adapt * 100, label='AdapTFormer', color='blue')
plt.xlabel('Neighborhood Size (%)')
plt.ylabel('Accumulated Attention Score (%)')
plt.title('Local Accumulated Attention Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 