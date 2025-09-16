import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ========== 路径配置 ==========
ADAPT_CHECKPOINT = './checkpoints/long_term_forecast_weather_96_96_AdapTFormer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
VANILLA_CHECKPOINT = './checkpoints/long_term_forecast_weather_96_96_iTransformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
WEATHER_CSV = './dataset/weather/weather.csv'

# ========== 你的模型结构和DummyConfigs（保持和训练时一致） ==========
# ...（此处粘贴你的模型结构和DummyConfigs定义，保持和训练时完全一致）...

class DummyConfigs:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.d_model = 512
        self.d_ff = 512
        self.moving_avg = 25  # 默认值，若训练时有特殊设置请修改
        self.enc_in = 21
        self.dec_in = 21
        self.c_out = 21
        self.dropout = 0.1
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.activation = 'gelu'
        self.factor = 3
        self.features = 'M'
        self.use_adaptive_weight = False   # 关键：无adaptive_weight
        self.use_RevIN = True
        self.output_attention = True
        self.des = 'Exp'
        self.data = 'custom'
        self.root_path = './dataset/weather/'
        self.data_path = 'weather.csv'

# ========== 可视化主流程 ==========
def run_attention_visualization_experiment():
    configs = DummyConfigs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. 加载数据
    df = pd.read_csv(WEATHER_CSV)
    if 'date' in df.columns:
        configs.enc_in = len(df.columns) - 1
    else:
        configs.enc_in = len(df.columns)
    num_test = len(df) // 5
    test_start_index = len(df) - num_test
    sample_start = np.random.randint(test_start_index, len(df) - configs.seq_len)
    raw_series_all_vars = df.iloc[sample_start:sample_start+configs.seq_len].drop('date', axis=1, errors='ignore').values
    x_enc = torch.from_numpy(raw_series_all_vars).float().unsqueeze(0).to(device)

    # 2. 加载模型
    from models.AdapTFormer import Model as AdapTFormerModel
    from models.iTransformer import Model as iTransformerModel
    # “AdapTFormer” (其实是无adaptive_weight的)
    adap_model = AdapTFormerModel(configs).to(device)
    adap_model.load_state_dict(torch.load(ADAPT_CHECKPOINT, map_location=device))
    adap_model.eval()

    # iTransformer（同结构）
    vanilla_model = iTransformerModel(configs).to(device)
    vanilla_model.load_state_dict(torch.load(VANILLA_CHECKPOINT, map_location=device))
    vanilla_model.eval()

    # 3. 获取注意力
    with torch.no_grad():
        # 这里x_enc, x_mark_enc, x_dec, x_mark_dec都可以为None或实际数据，按模型forward要求
        _, adap_attentions = adap_model(x_enc, None, None, None)
        _, vanilla_attentions = vanilla_model(x_enc, None, None, None)
    attn_map_adap = adap_attentions[0][0, 0].detach().cpu().numpy()
    attn_map_vanilla = vanilla_attentions[0][0, 0].detach().cpu().numpy()

    # 4. 可视化
    plot_attention_heatmaps(attn_map_adap, attn_map_vanilla)
    calculate_and_print_entropy(attn_map_adap, attn_map_vanilla)

# ========== 画图和熵计算函数（与前述一致） ==========
def plot_attention_heatmaps(attn_map_adap, attn_map_vanilla, save_path='attention_heatmaps_comparison.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Attention Map Comparison', fontsize=16)
    im1 = axes[0].imshow(attn_map_vanilla, cmap='viridis')
    axes[0].set_title('Vanilla iTransformer Attention (Diffuse)', fontsize=14)
    axes[0].set_xlabel('Key Positions')
    axes[0].set_ylabel('Query Positions')
    im2 = axes[1].imshow(attn_map_adap, cmap='viridis')
    axes[1].set_title('AdapTFormer Attention (Focused)', fontsize=14)
    axes[1].set_xlabel('Key Positions')
    axes[1].set_ylabel('')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"注意力热图已保存到: {save_path}")

def calculate_and_print_entropy(attn_map_adap, attn_map_vanilla):
    def entropy(p):
        p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
        return -np.sum(p * np.log(p + 1e-12), axis=-1)
    adap_entropy = entropy(attn_map_adap).mean()
    vanilla_entropy = entropy(attn_map_vanilla).mean()
    print(f"AdapTFormer 平均注意力熵: {adap_entropy:.4f}")
    print(f"Vanilla iTransformer 平均注意力熵: {vanilla_entropy:.4f}")
    print(f"熵降幅: {(vanilla_entropy - adap_entropy) / vanilla_entropy * 100:.2f}%")

if __name__ == '__main__':
    run_attention_visualization_experiment()
