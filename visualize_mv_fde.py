import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from sample import series_decomp

# 1. 读取数据
csv_path = 'dataset/electricity/electricity.csv'
df = pd.read_csv(csv_path)
all_columns = [col for col in df.columns if col != 'date']

# 2. 随机选取1个变量
num_vars = 1
seq_len = 400
num_rows = len(df)
random.seed(42)
selected_var = random.choice(all_columns)

# 3. 随机选时间段
max_start = num_rows - seq_len
start_idx = random.randint(0, max_start)
end_idx = start_idx + seq_len
raw_series = df[selected_var].values[start_idx:end_idx]
input_tensor = torch.from_numpy(raw_series).float().unsqueeze(0).unsqueeze(-1)
decomposer = series_decomp(kernel_size=25)
with torch.no_grad():
    residual_tensor, trend_tensor = decomposer(input_tensor)
trend_component = trend_tensor.squeeze().cpu().numpy()
residual_component = residual_tensor.squeeze().cpu().numpy()

# 4. 绘图
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(raw_series, color='black', label='Raw', linewidth=2)
axes[0].set_title(f'Raw Input Signal (Var {selected_var})', fontsize=16)
axes[0].legend(fontsize=12)
axes[0].grid(True)
axes[0].tick_params(axis='both', labelsize=12)

axes[1].plot(raw_series, color='gray', linestyle='--', label='Raw', linewidth=1.5)
axes[1].plot(trend_component, color='blue', label='Trend', linewidth=2)
axes[1].set_title('Trend Component vs Raw', fontsize=16)
axes[1].legend(fontsize=12)
axes[1].grid(True)
axes[1].tick_params(axis='both', labelsize=12)

axes[2].plot(raw_series, color='gray', linestyle='--', label='Raw', linewidth=1.5)
axes[2].plot(residual_component, color='red', label='Residual', linewidth=2)
axes[2].set_title('Residual Component vs Raw', fontsize=16)
axes[2].legend(fontsize=12)
axes[2].grid(True)
axes[2].tick_params(axis='both', labelsize=12)
axes[2].set_xlabel('Time Index', fontsize=14)

plt.tight_layout()
plt.savefig('mv_fde_decomp_compare.png', dpi=300)
plt.show()
print('图片已保存为 mv_fde_decomp_compare.png') 