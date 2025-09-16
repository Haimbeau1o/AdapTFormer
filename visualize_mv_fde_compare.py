import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sample import series_decomp

def fedformer_fourier_decomp(series, keep_ratio=0.1):
    fft = np.fft.fft(series)
    L = len(series)
    keep = int(L * keep_ratio)
    fft_low = np.zeros_like(fft)
    fft_low[:keep] = fft[:keep]
    fft_low[-keep:] = fft[-keep:]
    trend = np.fft.ifft(fft_low).real
    residual = series - trend
    return trend, residual

# 1. 读取数据
csv_path = 'dataset/electricity/electricity.csv'
df = pd.read_csv(csv_path)
all_columns = [col for col in df.columns if col != 'date']

# 2. 固定变量20，时间步1000-1400
col_name = '20'
start_idx = 1000
end_idx = 1400
raw_series = df[col_name].values[start_idx:end_idx]
kernel_size = 75

# 3. Autoformer分解
input_tensor = torch.from_numpy(raw_series).float().unsqueeze(0).unsqueeze(-1)
decomposer = series_decomp(kernel_size=kernel_size)
with torch.no_grad():
    auto_residual_tensor, auto_trend_tensor = decomposer(input_tensor)
auto_trend = auto_trend_tensor.squeeze().cpu().numpy()
auto_residual = auto_residual_tensor.squeeze().cpu().numpy()

# 4. FEDformer分解
fed_trend, fed_residual = fedformer_fourier_decomp(raw_series, keep_ratio=0.1)

# 5. 绘图
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 第一幅：原始序列
axes[0].plot(raw_series, color='black', label='Raw', linewidth=2)
axes[0].set_title(f'Raw Input Signal (Variable 20)', fontsize=16)
axes[0].legend(fontsize=12)
axes[0].grid(True)
axes[0].tick_params(axis='both', labelsize=12)
axes[0].set_ylabel('Load (kW)', fontsize=12)

# 第二幅：趋势分量对比
axes[1].plot(auto_trend, color='red', label='Autoformer Trend', linewidth=2)
axes[1].plot(fed_trend, color='blue', label='FEDformer Trend', linewidth=2)
axes[1].set_title('Trend Component Comparison', fontsize=16)
axes[1].legend(fontsize=12)
axes[1].grid(True)
axes[1].tick_params(axis='both', labelsize=12)
axes[1].set_ylabel('Load (kW)', fontsize=12)

# 第三幅：残差分量对比
axes[2].plot(auto_residual, color='red', label='Autoformer Residual', linewidth=2)
axes[2].plot(fed_residual, color='blue', label='FEDformer Residual', linewidth=2)
axes[2].set_title('Residual Component Comparison', fontsize=16)
axes[2].legend(fontsize=12)
axes[2].grid(True)
axes[2].tick_params(axis='both', labelsize=12)
axes[2].set_xlabel('Time Index', fontsize=14)
axes[2].set_ylabel('Load (kW)', fontsize=12)

plt.tight_layout()
plt.savefig('mv_fde_decomp_compare_var20_1000_1400.png', dpi=300)
plt.show()
print('图片已保存为 mv_fde_decomp_compare_var20_1000_1400.png') 