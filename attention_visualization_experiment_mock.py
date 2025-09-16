import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================================
# 1. 核心函数 (calculate_local_attention_curve 和 generate_simulated_attention_matrix 不变)
# ==============================================================================

def calculate_local_attention_curve(attention_matrix: np.ndarray):
    """根据注意力矩阵计算局部累积注意力分数曲线。"""
    # ... (此函数代码与上一版完全相同，此处省略以保持简洁) ...
    if not isinstance(attention_matrix, np.ndarray):
        attention_matrix = attention_matrix.detach().cpu().numpy()
    L = attention_matrix.shape[0]
    if L == 0: return np.array([0]), np.array([0])
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
                if (left < 0 and right >= L): break
                dist_left = i - left if left >= 0 else float('inf')
                dist_right = right - i if right < L else float('inf')
                if dist_left <= dist_right:
                    if left >= 0:
                        accumulated_prob += att_row[left]; left -= 1
                    else:
                        accumulated_prob += att_row[right]; right += 1
                else:
                    if right < L:
                        accumulated_prob += att_row[right]; right += 1
                    else:
                        accumulated_prob += att_row[left]; left -= 1
                neighborhood_size += 1
        x_coords_for_row[y_idx:] = neighborhood_size / L
        avg_x_coords += x_coords_for_row
    avg_x_coords /= L
    return avg_x_coords, y_coords


def generate_simulated_attention_matrix(size, focus_sigma, noise_level=0.01):
    """生成一个模拟的注意力矩阵."""
    coords = np.arange(size)
    x, y = np.meshgrid(coords, coords)
    dist_sq = (x - y)**2
    focused_attention = np.exp(-dist_sq / (2 * focus_sigma**2))
    noise = np.random.rand(size, size) * noise_level
    sim_matrix = focused_attention + noise
    return sim_matrix

# ==============================================================================
# 2. 优化后的绘图函数
# ==============================================================================

def plot_attention_heatmaps_optimized(attn_map_vanilla, attn_map_adap, save_path='sim_attention_heatmaps_optimized.png'):
    """绘制注意力热图对比 (美观居中，colorbar紧贴右图)"""

    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)

    # 统一颜色范围
    vmin = min(attn_map_vanilla.min(), attn_map_adap.min())
    vmax = max(attn_map_vanilla.max(), attn_map_adap.max())

    # 左图
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(attn_map_vanilla, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax0.set_title('Baseline Attention (Focused)', fontsize=14)
    ax0.set_xlabel('Key Positions', fontsize=12)
    ax0.set_ylabel('Query Positions', fontsize=12)
    ax0.tick_params(axis='both', labelsize=10)

    # 右图
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(attn_map_adap, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax1.set_title('AdapTFormer Attention (More Focused)', fontsize=14)
    ax1.set_xlabel('Key Positions', fontsize=12)
    ax1.set_ylabel('')
    ax1.tick_params(axis='y', labelleft=False)
    ax1.tick_params(axis='both', labelsize=10)

    # colorbar只绑定右图
    cax = fig.add_subplot(gs[2])
    cb = fig.colorbar(im1, cax=cax)
    cb.ax.tick_params(labelsize=10)

    plt.suptitle('Simulated Attention Map Comparison', fontsize=18, x=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"优化后的注意力热图已保存到: {save_path}")

def plot_attention_curves(x_adap, y_adap, x_vanilla, y_vanilla, save_path='sim_attention_curve_optimized.png'):
    """绘制注意力曲线对比 (函数不变)"""
    plt.figure(figsize=(8, 8))
    plt.plot(x_adap, y_adap, label='AdapTFormer (Simulated)', color='red', linewidth=2.5)
    plt.plot(x_vanilla, y_vanilla, label='Baseline (Simulated)', color='blue', linestyle='--', linewidth=2.5)
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Uniform (Worst Case)')
    
    plt.xlabel('Neighborhood Size (%)', fontsize=14)
    plt.ylabel('Accumulated Attention Score (%)', fontsize=14)
    plt.title('Simulated Comparison of Attention Concentration', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 1); plt.ylim(0, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"优化后的注意力曲线已保存到: {save_path}")

# ==============================================================================
# 3. 主实验流程 (修改了模拟参数)
# ==============================================================================
if __name__ == '__main__':
    # --- 实验参数设置 (调整后，效果更微妙和真实) ---
    MATRIX_SIZE = 96 
    
    # 为基线模型设置一个较好的聚焦度 (sigma稍大)
    SIGMA_VANILLA = 5.0
    
    # 为AdapTFormer设置一个更强的聚焦度 (sigma稍小)
    SIGMA_ADAPTFORMER = 2.5
    
    print("开始生成模拟注意力矩阵 (参数已调整)...")
    # --- 生成模拟数据 ---
    attn_map_adap = generate_simulated_attention_matrix(MATRIX_SIZE, SIGMA_ADAPTFORMER)
    attn_map_vanilla = generate_simulated_attention_matrix(MATRIX_SIZE, SIGMA_VANILLA)
    print("模拟矩阵生成完毕。")
    
    # --- 实验一：绘制优化后的注意力热图 ---
    print("\n实验一：正在绘制优化后的注意力热图对比...")
    plot_attention_heatmaps_optimized(attn_map_vanilla, attn_map_adap)
    
    # --- 实验二：绘制注意力曲线 ---
    print("\n实验二：正在计算并绘制注意力集中度曲线...")
    x_adap, y_adap = calculate_local_attention_curve(attn_map_adap)
    x_vanilla, y_vanilla = calculate_local_attention_curve(attn_map_vanilla)
    plot_attention_curves(x_adap, y_adap, x_vanilla, y_vanilla)

    print("\n模拟实验完成！")