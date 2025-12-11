import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
import os

# 配置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
# 设置 seaborn 风格
sns.set_context("paper", font_scale=1.2)

OUTPUT_DIR = "paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 元素周期表映射 (部分)
ELEMENTS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 
    # 镧系
    57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 
    67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 
    # 锕系
    89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 
    99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr',
    # 第七周期后续
    104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn', 
    113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
            }

def load_data(filename='alpha_values.npz'):
    data = np.load(filename, allow_pickle=True)
    return data['alphas'], data['atom_types'], data['targets']

def plot_figure_1_distribution(alphas, targets):
    """Figure 1: 整体分布分析 (保持之前的深紫色调)"""
    print("Plotting Figure 1...")
    
    # 展平所有原子
    flat_alphas = np.concatenate(alphas)
    
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 定义颜色
    color_main = '#482878'  # 核心深紫色
    
    # (a) 直方图
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(flat_alphas, bins=50, kde=True, color=color_main, edgecolor='white', ax=ax1, alpha=0.8)
    
    ax1.axvline(0.3, color='#fde725', linestyle='--', linewidth=2, label='Text-heavy (<0.3)')
    ax1.axvline(0.7, color='#35b779', linestyle='--', linewidth=2, label='Graph-heavy (>0.7)')
    
    ax1.set_xlabel('Gate Value (α)')
    ax1.set_title('(a) Overall Distribution of Gate Values')
    ax1.legend()
    
    # (b) 按原子数分组 (Boxplot)
    ax2 = fig.add_subplot(gs[0, 1])
    num_atoms = [len(a) for a in alphas]
    mean_alphas = [np.mean(a) for a in alphas]
    
    df_size = pd.DataFrame({'Num Atoms': num_atoms, 'Mean Alpha': mean_alphas})
    df_size['Size Group'] = pd.cut(df_size['Num Atoms'], 
                                 bins=[0, 10, 20, 50, 100, 1000], 
                                 labels=['<10', '10-20', '20-50', '50-100', '>100'])
    
    # 使用 magma 调色板 (黑/紫/红)
    sns.boxplot(data=df_size, x='Size Group', y='Mean Alpha', ax=ax2, palette='magma')
    ax2.set_title('(b) Gate Values by Crystal Size')
    
    # (c) 按目标值分组 (Violin)
    ax3 = fig.add_subplot(gs[1, 0])
    df_target = pd.DataFrame({'Target': targets, 'Mean Alpha': mean_alphas})
    df_target['Target Group'] = pd.qcut(df_target['Target'], q=3, labels=['Low', 'Medium', 'High'])
    
    sns.violinplot(data=df_target, x='Target Group', y='Mean Alpha', ax=ax3, palette='magma')
    ax3.set_title('(c) Gate Values by Target Property')
    
    # (d) 散点图
    ax4 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(x=targets, y=mean_alphas, alpha=0.6, ax=ax4, color=color_main)
    sns.regplot(x=targets, y=mean_alphas, scatter=False, ax=ax4, color='#fde725') 
    
    ax4.set_xlabel('Target Value (e.g., Bandgap)')
    ax4.set_ylabel('Mean Alpha')
    ax4.set_title(f'(d) Correlation (r={np.corrcoef(targets, mean_alphas)[0,1]:.2f})')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_gate_analysis.pdf")

def plot_figure_2_elements(alphas, atom_types):
    """Figure 2: 元素分析 (改为深紫色系 Purples)"""
    print("Plotting Figure 2...")
    
    element_alphas = {}
    
    for cryst_alphas, cryst_atoms in zip(alphas, atom_types):
        if cryst_atoms.ndim > 1: cryst_atoms = cryst_atoms.flatten()
        for alpha, z in zip(cryst_alphas, cryst_atoms):
            z = int(z)
            if z not in element_alphas: element_alphas[z] = []
            element_alphas[z].append(alpha)
            
    stats = []
    for z, vals in element_alphas.items():
        if len(vals) > 50:
            stats.append({
                'Z': z, 
                'Element': ELEMENTS.get(z, str(z)),
                'Mean': np.mean(vals),
                'Std': np.std(vals),
                'Count': len(vals)
            })
            
    df_stats = pd.DataFrame(stats).sort_values('Mean')
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # --- 颜色修改区域 ---
    # 使用 Purples 色带，但为了避免数值低时颜色太浅看不见，
    # 我们截取色带的 0.4 到 1.0 区间，让整体呈现从"中紫"到"深紫"
    # 根据 Mean 值进行映射
    norm = plt.Normalize(df_stats['Mean'].min(), df_stats['Mean'].max())
    # 获取 Purples 色带
    cmap = plt.cm.Purples
    # 生成颜色数组：数值越高，紫色越深
    colors = cmap(np.linspace(0.4, 1.0, len(df_stats)))
    
    barplot = ax.bar(range(len(df_stats)), df_stats['Mean'], yerr=df_stats['Std'], 
                     color=colors, capsize=3, alpha=1.0) # alpha=1.0 保证颜色饱和
    
    ax.set_xticks(range(len(df_stats)))
    ax.set_xticklabels(df_stats['Element'], rotation=45)
    ax.set_ylabel('Average Gate Value (α)')
    ax.set_title('Average Gate Value by Element (Sorted)')
    
    # 辅助线改为深灰色
    ax.axhline(0.5, linestyle='--', color='gray', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_alpha_by_element.pdf")
    
    df_stats[['Element', 'Mean', 'Std', 'Count']].to_csv(f"{OUTPUT_DIR}/element_alpha_statistics.csv", index=False)

def plot_figure_heatmaps(alphas, atom_types, targets):
    """Figure 4: 材料热图网格 (改为 Purples 深紫风格)"""
    print("Plotting Figure 4...")
    
    mean_alphas = np.array([np.mean(a) for a in alphas])
    indices = np.argsort(mean_alphas)
    
    selection = [indices[0], indices[1], 
                 indices[len(indices)//2], indices[len(indices)//2 + 1],
                 indices[-1], indices[-2]]
    
    fig, axes = plt.subplots(1, 6, figsize=(18, 6))
    
    for i, idx in enumerate(selection):
        ax = axes[i]
        mat_alpha = alphas[idx]
        mat_atoms = atom_types[idx]
        if mat_atoms.ndim > 1: mat_atoms = mat_atoms.flatten()
        
        limit = 15
        if len(mat_alpha) > limit:
            display_alpha = mat_alpha[:limit]
            display_atoms = mat_atoms[:limit]
            title_suffix = "..."
        else:
            display_alpha = mat_alpha
            display_atoms = mat_atoms
            title_suffix = ""
            
        # --- 颜色修改区域 ---
        # 使用 'Purples' cmap: 0=白色/浅紫, 1=深紫
        im = ax.imshow(display_alpha.reshape(-1, 1), cmap='Purples', vmin=0, vmax=1, aspect='auto')
        
        # 标注
        for j, (val, z) in enumerate(zip(display_alpha, display_atoms)):
            elem = ELEMENTS.get(int(z), str(int(z)))
            
            # 文字颜色自适应：
            # Purples 色带中，数值 > 0.5 时背景较深，使用白色文字
            # 数值 < 0.5 时背景较浅，使用黑色文字
            text_color = 'white' if val > 0.5 else 'black'
            
            ax.text(0, j, f"{elem}\n{val:.2f}", ha='center', va='center', fontsize=8, color=text_color)
            
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Target: {targets[idx]:.2f}\nMean α: {mean_alphas[idx]:.2f}")
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_alpha_heatmap_materials.pdf")

if __name__ == "__main__":
    if os.path.exists('alpha_values.npz'):
        alphas, atom_types, targets = load_data()
        plot_figure_1_distribution(alphas, targets)
        plot_figure_2_elements(alphas, atom_types)
        plot_figure_heatmaps(alphas, atom_types, targets)
        print("All figures generated successfully!")
    else:
        print("alpha_values.npz not found. Please run extraction script first.")
