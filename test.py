import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.gridspec as gridspec

# ==========================================
# 1. 全局绘图设置
# ==========================================
# 字体设置：防止 Mac 上找不到字体报错
try:
    plt.rcParams['font.family'] = 'serif'
    # Mac 用户通常有 STSong (宋体) 或 Times New Roman
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Songti SC']
except Exception as e:
    pass

plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300 
# 关闭自动紧凑布局警告
plt.rcParams['figure.autolayout'] = False 

# ==========================================
# 2. 辅助绘图函数
# ==========================================
def draw_box(ax, center_x, center_y, text, color, w=0.22, h=0.15, label=None, subtext=None, highlight=False):
    # 计算左下角坐标
    xy = (center_x - w/2, center_y - h/2)
    
    if highlight:
        bg_color = color + '20' # 16进制颜色加透明度
        edge_width = 2.0
    else:
        bg_color = '#f5f5f5'
        edge_width = 1.2

    # 绘制圆角矩形
    rect = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", 
                                  linewidth=edge_width, edgecolor=color, facecolor=bg_color,
                                  mutation_scale=0.8) # 减小 mutation_scale 防止圆角过大
    ax.add_patch(rect)
    
    # 绘制框内文字
    ax.text(center_x, center_y, text, ha='center', va='center', 
            fontsize=10, fontweight='bold', color='black')
    
    # 顶部标签 (如 LLM)
    if label:
        ax.text(center_x, center_y + h/2 + 0.02, label, ha='center', va='bottom', 
                fontsize=9, color='dimgray', style='italic')
        
    # 底部说明 (如 Loss weight)
    if subtext:
        ax.text(center_x, center_y - h/2 - 0.02, subtext, ha='center', va='top', 
                fontsize=9, color='#d62728', fontweight='bold')

def draw_arrow(ax, x_start, x_end, y, color='black'):
    ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, shrinkA=0, shrinkB=0))

# ==========================================
# 3. 主绘图函数
# ==========================================
def main():
    # 创建画布
    fig = plt.figure(figsize=(15, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.6, 1.2], figure=fig, wspace=0.25)

    # -------------------------------------------------------
    # Panel A: Curriculum Schedule
    # -------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    
    steps = np.linspace(0, 100, 100)
    p_start, p_end = 0.95, 0.10
    p_hint = p_start - (steps / 100) * (p_start - p_end)

    ax1.plot(steps, p_hint, color='#1f77b4', linewidth=2.5, label='$p_{hint}(t)$')
    ax1.fill_between(steps, 0, p_hint, color='#1f77b4', alpha=0.15)
    ax1.fill_between(steps, p_hint, 1.1, color='#ff7f0e', alpha=0.15)

    ax1.text(35, 0.3, 'Mode A:\nHint Utilization\n(Scaffolding)', 
             ha='center', va='center', color='#0d4f8b', fontweight='bold', fontsize=9)
    ax1.text(65, 0.8, 'Mode B:\nHint Generation\n(Internalization)', 
             ha='center', va='center', color='#b45f06', fontweight='bold', fontsize=9)

    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Training Steps (t)')
    ax1.set_ylabel('Probability ($p_{hint}$)')
    ax1.set_title('(a) Curriculum Schedule', fontweight='bold', y=-0.18)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # -------------------------------------------------------
    # Panel B: Dual-Mode Training
    # -------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    # --- 上分支: Mode A ---
    y_top = 0.75
    ax2.text(0.05, y_top + 0.12, 'Mode A: Hint Utilization', color='#1f77b4', fontweight='bold', ha='left')
    
    draw_box(ax2, 0.15, y_top, "Q + Hint", '#808080', w=0.2)
    draw_box(ax2, 0.5, y_top, "Student\nModel", '#1f77b4', label="LLM", highlight=True)
    draw_box(ax2, 0.85, y_top, "Answer", '#2ca02c', subtext="w = 1.0", highlight=True)
    
    draw_arrow(ax2, 0.26, 0.39, y_top)
    draw_arrow(ax2, 0.61, 0.74, y_top)

    # --- 下分支: Mode B ---
    y_bot = 0.3
    ax2.text(0.05, y_bot + 0.12, 'Mode B: Hint Generation', color='#ff7f0e', fontweight='bold', ha='left')
    
    draw_box(ax2, 0.15, y_bot, "Question", '#808080', w=0.2)
    draw_box(ax2, 0.42, y_bot, "Student\nModel", '#ff7f0e', label="LLM", highlight=True)
    
    # --- 修复点：这里使用了 mathtext 支持的语法 ---
    # 使用 \mathtt 代替 \texttt，或者直接用换行符
    # 这里的字符串被Matplotlib直接渲染，不需要 $$ 包裹复杂的LaTeX命令，简单的字符串更稳健
    hint_text = "<KNOWN>\nHint\n</KNOWN>" 
    draw_box(ax2, 0.68, y_bot, hint_text, '#ff7f0e', w=0.22, h=0.18, 
             subtext="λ = 4.0", highlight=True)
    
    draw_box(ax2, 0.92, y_bot, "Answer", '#2ca02c', w=0.15, subtext="w = 1.0", highlight=True)

    draw_arrow(ax2, 0.26, 0.31, y_bot)
    draw_arrow(ax2, 0.53, 0.57, y_bot)
    draw_arrow(ax2, 0.79, 0.84, y_bot)
    
    ax2.set_title('(b) Dual-Mode Training & Loss Weights', fontweight='bold', y=-0.18)

    # -------------------------------------------------------
    # Panel C: Theoretical Anchor
    # -------------------------------------------------------
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    # 绘制流形
    for i in range(1, 4):
        ellipse = patches.Ellipse((0.5, 0.4), width=i*0.25, height=i*0.15, angle=20, 
                                  fill=False, edgecolor='green', alpha=0.4 - i*0.05, lw=1.5)
        ax3.add_patch(ellipse)
    ax3.text(0.5, 0.4, 'Reasoning\nManifold', ha='center', va='center', fontsize=8, color='green')

    # 当前参数点
    tx, ty = 0.7, 0.75
    ax3.scatter([tx], [ty], color='black', s=50, zorder=10)
    ax3.text(tx + 0.05, ty, r'$\theta_t$', fontsize=12, va='center')

    # 向量 1
    ax3.arrow(tx, ty, 0.2, 0.05, head_width=0.03, head_length=0.03, 
              fc='#ff7f0e', ec='#ff7f0e', lw=2)
    # 修复点：确保 math text 正确
    ax3.text(tx + 0.1, ty + 0.08, r'$\nabla \mathcal{L}_{B}$', color='#ff7f0e', fontsize=11)
    
    # 向量 2
    ax3.arrow(tx, ty, -0.15, -0.25, head_width=0.03, head_length=0.03, 
              fc='#1f77b4', ec='#1f77b4', lw=2, linestyle='--')
    ax3.text(tx - 0.22, ty - 0.15, r'$\nabla \mathcal{L}_{A}$', color='#1f77b4', fontsize=11)
    ax3.text(tx - 0.15, ty - 0.3, '(Implicit Anchor)', color='#1f77b4', fontsize=8, ha='center')

    # 合成向量
    ax3.arrow(tx, ty, 0.05, -0.2, head_width=0.03, head_length=0.03, 
              fc='black', ec='black', lw=2.5)
    ax3.text(tx + 0.08, ty - 0.2, 'Update', color='black', fontsize=10)

    ax3.set_xlim(0.2, 1.1)
    ax3.set_ylim(0.1, 0.9)
    ax3.set_title('(c) Implicit Regularization', fontweight='bold', y=-0.18)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    plt.savefig('dha_ft_overview.pdf', bbox_inches='tight')
    plt.savefig('dha_ft_overview.png', bbox_inches='tight', dpi=300)
    print("Done! Saved as dha_ft_overview.png and .pdf")

if __name__ == "__main__":
    main()
