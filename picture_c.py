import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.patheffects as pe # 用于文字描边效果

# ==========================================
# 1. 全局绘图设置
# ==========================================
try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Songti SC']
    plt.rcParams['mathtext.fontset'] = 'stix' # 数学公式字体更接近 LaTeX
except Exception as e:
    pass

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 300 

# ==========================================
# 辅助函数：绘制带描边的文字（防遮挡）
# ==========================================
def text_with_halo(ax, x, y, text, color='black', fontsize=10, ha='center', va='center', **kwargs):
    t = ax.text(x, y, text, color=color, fontsize=fontsize, ha=ha, va=va, **kwargs)
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')]) # 白色描边

# ==========================================
# 辅助函数：绘制漂亮的向量箭头
# ==========================================
def draw_fancy_arrow(ax, x, y, dx, dy, color, label=None, label_pos='end', label_offset=(0,0), linestyle='-'):
    # 使用 FancyArrowPatch 获得更好的箭头形状
    arrow = patches.FancyArrowPatch(
        posA=(x, y), posB=(x+dx, y+dy),
        arrowstyle='-|>,head_length=6,head_width=4', # 现代箭头样式
        connectionstyle="arc3,rad=0",
        color=color,
        lw=2,
        linestyle=linestyle,
        zorder=20
    )
    ax.add_patch(arrow)
    
    # 计算标签位置
    if label:
        if label_pos == 'end':
            lx, ly = x + dx + label_offset[0], y + dy + label_offset[1]
        elif label_pos == 'mid':
            lx, ly = x + dx/2 + label_offset[0], y + dy/2 + label_offset[1]
        elif label_pos == 'start':
            lx, ly = x + label_offset[0], y + label_offset[1]
            
        text_with_halo(ax, lx, ly, label, color=color, fontsize=12, fontweight='bold')

# ==========================================
# 主绘图函数
# ==========================================
def main():
    # 设定正方形画布，紧凑布局
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.axis('off')

    # -------------------------------------------------------
    # 1. 绘制推理流形 (Reasoning Manifold) - 渐变填充效果
    # -------------------------------------------------------
    center = (0.45, 0.45)
    
    # 绘制三层，制造"深坑"的立体感
    # 最外层 - 浅色
    e1 = patches.Ellipse(center, width=1.0, height=0.6, angle=30, 
                         facecolor='#e5f5e0', edgecolor='#a1d99b', lw=1, alpha=0.8, zorder=1)
    # 中间层
    e2 = patches.Ellipse(center, width=0.7, height=0.42, angle=30, 
                         facecolor='#c7e9c0', edgecolor='#74c476', lw=1, alpha=0.8, zorder=2)
    # 最内层 - 深色核心
    e3 = patches.Ellipse(center, width=0.4, height=0.24, angle=30, 
                         facecolor='#a1d99b', edgecolor='#41ab5d', lw=1.2, alpha=0.8, zorder=3)
    
    ax.add_patch(e1)
    ax.add_patch(e2)
    ax.add_patch(e3)
    
    text_with_halo(ax, 0.45, 0.45, 'Reasoning\nManifold', color='#006d2c', fontsize=9, zorder=5)

    # -------------------------------------------------------
    # 2. 绘制当前参数点 theta_t
    # -------------------------------------------------------
    tx, ty = 0.65, 0.70 # 起点位置
    
    # 画一个带白边的黑点
    ax.scatter([tx], [ty], color='black', s=60, zorder=30, edgecolors='white', linewidth=1.5)
    text_with_halo(ax, tx + 0.04, ty + 0.02, r'$\theta_t$', fontsize=14, ha='left', zorder=30)

    # -------------------------------------------------------
    # 3. 绘制向量
    # -------------------------------------------------------
    
    # 向量 B (Exploration): 橙色，向外探索
    # 更加水平向右，代表学习新知识
    dx_b, dy_b = 0.22, 0.05
    draw_fancy_arrow(ax, tx, ty, dx_b, dy_b, color='#d95f02', 
                     label=r'$\nabla \mathcal{L}_{B}$', label_pos='end', label_offset=(0.02, 0.02))
    text_with_halo(ax, tx+0.12, ty+0.1, '(New Skill)', color='#d95f02', fontsize=8, zorder=25)

    # 向量 A (Anchor): 蓝色，指向流形中心，虚线
    # 这是一个恢复力
    dx_a, dy_a = -0.12, -0.22 
    draw_fancy_arrow(ax, tx, ty, dx_a, dy_a, color='#1f78b4', linestyle='--',
                     label=r'$\nabla \mathcal{L}_{A}$', label_pos='end', label_offset=(-0.08, -0.05))
    text_with_halo(ax, tx-0.08, ty-0.12, '(Anchor)', color='#1f78b4', fontsize=8, zorder=25)

    # 向量 Update: 黑色，两者的合成
    # 根据向量加法原则微调
    dx_u, dy_u = dx_b * 0.5 + dx_a * 0.8, dy_b * 0.5 + dy_a * 0.8 # 稍微加权的合成
    # 让它看起来在两个向量中间
    dx_u, dy_u = 0.08, -0.15 
    
    draw_fancy_arrow(ax, tx, ty, dx_u, dy_u, color='#333333', 
                     label='Update', label_pos='end', label_offset=(0.02, -0.05))

    # -------------------------------------------------------
    # 4. 最终微调
    # -------------------------------------------------------
    # 设置显示范围，留出一点呼吸感
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.15, 0.9)

    # 保存
    plt.savefig('implicit_regularization_polished.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig('implicit_regularization_polished.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    print("Done! Figure saved.")
    plt.show()

if __name__ == "__main__":
    main()
