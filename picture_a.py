import matplotlib.pyplot as plt
import numpy as np

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

# 全局字体大小调大（从11→14）
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300 
# 关闭自动紧凑布局警告
plt.rcParams['figure.autolayout'] = False 

# ==========================================
# 3. 主绘图函数
# ==========================================
def main():
    # 创建画布 - 调整为适合单个图的尺寸（稍微放大画布适配大文字）
    fig, ax1 = plt.subplots(figsize=(9, 7))

    # -------------------------------------------------------
    # Panel A: Curriculum Schedule
    # -------------------------------------------------------
    steps = np.linspace(0, 100, 100)
    p_start, p_end = 0.95, 0.10
    p_hint = p_start - (steps / 100) * (p_start - p_end)

    # 绘制主曲线（线条也适当加粗）
    ax1.plot(steps, p_hint, color='#1f77b4', linewidth=3, label='$p_{hint}(t)$')
    
    # 填充颜色区域
    ax1.fill_between(steps, 0, p_hint, color='#1f77b4', alpha=0.15)
    ax1.fill_between(steps, p_hint, 1.1, color='#ff7f0e', alpha=0.15)

    # 添加文本标注（字体大小从9→12，同时调整位置适配）
    ax1.text(35, 0.3, 'Mode A:\nImitation Utilization', 
             ha='center', va='center', color='#0d4f8b', fontweight='bold', fontsize=25)
    ax1.text(65, 0.8, 'Mode B:\nSpontaneous Generation', 
             ha='center', va='center', color='#b45f06', fontweight='bold', fontsize=25)

    # 设置坐标轴范围和标签（显式指定字体大小，确保覆盖全局设置）
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Training Steps (%)', fontsize=25)
    ax1.set_ylabel('Probability ($p_{hint}$)', fontsize=25)
    
    # 调整刻度字体大小（额外放大刻度数字）
    ax1.tick_params(axis='both', labelsize=20)
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 添加图例（字体大小放大，边框适当调整）
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, fontsize=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('curriculum_schedule.pdf', bbox_inches='tight')
    plt.savefig('curriculum_schedule.png', bbox_inches='tight', dpi=300)
    print("Done! Saved as curriculum_schedule.png and .pdf")

if __name__ == "__main__":
    main()