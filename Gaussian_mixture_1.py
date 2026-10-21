# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


test_data = pd.DataFrame({
    'has_los': [0, 1, 1, 1, 0],      # 是否有直视
    'has_diff': [0, 0, 1, 1, 1],     # 是否有衍射
    'has_reflec': [1, 0, 1, 1, 1],   # 是否有反射
    'diff_coe': [0, 0, 0.05, 0.2, 0.03],   # 衍射系数（现在不用了）
    'refl_dist': [2, 8, 8, 30, 20],   # 反射距离 → 用作 mu_refl
    'diff_dist': [0, 0, 3, 6, 4],    # 新增：衍射距离 → 用作 mu_diff；>5 则丢弃
})

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 误差横轴范围
x = np.linspace(-10, 60, 1000)

# 高斯分布参数（sigma 不变）
LOS_PARAMS = {'mu': 0.0, 'sigma': 1.5}
DIFF_SIGMA = 3.0
REFL_SIGMA = 4.0


def build_gmm_dynamic_mu_and_weight(row):
    score = 0
    w_los = 0.0
    w_diff = 0.0
    w_refl = 0.0
    mu_diff = None
    mu_refl = None

    # === LOS 成分 ===
    if row['has_los']:
        w_los = 1.0
        score += 0  # 可按需加分

    # === Reflection 成分 ===
    if row['has_reflec']:
        mu_refl = row['refl_dist']          # 直接用 refl_dist 作 μ
        w_refl = 1.0                        # 初始权重设为1
        score += 1
        # 根据 refl_dist 调整权重和分数
        if row['refl_dist'] > 5:
            w_refl += 0.1
            score += 1
        if row['refl_dist'] > 25:
            w_refl += 0.2
            score += 2
    else:
        mu_refl = 1.0  # 默认值（备用）
        w_refl = 0.0   # 无反射则权重为0

    # === Diffraction 成分 ===
    effective_has_diff = row['has_diff']
    if effective_has_diff:
        diff_dist = row['diff_dist']
        if diff_dist > 5:
            effective_has_diff = 0  # 丢弃：视为无衍射
        else:
            mu_diff = diff_dist     # 使用 diff_dist 作为 μ

    if effective_has_diff:
        w_diff = 0.1                # 基础权重
        score += 1
        if row['diff_coe'] > 0.1:   # 可选：仍用 diff_coe 增加权重
            w_diff += 0.1
            score += 1

    # 确保权重非负
    w_los = max(w_los, 0.0)
    w_diff = max(w_diff, 0.0)
    w_refl = max(w_refl, 0.0)

    # 归一化权重
    total_weight = w_los + w_diff + w_refl
    if total_weight == 0:
        total_weight = 1.0
    w_los /= total_weight
    w_diff /= total_weight
    w_refl /= total_weight

    weight_dict = {'los': w_los, 'diff': w_diff, 'refl': w_refl}
    # 构建总 PDF
    pdf = np.zeros_like(x)
    components = []
    if w_los > 0:
        comp = w_los * norm.pdf(x, LOS_PARAMS['mu'], LOS_PARAMS['sigma'])
        pdf += comp
        components.append(('LOS', comp, 0.0))
    if w_diff > 0 and mu_diff is not None:
        comp = w_diff * norm.pdf(x, mu_diff, DIFF_SIGMA)
        pdf += comp
        components.append(('Diffraction', comp, mu_diff))
    if w_refl > 0:
        comp = w_refl * norm.pdf(x, mu_refl, REFL_SIGMA)
        pdf += comp
        components.append(('Reflection', comp, mu_refl))

    # 不确定性等级
    if score <= 0:
        uncertainty = 0
    elif score <= 2:
        uncertainty = 1
    elif score <= 4:
        uncertainty = 2
    else:
        uncertainty = 3


    

    return pdf, score, uncertainty, weight_dict, mu_refl, mu_diff, components   



colors = {'LOS': 'blue', 'Diffraction': 'orange', 'Reflection': 'red'}
linestyles = {'LOS': '--', 'Diffraction': '--', 'Reflection': '--'}

for idx, (i, row) in enumerate(test_data.iterrows(), start=1):
    # 创建新图
    plt.figure(figsize=(10, 6))

    # 计算分布
    pdf, score, unc, weight_dict, mu_refl, mu_diff, components = build_gmm_dynamic_mu_and_weight(row)
    
    # 绘制每个成分
    for name, comp, mu_val in components:
        w = weight_dict['los' if name == 'LOS' else 'diff' if name == 'Diffraction' else 'refl']
        label = f'{name} (μ={mu_val:.0f}, w={w:.2f})'
        plt.plot(x, comp, color=colors[name], linestyle=linestyles[name], linewidth=2, label=label)


    # 绘制总分布
    plt.plot(x, pdf, 'k-', linewidth=2, label='Total GMM')
    plt.fill_between(x, pdf, alpha=0.3, color='gray')

    # 标题和说明
    title_parts = []
    if row['has_los']: title_parts.append("LOS")
    if row['has_diff'] and row['diff_dist'] <= 5: title_parts.append(f"Diff(μ={row['diff_dist']})")
    if row['has_reflec'] and row['refl_dist'] > 0: title_parts.append(f"Refl(μ={row['refl_dist']})")
    if row['has_diff'] and row['diff_dist'] > 5: title_parts.append(f"Diff(>{5}, 已丢弃)")

    plt.suptitle(f"样本 {idx}：信号组合 = {', '.join(title_parts) if title_parts else '无有效信号'}", fontsize=14)
    plt.title(f"权重: LOS={weight_dict['los']:.2f}, Diff={weight_dict['diff']:.2f}, Refl={weight_dict['refl']:.2f} | "
              f"Score={score}, Unc={unc}", fontsize=10, style='italic')

    plt.xlabel('伪距误差 (米)')
    plt.ylabel('概率密度')
    plt.ylim(0, 0.4)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()  