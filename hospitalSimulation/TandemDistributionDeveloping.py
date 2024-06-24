import numpy as np
import matplotlib.pyplot as plt
import math

# 函数生成指定阶段的参数字典
def generate_dictionaries_final(mus):
    stages = [{mus[0]: [1]}]  # 初始阶段，键为第一个mu值，值为列表[1]
    for i in range(1, len(mus)):
        current_mu = mus[i]  # 当前mu值
        previous_stage = stages[-1]  # 上一阶段的字典
        new_stage = {}  # 创建新阶段的空字典
        for mu_k, alphas in previous_stage.items():
            # 更新alphas，根据公式计算新的alpha值
            new_alphas = [alpha * mus[i-1] / (current_mu - mu_k) for alpha in alphas]
            new_stage[mu_k] = new_alphas
        # 收集并更新所有当前alpha的负值
        all_current_alphas = [alpha for alphas in new_stage.values() for alpha in alphas]
        new_stage[current_mu] = [-alpha for alpha in all_current_alphas]
        stages.append(new_stage)  # 将新阶段添加到阶段列表中
    return stages

# 计算给定阶段和时间T的概率
def probabilityIn_ith_shop(mu_values, i, T):
    stages = generate_dictionaries_final(mu_values)  # 生成所有阶段的字典
    if i > len(stages):
        return 0  # 如果i超出阶段数，返回概率0
    ith_stage = stages[i-1]  # 获取第i阶段的字典
    probability_sum = 0
    for a, b_a in ith_stage.items():
        for b in b_a:
            # 计算并累加负指数概率
            probability_sum += b * math.exp(-a * T)
    return probability_sum

# 计算给定阶段和时间T离开商店的概率
def probabilityLeaving_ith_shop(mu_values, i, T):
    in_shop_probability = probabilityIn_ith_shop(mu_values, i, T)  # 获取在店概率
    mu_i = mu_values[i - 1]  # 获取当前阶段的mu值
    return in_shop_probability * mu_i  # 返回离店概率


# 模拟指定阶段数的指数分布和
def simulate_exponential_sums(mu_values, i, T_values, num_samples=10000):
    mu_values = mu_values[:i]  # 限制到第i个阶段的mu值
    sums = []
    for _ in range(num_samples):
        samples = [np.random.exponential(1.0 / mu) for mu in mu_values]
        sums.append(sum(samples))  # 计算每次模拟的总和
    return sums


'''
# 绘制理论概率和模拟数据密度的对比图
def plot_probability_comparison(mu_values, i, T_values, num_samples=10000):
    theoretical_probs = [probabilityLeaving_ith_shop(mu_values, i, T) for T in T_values]  # 理论概率
    simulated_data = simulate_exponential_sums(mu_values, i, T_values, num_samples)  # 模拟数据

    plt.figure(figsize=(12, 6))
    plt.plot(T_values, theoretical_probs, label='Theoretical Probability (Stage {})'.format(i), color='red', linewidth=2)
    plt.hist(simulated_data, bins=50, density=True, alpha=0.5, color='blue', label='Simulated Density (Stage {})'.format(i))
    plt.title('Theoretical Probability and Simulated Density Comparison for Stage {}'.format(i))
    plt.xlabel('T')
    plt.ylabel('Probability / Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# 参数设置
mu_values_test = [6, 2, 3, 4, 5]  # mu值列表
i_stage = 3  # 指定阶段数
T_values_test = np.linspace(0, 5, 100)  # T值的范围
num_samples_test = 10000  # 模拟的样本数量

# 执行绘图函数
plot_probability_comparison(mu_values_test, i_stage, T_values_test, num_samples_test)
'''

def plot_probability_accumulation(mu_values, T_values):
    colors = ['purple', 'blue', 'green', 'orange', 'red']  # 阶段颜色
    plt.figure(figsize=(12, 6))
    cumulative_prob = np.zeros_like(T_values)

    for i in range(1, len(mu_values) + 1):
        stage_probs = np.array([probabilityIn_ith_shop(mu_values, i, T) for T in T_values])
        cumulative_prob += stage_probs
        plt.fill_between(T_values, cumulative_prob, color=colors[i-1], alpha=0.5, label=f'Stage {i} Accumulated')

    plt.title('Accumulated In-Shop Probability for Each Stage')
    plt.xlabel('T')
    plt.ylabel('Accumulated Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

mu_values_test = [6, 2, 3, 4, 5]
T_values_test = np.linspace(0, 5, 100)

plot_probability_accumulation(mu_values_test, T_values_test)