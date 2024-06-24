import numpy as np
import math


"""
ServiceQueue 类用于模拟具有多个服务台的串联服务队列。此类提供了计算顾客在特定服务台的在店概率和离店概率的功能。

Attributes:
    entry_time (float): 顾客进入队列的时间。
    service_rates (list of float): 各服务台的服务率参数。

Methods:
    probability_in_service_desk(desk_index, current_time): 计算在给定服务台的在店概率。
    probability_leaving_service_desk(desk_index, current_time): 计算在给定服务台的离店概率。
"""
class ServiceQueue:


    def __init__(self, entry_time, service_rates):
        """
        初始化一个服务队列实例。

        Args:
            entry_time (float): 顾客进入队列的时间点。
            service_rates (list of float): 每个服务台的服务率，服务率影响处理时间。
        """
        self.entry_time = entry_time
        self.service_rates = service_rates

    def generate_dictionaries_final(self):
        """
        内部方法，用于生成每个服务台的参数字典。
        """
        mus = self.service_rates
        stages = [{mus[0]: [1]}]
        for i in range(1, len(mus)):
            current_mu = mus[i]
            previous_stage = stages[-1]
            new_stage = {}
            for mu_k, alphas in previous_stage.items():
                new_alphas = [alpha * mus[i-1] / (current_mu - mu_k) for alpha in alphas]
                new_stage[mu_k] = new_alphas
            all_current_alphas = [alpha for alphas in new_stage.values() for alpha in alphas]
            new_stage[current_mu] = [-alpha for alpha in all_current_alphas]
            stages.append(new_stage)
        return stages

    def probability_in_service_desk(self, desk_index, current_time):
        """
        计算给定服务台在当前时间的在店概率。

        Args:
            desk_index (int): 服务台索引，从1开始。
            current_time (float): 当前时间点。

        Returns:
            float: 指定服务台的在店概率。
        """
        T = current_time - self.entry_time
        stages = self.generate_dictionaries_final()
        if desk_index > len(stages):
            return 0
        ith_stage = stages[desk_index - 1]
        probability_sum = 0
        for a, b_a in ith_stage.items():
            for b in b_a:
                probability_sum += (b/a) * (math.exp(-a * T)-math.exp(-a * (T+1)))
        return probability_sum

    def probability_leaving_service_desk(self, desk_index, current_time):
        """
        计算给定服务台在当前时间的离店概率。

        Args:
            desk_index (int): 服务台索引，从1开始。
            current_time (float): 当前时间点。

        Returns:
            float: 指定服务台的离店概率。
        """
        T = current_time - self.entry_time
        in_shop_probability = self.probability_in_service_desk(desk_index, current_time)
        mu_i = self.service_rates[desk_index - 1]
        return in_shop_probability * mu_i
'''
# 使用示例：
if __name__ == "__main__":
    # 创建服务队列实例
    queue = ServiceQueue(entry_time=0, service_rates=[1, 2, 3])

    # 计算第2号服务台在时间5时的离店概率
    current_time = 5
    desk_index = 2
    probability = queue.probability_leaving_service_desk(desk_index, current_time)
    print(f"Probability of leaving the service desk {desk_index} at time {current_time}: {probability:.4f}")
'''