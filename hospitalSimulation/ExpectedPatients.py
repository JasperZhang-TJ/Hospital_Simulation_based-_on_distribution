import yaml
import pandas as pd
import numpy as np
import math

#--------------------------------------------------串联队列类----------------------------------------------------------
class ServiceQueue:
    def __init__(self, entry_time, service_rates):
        """
        初始化一个服务队列实例。

        参数:
            entry_time (float): 病人进入队列的时间点。
            service_rates (list of float): 每个科室的服务率，服务率影响处理时间。
        """
        self.entry_time = entry_time
        self.service_rates = service_rates

    def generate_dictionaries_final(self):
        """
        生成每个科室的参数字典，用于计算在店概率和离店概率。
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
        计算给定科室在当前时间的在店概率。(瞬时)

        参数:
            desk_index (int): 科室索引，从1开始。
            current_time (float): 当前时间点。

        返回:
            float: 指定科室的在店概率。
        """
        T = current_time - self.entry_time
        stages = self.generate_dictionaries_final()
        if desk_index > len(stages):
            return 0
        ith_stage = stages[desk_index - 1]
        probability_sum = 0
        for a, b_a in ith_stage.items():
            for b in b_a:
                probability_sum += b * math.exp(-a * T)
        return probability_sum

#--------------------------------------------------数据加载函数----------------------------------------------------------
def load_patient_data(file_path):
    """
    从 YAML 文件加载病人类型及其就诊顺序。

    参数:
    file_path (str): 包含病人数据的 YAML 文件路径。

    返回:
    list: 字典列表，每个字典包含病人类型和他们的就诊顺序。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

def load_patients_delay(file_path):
    """
    从 YAML 文件加载病人的允许延迟时间。

    参数:
    file_path (str): 包含病人延迟数据的 YAML 文件路径。

    返回:
    dict: 包含病人类型和允许延迟时间的字典。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

def load_patient_arrival_frequencies(file_path):
    """
    加载并转换病人到达频率。单位为人/分钟

    参数:
    file_path (str): 包含病人到达频率数据的 Excel 文件路径。

    返回:
    list: 字典列表，每个字典包含一个时间槽的病人到达频率数据。
    """
    data = pd.read_excel(file_path, index_col=0)
    return data.reset_index().rename(columns={'index': 'slot'}).to_dict(orient='records')

def load_room_data(file_path):
    """
    加载并转换科室的效率数据。将分钟基础的服务时间转换为每分钟可以服务的病人数量。

    参数:
    file_path (str): 包含科室数据的 YAML 文件路径。

    返回:
    list: 字典列表，每个字典包含科室的效率数据。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        for room in data:
            for k, v in room['efficiency'].items():
                room['efficiency'][k] = 1 / v
    return data

def get_consultation_sequence(patient_type, patient_data):
    """
    根据病人类型从 patient_data 中获取就诊顺序

    参数:
    patient_type (str): 病人类型
    patient_data (list): 包含病人类型和就诊顺序数据的列表

    返回:
    list: 包含病人就诊顺序的科室列表
    """
    for patient in patient_data:
        if patient['type'] == patient_type:
            return patient['consultation_sequence']
    return []

def generate_lambda_dictionary(patient_arrival_data):
    """
    生成到达频率的字典，其中键为 (k, m)，k 是病人种类，m 是时间槽索引。
    值为该时间槽中该种类病人的到达频率。

    参数:
    patient_arrival_data (list of dicts): 每个字典包含一个时间槽的病人到达频率数据，
                                          键为病人类型和到达频率。

    返回:
    dict: 键为 (k, m) 的字典，值为对应的到达频率。
    """
    lambda_dict = {}
    for index, slot_data in enumerate(patient_arrival_data):
        m = index
        for k in slot_data:
            if k != 'Time':
                lambda_dict[(k, m)] = slot_data[k]
    return lambda_dict

#--------------------------------------------------计算期望人数----------------------------------------------------------
def calculate_expected_patients(patient_data, room_data, lambda_dict, num_time_slots, T):
    """
    计算在每个时间点每个科室的期望人数，并逐步打印结果。

    参数:
    patient_data (list): 包含病人类型和就诊顺序数据的列表
    room_data (list): 包含科室效率数据的列表
    lambda_dict (dict): 键为 (k, m) 的字典，值为对应的到达频率
    num_time_slots (int): 时间槽总数
    T (int): 每个时间槽包含的分钟数

    返回:
    dict: 每个时间点每个科室的期望人数
    """
    expected_patients = {}  # 初始化存储期望人数的字典
    for t in range(1, num_time_slots * T + 1):  # 遍历每一个具体的分钟
        expected_patients[t] = {}  # 初始化每一分钟的期望人数字典
        for room in room_data:  # 遍历每一个科室
            room_name = room['name']
            expected_patients[t][room_name] = 0  # 初始化当前时间点该科室的期望人数为0
            for k in patient_data:  # 遍历每一种病人类型
                patient_type = k['type']
                consultation_sequence = get_consultation_sequence(patient_type, patient_data)  # 获取病人类型的就诊顺序
                if room_name in consultation_sequence:  # 如果当前科室在病人的就诊顺序中
                    index = consultation_sequence.index(room_name) + 1  # 获取当前科室在就诊顺序中的位置
                    room_efficiency = [r['efficiency'][patient_type] for r in room_data if r['name'] in consultation_sequence]  # 获取病人类型在相关科室的服务效率
                    for t_prime in range(1, t + 1):  # 遍历当前时间点之前的所有时间点
                        if (patient_type, (t_prime - 1) // T) in lambda_dict:  # 如果病人类型在时间槽 t_prime-1 中有到达频率
                            queue = ServiceQueue(entry_time=t_prime, service_rates=room_efficiency)  # 初始化服务队列
                            probability = queue.probability_in_service_desk(index, t)  # 计算病人在当前时间点 t 在当前科室的在店概率
                            # 累加期望人数：概率乘以到达频率
                            expected_patients[t][room_name] += probability * lambda_dict[(patient_type, (t_prime - 1) // T)]
            # 打印当前时间点和科室的期望人数
            print(f"Time {t}, Room {room_name}, Expected Patients: {expected_patients[t][room_name]}")
    return expected_patients  # 返回计算得到的期望人数字典


#-----------------------------------------------------主函数-------------------------------------------------------------
def main():
    #时间槽总数（一天有多少个时段可以预约）
    num_time_slots = 26

    # 每个时间槽包含的分钟数
    T = 30

    # 加载病人类型及其就诊顺序数据
    patient_data = load_patient_data('patients.yaml')

    # 加载病人的允许延迟时间
    patients_delay = load_patients_delay('patientsDelay.yaml')

    # 加载并转换病人到达频率数据，单位为人/分钟
    patient_arrival_data = load_patient_arrival_frequencies('frequences_for_calculate.xlsx')

    # 加载科室的效率数据，将分钟基础的服务时间转换为每分钟可以服务的病人数量
    room_data = load_room_data('rooms.yaml')

    # 生成到达频率字典，键为 (k, m)，值为对应的到达频率
    lambda_dict = generate_lambda_dictionary(patient_arrival_data)

    # 计算在每个时间点每个科室的期望人数
    expected_patients = calculate_expected_patients(patient_data, room_data, lambda_dict, num_time_slots, T)

    # 将结果输出为CSV文件
    df = pd.DataFrame.from_dict(expected_patients, orient='index')
    df.to_csv('expected_patients.csv')
    print("结果已保存为 'expected_patients.csv'")

if __name__ == '__main__':
    main()
