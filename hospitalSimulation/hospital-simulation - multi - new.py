import simpy  # 导入simpy模块，用于创建和管理仿真环境
import pandas as pd  # 导入pandas库，用于数据分析和操作Excel文件
import numpy as np  # 导入numpy库，用于数学计算，包括生成指数分布的随机数
import yaml  # 导入yaml库，用于解析YAML文件
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

'''
这里补充一下读入动态capacity的函数
'''
def load_capacities(filename):
    """从Excel文件加载诊室的动态容量数据"""
    df = pd.read_excel(filename)
    capacities = {col: df[col].tolist() for col in df.columns if col != 'Time'}  # 排除时间列
    return capacities


'''
1. load_yaml_data
功能: 加载YAML文件中的数据，通常包含诊室或病人的配置信息。

逻辑:
    打开指定的YAML文件。
    使用yaml.safe_load读取文件内容，将YAML格式转换成Python字典。
    返回转换后的字典，可用于仿真中的进一步处理。
'''
def load_yaml_data(filename):
    """从YAML文件加载数据，支持读取病人和诊室的配置信息"""
    with open(filename, 'r') as file:
        return yaml.safe_load(file)  # 使用yaml.safe_load来读取YAML格式的配置数据

'''
2. load_frequencies
功能: 从Excel文件中加载病人到达频率，这些数据决定了病人到达的时间间隔。

逻辑:
    使用pandas库的read_excel函数打开并读取Excel文件。
    遍历除时间列之外的每一列，将每种病人类型的到达频率存储为列表，并构建成字典格式返回。
    这样，字典的键是病人类型，值是一个列表，包含一天中每个半小时时间槽的到达频率。
'''
def load_frequencies(filename):
    """从Excel文件加载病人到达频率"""
    df = pd.read_excel(filename)  # 使用pandas读取Excel文件
    frequencies = {col: df[col].tolist() for col in df.columns[1:]}  # 将除第一列外的每列数据转换为列表，并存储为字典
    return frequencies

'''
3. Patient
功能: 定义一个病人类，存储关于病人的基本信息和就诊序列。

逻辑:
    初始化时接收仿真环境、病人类型和就诊科室顺序。
    这些信息在病人处理过程中用来引导病人访问正确的诊室并按照预定的顺序进行。
'''
class Patient:
    """定义病人类，包括病人类型和就诊流程"""
    def __init__(self, env, type, consultation_sequence):
        self.env = env  # SimPy环境
        self.type = type  # 病人类型
        self.consultation_sequence = consultation_sequence  # 病人的就诊科室顺序

'''
4. ConsultationRoom
功能: 定义诊室类，包括诊室的基本属性如名称、容量和病人处理效率。

逻辑:
    初始化时接收仿真环境、诊室名称、容量和处理效率。
    创建一个SimPy资源，表示诊室的可用性，容量限制了同时可以处理的病人数量。
'''


class ConsultationRoom:
    """
    诊室类定义，包括诊室的基本属性和数据收集功能，支持动态容量变化。

    Attributes:
        env (simpy.Environment): 仿真环境，用于调度和控制仿真进程。
        name (str): 诊室的名称。
        capacity_schedule (list): 按时间槽存储的诊室容量变化表。
        efficiency (dict): 不同病人类型的处理效率，字典形式。
        room (simpy.Resource): SimPy资源，表示诊室，用于管理病人的进入和处理。
        usage (list): 记录仿真期间每个时间点诊室正在处理的病人数。
        queue (list): 记录仿真期间每个时间点诊室排队等待的病人数。
        capacity (list): 记录仿真期间变化的动态容量（多写了）
        utilization (list): 记录仿真期间变化的动态利用率

    Methods:
        update_usage(): 每个仿真时间单位收集并记录诊室的使用情况和排队情况、记录科室的实时利用率。
        update_capacity(): 根据时间变化更新诊室容量。
    """

    def __init__(self, env, name, capacity_schedule, efficiency):
        """
        初始化诊室实例。

        参数:
        env (simpy.Environment): 传入的SimPy仿真环境。
        name (str): 诊室的名称。
        capacity_schedule (list): 每半小时更新一次的诊室容量表。
        efficiency (dict): 指定各类型病人的处理时间。
        """
        self.env = env
        self.name = name
        self.capacity_schedule = capacity_schedule
        self.efficiency = efficiency
        self.room = simpy.Resource(env, capacity=capacity_schedule[0])  # 初始容量使用第一时间槽的容量
        self.usage = []  # 初始化用于记录每个仿真时间点的资源使用情况的列表
        self.queue = []  # 初始化用于记录每个仿真时间点的排队情况的列表
        self.capacity = []  #用于存储动态变化的科室容量
        self.utilization = []  # 用于记录每个仿真时间点的利用率
        self.env.process(self.update_capacity())  # 启动一个持续的仿真过程，用于定时更新诊室容量
        self.env.process(self.update_usage())  # 启动一个持续的仿真过程，用于更新使用和队列数据
        self.capacity = []  # 记录每个仿真时间点的当前容量

    def update_usage(self):
        """追踪并更新诊室的使用和排队情况。"""
        while True:
            current_usage = self.room.count
            current_capacity = self.room.capacity
            self.usage.append(self.room.count)  # 记录当前时间点的使用情况
            self.queue.append(len(self.room.queue))  # 记录当前时间点的排队人数
            self.capacity.append(current_capacity)  # 记录当前容量
            if current_capacity > 0:
                self.utilization.append(current_usage / max(current_usage,current_capacity))  # 计算并记录利用率
            else:
                self.utilization.append(0)  # 防止除以零的情况
            yield self.env.timeout(1)  # 暂停1个仿真时间单位，之后继续执行该方法

    def update_capacity(self):
        """每半小时根据预设表更新诊室的容量。"""
        for capacity in self.capacity_schedule:
            yield self.env.timeout(30)  # 每30分钟更新一次容量
            self.room._capacity = capacity  # 更新资源容量
            self.room._trigger_put(None)  # 检查是否有等待的请求现在可以处理


'''
5. schedule_patient_arrivals
功能: 根据提供的频率数据安排病人到达，使用指数分布模拟到达间隔。

逻辑:
    遍历一天中的每个半小时时间槽。
    对于每个时间槽，根据病人类型和对应的到达频率计算到达间隔。
    如果当前仿真时间仍在该时间槽内，按计算出的间隔安排病人到达，并立即开始处理病人的就诊过程。
    这个函数确保病人到达的时间和频率与输入数据相符。
'''
def schedule_patient_arrivals(env, type, frequency, consultation_sequence, rooms, patient_counts, all_patient_visits):
    """根据指定频率安排病人到达，使用指数分布来模拟到达间隔"""
    for slot in range(26):  # 遍历一天的26个时间槽
        rate = frequency[type][slot]  # 获取当前时间槽的到达频率
        while env.now < slot * 30 + 30:  # 在当前时间槽内持续生成病人
            arrival_interval = np.random.exponential(1 / (rate)) if rate > 0 else float('inf')
            yield env.timeout(arrival_interval)  # 等待到下一个病人到达时间
            if env.now < slot * 30 + 30:  # 确保仍在时间槽内
                patient_counts[type] += 1  # 增加该类型病人的计数
                env.process(dispatch(env, Patient(env, type, consultation_sequence), rooms, all_patient_visits))


'''
6. dispatch
功能: 处理病人在各个诊室的就诊过程。

逻辑:
    对于病人的每一个就诊科室，病人尝试进入诊室。
    如果诊室资源（如床位）可用，则病人进入并根据诊室的效率进行处理，否则病人等待直到资源可用。
    每完成一次就诊，打印病人的处理信息。
'''

#-----------------------------------------------------------------------------------------------------------------------这里是指派策略函数 如果有指派策略需要修改 可以在此处加入（此处默认是FCFS）
def dispatch(env, patient, rooms, all_patient_visits):
    """
    处理病人的就诊流程，并记录就诊时间。

    参数:
    env (simpy.Environment): SimPy仿真环境。
    patient (Patient): 病人实例。
    rooms (dict): 包含所有诊室的字典。
    all_patient_visits (list): 存储所有病人就诊时间的列表。
    """
    patient_visit_times = []  # 记录病人每个科室的就诊时间
    for room_name in patient.consultation_sequence:
        room = rooms[room_name]
        with room.room.request() as request:
            yield request
            start_time = env.now
            process_time = np.random.exponential(scale=room.efficiency[patient.type])
            yield env.timeout(process_time)
            end_time = env.now
            # 记录病人的就诊时间信息
            patient_visit_times.append({
                'type': patient.type,
                'room_name': room_name,
                'start_time': start_time,
                'end_time': end_time
            })
    all_patient_visits.append(patient_visit_times)
            #print(f"Patient of type {patient.type} processed in {room.name} from {start_time} to {end_time}")


'''
    仿真结束后，添加一个新的函数来绘制利用率图表。
'''
def plot_utilization(rooms):
    plt.figure(figsize=(16, 8))
    for room_name, room in rooms.items():
        times = range(len(room.utilization))
        plt.plot(times, room.utilization, label=f'{room_name} Utilization')

    plt.xlabel('Time (minutes)')
    plt.ylabel('Utilization Rate')
    plt.title('Utilization Rate Over Time for Each Room')
    plt.legend()
    plt.show()

'''
    函数用来绘制散点图（病人滞留时间的散点图）
'''
def plot_patient_stay(all_patient_visits, color_dict):
    entry_times = []
    stay_lengths = []
    colors = []

    for patient_visits in all_patient_visits:
        if not patient_visits:
            continue
        entry_time = min(visit['start_time'] for visit in patient_visits)
        leave_time = max(visit['end_time'] for visit in patient_visits)
        stay_length = leave_time - entry_time

        entry_times.append(entry_time)
        stay_lengths.append(stay_length)
        patient_type = patient_visits[0]['type']
        colors.append(color_dict[patient_type])

    plt.scatter(entry_times, stay_lengths, c=colors, alpha=0.5)
    plt.title('Patient Stay Time vs Arrival Time')
    plt.xlabel('Arrival Time (min)')
    plt.ylabel('Stay Length (min)')
    plt.grid(True)
    plt.show()

'''
    函数用来绘制需要的几张图
'''


'''
7. simulation
功能: 主仿真函数，设置仿真环境，并运行整个医院的病人处理流程。

逻辑:
    创建一个SimPy仿真环境。
    加载并创建所有必要的数据和仿真实体，包括诊室和病人。
    安排每种病人按照频率到达，并处理其就诊。
    运行仿真直到指定的时间（一天的结束）。
'''

def simulation(patients_file, rooms_file, capacities_file, frequencies_file):
    # 初始化SimPy环境，这是进行所有仿真操作的基础
    env = simpy.Environment()

    # 从YAML文件加载诊室数据
    rooms_data = load_yaml_data(rooms_file)

    capacities = load_capacities(capacities_file)  # 加载新的容量数据

    # 从加载的诊室数据创建诊室实体，每个诊室是一个ConsultationRoom对象
    rooms = {}
    for room in rooms_data:
        room_name = room['name']
        capacity_schedule = capacities[room_name]  # 从加载的数据中获取相应诊室的容量表
        rooms[room_name] = ConsultationRoom(env, room_name, capacity_schedule, room['efficiency'])

    # 从YAML文件加载病人数据
    patients_data = load_yaml_data(patients_file)

    #print(patients_data)#检查病人数据输入

    # 从Excel文件加载病人到达频率数据
    frequencies = load_frequencies(frequencies_file)

    #print(frequencies)#检查病人数据输入


    patient_types = set(patient['type'] for patient in patients_data)

    patient_counts = {ptype: 0 for ptype in patient_types}
    all_patient_visits = []

    # 为每种病人类型创建并安排到达进程
    for patient_data in patients_data:
        patient_type = patient_data['type']
        env.process(
            schedule_patient_arrivals(env, patient_type, frequencies, patient_data['consultation_sequence'], rooms,
                                      patient_counts, all_patient_visits)
        )

    # 运行仿真直到1440分钟，代表一整天的时间
    env.run(until=1440)#-------------------------------------------------------------------------------------------------改成了26个时间槽 但是这里还是要延后 要不会造成漏斗形状的滞留时间

    # 创建一个字典来存储所有需要的绘图数据
    simulation_data = {
        'patient_visits': all_patient_visits,
        'patient_counts': patient_counts,
        'rooms': {name: {'utilization': room.utilization, 'queue': room.queue} for name, room in rooms.items()},
        'patient_types': patient_types
    }

    return simulation_data


def run_multiple_simulations(num_simulations, patients_file, rooms_file, capacities_file, frequencies_file):
    simulations_data = []
    for _ in range(num_simulations):
        print(_)
        result = simulation(patients_file, rooms_file, capacities_file, frequencies_file)
        simulations_data.append(result)
    return simulations_data


# 计算并打印纵轴坐标的和的函数
def print_vertical_sums(results):
    # 汇总并打印病人类型占比图的纵轴坐标和
    total_patient_counts = {ptype: 0 for ptype in results[0]['patient_types']}
    for data in results:
        for ptype, count in data['patient_counts'].items():
            total_patient_counts[ptype] += count
    patient_counts_sum = sum(total_patient_counts.values())
    print(f'病人类型占比图的纵轴坐标和: {patient_counts_sum}')

    # 汇总并打印非零利用率图的纵轴坐标和
    avg_non_zero_utilization_sum = 0
    for room_name in results[0]['rooms'].keys():
        for t in range(len(results[0]['rooms'][room_name]['utilization'])):
            non_zero_utilizations = [
                data['rooms'][room_name]['utilization'][t]
                for data in results
                if data['rooms'][room_name]['utilization'][t] > 0
            ]
            if non_zero_utilizations:
                avg_non_zero_utilization_sum += sum(non_zero_utilizations)
    print(f'非零利用率图的纵轴坐标和: {avg_non_zero_utilization_sum}')

    # 汇总并打印平均利用率图的纵轴坐标和
    avg_utilization_sum = 0
    for room_name in results[0]['rooms'].keys():
        avg_utilization_sum += sum(np.mean([data['rooms'][room_name]['utilization'] for data in results], axis=0))
    print(f'平均利用率图的纵轴坐标和: {avg_utilization_sum}')

    # 汇总并打印排队数量图的纵轴坐标和
    avg_queue_sum = 0
    for room_name in results[0]['rooms'].keys():
        avg_queue_sum += sum(np.mean([data['rooms'][room_name].get('queue', []) for data in results], axis=0))
    print(f'排队数量图的纵轴坐标和: {avg_queue_sum}')

    # 汇总并打印滞留时间散点图的纵轴坐标和
    stay_lengths = []
    for data in results:
        for patient_visits in data['patient_visits']:
            if not patient_visits:
                continue
            leave_time = max(visit['end_time'] for visit in patient_visits)
            entry_time = min(visit['start_time'] for visit in patient_visits)
            stay_lengths.append(leave_time - entry_time)
    stay_lengths_sum = sum(stay_lengths)
    print(f'滞留时间散点图的纵轴坐标和: {stay_lengths_sum}')

def main():
    # 指定文件路径
    patients_file = 'patients.yaml'
    rooms_file = 'rooms.yaml'
    capacities_file = 'Room_Capacities.xlsx'
    frequencies_file = 'frequences_for_calculate.xlsx'

    # 运行多次仿真
    num_simulations = 1000
    # ------------------------------------------------------------------------------------------------------------------此处设置实验重复次数
    results = run_multiple_simulations(num_simulations, patients_file, rooms_file, capacities_file, frequencies_file)

    #print(results)

    # 统计病人类型占比
    total_patient_counts = {ptype: 0 for ptype in results[0]['patient_types']}
    for data in results:
        for ptype, count in data['patient_counts'].items():
            total_patient_counts[ptype] += count
    # 病人类型占比图
    plt.figure(figsize=(10, 8))
    plt.pie([count for count in total_patient_counts.values()], labels=[ptype for ptype in total_patient_counts.keys()], autopct='%1.1f%%')
    plt.title('Percentage of Each Patient Type Across Simulations')
    plt.show()

    # 从所有仿真结果中提取和计算非零利用率的平均值
    avg_non_zero_utilization = {room_name: [] for room_name in results[0]['rooms'].keys()}
    for room_name in avg_non_zero_utilization.keys():
        for t in range(len(results[0]['rooms'][room_name]['utilization'])):
            non_zero_utilizations = [
                data['rooms'][room_name]['utilization'][t]
                for data in results
                if data['rooms'][room_name]['utilization'][t] > 0
            ]
            if non_zero_utilizations:
                avg_non_zero_utilization[room_name].append(np.mean(non_zero_utilizations))
            else:
                avg_non_zero_utilization[room_name].append(0)
    # 绘制每个科室的非零利用率平均值的折线图
    plt.figure(figsize=(16, 8))
    for room_name, utilizations in avg_non_zero_utilization.items():
        plt.plot(utilizations, label=f'{room_name} Non-Zero Utilization')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Average Non-Zero Utilization Rate')
    plt.title('Adjusted Utilization Rate for Each Room Across Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 统计各诊室的平均利用率变化
    avg_utilization = {room_name: np.mean([data['rooms'][room_name]['utilization'] for data in results], axis=0) for
                       room_name in results[0]['rooms'].keys()}
    # 利用率图
    plt.figure(figsize=(16, 8))
    for room_name, util in avg_utilization.items():
        plt.plot(util, label=f'{room_name} Utilization')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Utilization Rate')
    plt.title('Average Utilization Rate Over Time for Each Room')
    plt.legend()
    plt.show()

    # 统计各诊室处理病人的平均数量和排队数量
    avg_queue = {room_name: np.mean([data['rooms'][room_name].get('queue', []) for data in results], axis=0) for
                 room_name in results[0]['rooms'].keys()}
    # 绘制各诊室平均处理病人数量和排队数量图
    plt.figure(figsize=(16, 8))
    for room_name in avg_queue.keys():
        if avg_queue[room_name].size > 0:  # 确保列表不为空
            plt.plot(avg_queue[room_name], label=f'{room_name} Queue', linestyle='--')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Number of Patients')
    plt.title('Average utilization and Queue Length Over Time for Each Room')
    plt.legend()
    plt.show()

    # 统计所有病人的滞留时间和进入时间
    entry_times = []
    stay_lengths = []
    colors = []
    color_labels = {}
    # 创建颜色映射，确保每种类型的颜色一致性
    color_map = {ptype: plt.cm.tab20(i / len(total_patient_counts)) for i, ptype in
                 enumerate(total_patient_counts.keys())}
    for data in results:
        for patient_visits in data['patient_visits']:
            if not patient_visits:
                continue
            entry_time = min(visit['start_time'] for visit in patient_visits)
            leave_time = max(visit['end_time'] for visit in patient_visits)
            stay_length = leave_time - entry_time
            entry_times.append(entry_time)
            stay_lengths.append(stay_length)
            patient_type = patient_visits[0]['type']
            colors.append(color_map[patient_type])
            if patient_type not in color_labels:
                color_labels[patient_type] = color_map[patient_type]
    # 计算滞留时间的平均值和标准差
    mean_stay_length = np.mean(stay_lengths)
    std_stay_length = np.std(stay_lengths)
    # 滞留时间散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(entry_times, stay_lengths, c=colors, s=10, alpha=0.5)  # 小点，透明度为0.5
    # 添加平均线及±3σ线
    plt.axhline(y=mean_stay_length, color='r', linestyle='--')
    plt.axhline(y=mean_stay_length + 3 * std_stay_length, color='r', linestyle=':', label='Mean + 3σ')
    plt.text(max(entry_times) * 0.95, mean_stay_length, f'Mean: {mean_stay_length:.2f} min', color='red',
             verticalalignment='bottom')
    plt.text(max(entry_times) * 0.95, mean_stay_length + 3 * std_stay_length,
             f'+3σ: {mean_stay_length + 3 * std_stay_length:.2f} min', color='red', verticalalignment='bottom')
    # 为每种病人类型创建图例
    for label, color in color_labels.items():
        plt.scatter([], [], c=[color], label=label, s=10)
    plt.title('Patient Stay Time vs Arrival Time Across Simulations')
    plt.xlabel('Arrival Time (min)')
    plt.ylabel('Stay Length (min)')
    plt.grid(True)
    plt.legend(title="Patient Type")
    plt.show()

    # 打印图表中所有纵轴坐标的和
    print_vertical_sums(results)

if __name__ == '__main__':
    main()

