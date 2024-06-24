import simpy  # 导入simpy模块，用于创建和管理仿真环境
import pandas as pd  # 导入pandas库，用于数据分析和操作Excel文件
import numpy as np  # 导入numpy库，用于数学计算，包括生成指数分布的随机数
import yaml  # 导入yaml库，用于解析YAML文件
import matplotlib.pyplot as plt

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
        self.capacity = []  # 用于存储动态变化的科室容量
        self.utilization = []  # 用于记录每个仿真时间点的利用率
        self.env.process(self.update_usage())  # 启动一个持续的仿真过程，用于更新使用和队列数据
        self.env.process(self.update_capacity())  # 启动一个持续的仿真过程，用于定时更新诊室容量
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
                self.utilization.append(current_usage / current_capacity)  # 计算并记录利用率
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
def schedule_patient_arrivals(env, type, frequency, consultation_sequence, rooms, patient_counts):
    """根据指定频率安排病人到达，使用指数分布来模拟到达间隔"""
    for slot in range(48):  # 遍历一天的48个时间槽
        rate = frequency[type][slot]  # 获取当前时间槽的到达频率
        while env.now < slot * 30 + 30:  # 在当前时间槽内持续生成病人
            arrival_interval = np.random.exponential(1/(rate)) if rate > 0 else float('inf')
            yield env.timeout(arrival_interval)  # 等待到下一个病人到达时间
            if env.now < slot * 30 + 30:  # 确保仍在时间槽内
                patient_counts[type] += 1  # 增加该类型病人的计数
                env.process(dispatch(env, Patient(env, type, consultation_sequence), rooms))
'''
6. dispatch
功能: 处理病人在各个诊室的就诊过程。

逻辑:
    对于病人的每一个就诊科室，病人尝试进入诊室。
    如果诊室资源（如床位）可用，则病人进入并根据诊室的效率进行处理，否则病人等待直到资源可用。
    每完成一次就诊，打印病人的处理信息。
'''

#------------------------------------------------------------------------------------------------这里是指派策略函数 如果有指派策略需要修改 可以在此处加入（此处默认是FCFS）
def dispatch(env, patient, rooms):
    """
    处理病人的就诊流程。此函数负责模拟病人依次访问其就诊科室序列中的每一个诊室，
    并在每个诊室中根据病人类型和诊室的处理效率进行处理。

    参数:
    env (simpy.Environment): SimPy仿真环境，用于调度和管理仿真时间。
    patient (Patient): 一个病人实例，包含病人类型和就诊科室顺序。
    rooms (dict): 包含所有诊室的字典，键为诊室名称，值为ConsultationRoom实例。

    过程:
    1. 遍历病人的就诊科室序列。
    2. 对于每一个科室，尝试使用SimPy资源请求进入该诊室。
    3. 等待直到资源（诊室）可用。
    4. 一旦资源可用，根据诊室的处理效率和病人的类型进行处理。
    5. 打印处理信息，显示病人类型、诊室名称和处理时间。
    """
    for room_name in patient.consultation_sequence:
        room = rooms[room_name]  # 从诊室字典中获取当前需要访问的诊室对象

        # 使用with语句请求诊室资源，这确保了在请求完成后自动释放资源
        with room.room.request() as request:
            yield request  # 发出资源请求，等待直到诊室可用

            # 诊室资源可用后，模拟病人在诊室的处理时间
            # 生成一个符合负指数分布的随机处理时间
            # efficiency 字典中存储的是平均处理时间，因此我们将其作为 scale 参数
            process_time = np.random.exponential(scale=room.efficiency[patient.type])#--------------------------------------------这里调整不同病人在科室中就诊时间的分布类型（这里的rate代表的是每个病人的平均处理时间/分钟）

            yield env.timeout(process_time)  # 使用生成的随机处理时间

            # 输出处理信息，包括病人类型、诊室名称和当前仿真时间
            print(f"Patient of type {patient.type} processed in {room.name} at time {env.now}")


def simulation(env, rooms, patients_file, frequencies_file, patient_counts):
    # 从YAML文件加载病人数据
    patients_data = load_yaml_data(patients_file)

    # 从Excel文件加载病人到达频率数据
    frequencies = load_frequencies(frequencies_file)

    # 为每种病人类型创建并安排到达进程
    for patient_data in patients_data:
        patient_type = patient_data['type']
        # 初始化病人计数
        if patient_type not in patient_counts:
            patient_counts[patient_type] = 0
        # 安排病人到达
        env.process(
            schedule_patient_arrivals(env, patient_type, frequencies, patient_data['consultation_sequence'],
                                      rooms, patient_counts))
    # 运行仿真直到1440分钟，代表一整天的时间
    env.run(until=1440)

# 确保之前定义的类和函数（如 ConsultationRoom, load_yaml_data 等）都已经正确加载。

def run_multiple_simulations(n, patients_file, rooms_file, capacities_file, frequencies_file):
    # 初始化数据结构以存储累计结果
    total_usage = {}
    total_queue = {}
    total_utilization = {}
    counter=0
    total_patient_counts = {}  # 新增字典，存储每种病人类型的累计人数

    for _ in range(n):
        env = simpy.Environment()
        rooms_data = load_yaml_data(rooms_file)
        capacities = load_capacities(capacities_file)

        rooms = {}
        patient_counts = {}  # 存储单次仿真中各类型病人的数量
        for room in rooms_data:
            room_name = room['name']
            capacity_schedule = capacities[room_name]
            rooms[room_name] = ConsultationRoom(env, room_name, capacity_schedule, room['efficiency'])

        # 执行仿真，并传递病人数量统计字典
        simulation(env, rooms, patients_file, frequencies_file, patient_counts)
        counter = counter+1
        print(counter)

        # 累加每个诊室的使用情况、排队情况和利用率
        for room_name, room in rooms.items():
            if room_name not in total_usage:
                total_usage[room_name] = np.array(room.usage)
                total_queue[room_name] = np.array(room.queue)
                total_utilization[room_name] = np.array(room.utilization)
            else:
                total_usage[room_name] += np.array(room.usage)
                total_queue[room_name] += np.array(room.queue)
                total_utilization[room_name] += np.array(room.utilization)

    # 计算每种病人类型的平均到达人数
    average_patient_counts = {ptype: total / n for ptype, total in total_patient_counts.items()}

    # 生成并显示每种病人类型的平均百分比饼图
    plt.figure(figsize=(10, 8))
    plt.pie([count for count in average_patient_counts.values()],
            labels=[ptype for ptype in average_patient_counts.keys()],
            autopct='%1.1f%%')
    plt.title('Average Percentage of Each Patient Type per Day Across Simulations')
    plt.show()

    # Plot average usage
    plt.figure(figsize=(16, 8))
    for room_name in total_usage:
        times = range(len(total_usage[room_name]))
        plt.plot(times, total_usage[room_name] / n, label=f'{room_name} average usage')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Average number of patients being treated')
    plt.title('Average Usage Over Time for Each Room Across Simulations')
    plt.legend()
    plt.show()

    # Plot average queue length
    plt.figure(figsize=(16, 8))
    for room_name in total_queue:
        times = range(len(total_queue[room_name]))
        plt.plot(times, total_queue[room_name] / n, label=f'{room_name} average queue', linestyle='--')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Average queue length')
    plt.title('Average Queue Length Over Time for Each Room Across Simulations')
    plt.legend()
    plt.show()

    # Plot average utilization
    plt.figure(figsize=(16, 8))
    for room_name in total_utilization:
        times = range(len(total_utilization[room_name]))
        plt.plot(times, total_utilization[room_name] / n, label=f'{room_name} average utilization', linestyle='-.')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Utilization rate')
    plt.title('Average Utilization Over Time for Each Room Across Simulations')
    plt.legend()
    plt.show()

# 定义文件路径
patients_file = 'patients.yaml'
rooms_file = 'rooms.yaml'
capacities_file = 'Room_Capacities.xlsx'
frequencies_file = 'Patient_Arrival_Frequencies.xlsx'

# 调用新函数运行1000次仿真
run_multiple_simulations(100, patients_file, rooms_file, capacities_file, frequencies_file)