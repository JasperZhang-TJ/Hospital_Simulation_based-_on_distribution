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
    功能: 
        加载YAML文件中的数据，通常包含诊室或病人的配置信息。
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
    功能: 
        从Excel文件中加载病人到达频率，这些数据决定了病人到达的时间间隔。
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
    功能: 
        定义一个病人类，存储关于病人的基本信息和就诊序列。
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
    功能: 
        定义诊室类，包括诊室的基本属性如名称、容量和病人处理效率。
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
    功能: 
        根据提供的频率数据安排病人到达，使用指数分布模拟到达间隔。
    逻辑:
        遍历一天中的每个半小时时间槽。
        对于每个时间槽，根据病人类型和对应的到达频率计算到达间隔。
        如果当前仿真时间仍在该时间槽内，按计算出的间隔安排病人到达，并立即开始处理病人的就诊过程。
        这个函数确保病人到达的时间和频率与输入数据相符。
'''
def schedule_patient_arrivals(env, type, frequency, consultation_sequence, rooms, patient_counts, all_patient_visits):
    """根据指定频率安排病人到达，使用指数分布来模拟到达间隔"""
    for slot in range(26):  # 遍历一天的26个时间槽----------------------------------------------------------------------------------------此处改变时间槽的数量
        rate = frequency[type][slot]  # 获取当前时间槽的到达频率
        while env.now < slot * 30 + 30:  # 在当前时间槽内持续生成病人
            arrival_interval = np.random.exponential(1 / (rate)) if rate > 0 else float('inf')#------------------------------------------到达时间间隔的生成
            yield env.timeout(arrival_interval)  # 等待到下一个病人到达时间
            if env.now < slot * 30 + 30:  # 确保仍在时间槽内
                patient_counts[type] += 1  # 增加该类型病人的计数
                env.process(dispatch(env, Patient(env, type, consultation_sequence), rooms, all_patient_visits))


'''
6. dispatch
    功能: 
        处理病人在各个诊室的就诊过程。
    逻辑:
        对于病人的每一个就诊科室，病人尝试进入诊室。
        如果诊室资源（如床位）可用，则病人进入并根据诊室的效率进行处理，否则病人等待直到资源可用。
        每完成一次就诊，打印病人的处理信息。
'''

#------------------------------------------------------------------------------------------------这里是指派策略函数 如果有指派策略需要修改 可以在此处加入（此处默认是FCFS）
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
            process_time = np.random.exponential(scale=room.efficiency[patient.type])#---------------------------------------就诊时间的随机生成
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
7. simulation
    功能: 
        主仿真函数，设置仿真环境，并运行整个医院的病人处理流程。
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

    # 创建颜色字典
    patient_types = set(patient['type'] for patient in patients_data)
    colors = list(mcolors.TABLEAU_COLORS)  # 使用matplotlib的表格颜色
    color_dict = {ptype: colors[i] for i, ptype in enumerate(patient_types)}

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
    env.run(until=780)#-------------------------------------------------------改成了26个时间槽

    plot_patient_stay(all_patient_visits, color_dict)

    print(all_patient_visits)  # 输出所有病人的就诊时间数据

    # 生成饼图显示各类型病人的百分比
    plt.figure(figsize=(10, 8))
    plt.pie([count for count in patient_counts.values()], labels=[ptype for ptype in patient_counts.keys()],
            autopct='%1.1f%%')
    plt.title('Percentage of Each Patient Type per Day')
    plt.show()


    # 开始绘图，设置图表大小为16英寸宽，8英寸高
    plt.figure(figsize=(16, 8))

    # 对每个诊室绘制使用情况和排队情况的曲线
    for room_name, room in rooms.items():
        times = range(len(room.usage))  # 创建时间轴数据，对应每个仿真时间点
        plt.plot(times, room.usage, label=f'{room_name} usage')  # 绘制诊室使用情况的曲线
        plt.plot(times, room.queue, label=f'{room_name} queue', linestyle='--')  # 绘制诊室排队情况的曲线，使用虚线

    # 设置x轴标签为“Time (minutes)”
    plt.xlabel('Time (minutes)')

    # 设置y轴标签为“Number of people”
    plt.ylabel('Number of people')

    # 设置图表标题
    plt.title('Usage and Queue Length Over Time for Each Room')

    # 显示图例，帮助识别不同的曲线
    plt.legend()

    # 显示图表
    plt.show()

    # 绘制利用率的图表
    plot_utilization(rooms)

'''
模拟了一个医院环境，其中包含了病人的到达、病人按指定流程在不同诊室的就诊处理过程。病人的到达频率和诊室信息从外部文件中读取。以下是完整的代码解释，包括每行代码的注释和主程序逻辑的详细说明。

### 主程序逻辑

1. **环境初始化**：设置一个SimPy仿真环境。
2. **数据加载**：
   - 从YAML文件加载诊室和病人数据。
   - 从Excel文件加载病人每个半小时的到达频率。
3. **诊室和病人实体创建**：
   - 为每个诊室创建一个`ConsultationRoom`实例。
   - 为每种病人类型安排到达计划，每个病人实例在到达时将启动就诊过程。
4. **仿真执行**：运行仿真，直到一整天结束（1440分钟）。
5. **结果输出**：在控制台输出病人的处理信息，显示每个病人在哪个诊室被处理以及对应的时间。

'''



def main():
    # 指定文件路径
    patients_file = 'patients.yaml'
    rooms_file = 'rooms.yaml'
    capacities_file = 'Room_Capacities.xlsx'  # 添加这行来指定诊室容量文件路径
    frequencies_file = 'Patient_Arrival_Frequencies.xlsx'

    # 调用仿真函数，增加传递 capacities_file 参数
    simulation(patients_file, rooms_file, capacities_file, frequencies_file)

if __name__ == "__main__":
    main()