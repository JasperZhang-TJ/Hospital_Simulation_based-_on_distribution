import yaml
import numpy as np
from scipy.special import factorial, expit
from math import factorial
from scipy.special import factorial as sp_factorial

#单位说明 以下的 mu（科室处理效率）， lemda（到达率） 单位都为 人/分钟

# 加载 rooms 数据
def load_rooms_from_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    rooms = [item['name'] for item in data if 'name' in item]
    return rooms

# 根据 rooms.yaml 文件内容初始化参数的函数
def initialize_parameters(file_path):
    # 从 yaml 文件中加载数据
    with open(file_path, 'r', encoding='utf-8') as file:
        rooms_data = yaml.safe_load(file)

    # 初始化字典
    pi_dict = {}  # 存储堵塞概率的字典
    mu_dict = {}  # 存储有效服务率的字典
    s_dict = {}   # 存储有效处理台数的字典

    # 用 yaml 文件中的初始值填充字典
    for room in rooms_data:
        room_name = room['name']
        pi_dict[room_name] = 0  # 将堵塞概率初始化为 0
        efficiency = room['efficiency']['A']  # 由于各病人类型的效率相同，我们只取 'A' 的效率
        mu_dict[room_name] = 30 / efficiency  # 计算服务率 mu
        s_dict[room_name] = room['capacity']  # 设置有效服务器的数量

    return pi_dict, mu_dict, s_dict
# 假设 'rooms.yaml' 文件位于脚本的同一目录下
# 你可以像这样调用函数：
# pi, mu, s = initialize_parameters('rooms.yaml所在的路径')

# 加载 patients.yaml 文件以获取患者类型的函数
def load_patient_types(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        patient_data = yaml.safe_load(file)
    # 返回患者类型列表
    return [patient['type'] for patient in patient_data]

# 假设 patients.yaml 文件位于脚本的同一目录下
# patient_types = load_patient_types('patients.yaml所在的路径')

# 接下来，模拟询问每种患者类型的进入频率的函数
def get_entry_rates_for_patient_types(patient_types):
    entry_rates = {}
    for patient_type in patient_types:
        # 这里替换为实际询问用户的逻辑
        print(f"请输入 {patient_type} 类患者的进入频率（人/分钟）:")
        entry_rate = float(input())
        entry_rates[patient_type] = entry_rate
    return entry_rates

# 使用上面的函数，你可以按照如下方式调用它们：
# patient_types = load_patient_types('patients.yaml所在的路径')
# entry_rates = get_entry_rates_for_patient_types(patient_types)

# 读取 patients.yaml 文件，并构建每个科室的紧前科室和紧后科室列表的函数
def build_adjacency_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        patient_types = yaml.safe_load(file)

    # 初始化科室的紧前科室和紧后科室列表
    preceding_rooms = {}  # 紧前科室
    following_rooms = {}  # 紧后科室，包括一个特别条目 "outside"

    # 初始化 "outside" 键，用于记录所有从外部进入的科室
    following_rooms['outside'] = []

    # 遍历每个病人类型的就诊顺序，构建紧前和紧后科室的列表
    for patient in patient_types:
        sequence = patient['consultation_sequence']
        for i, room in enumerate(sequence):
            # 如果当前科室不在列表中，则初始化空列表
            preceding_rooms.setdefault(room, [])
            following_rooms.setdefault(room, [])

            # 如果是就诊序列的第一个科室，添加一个特殊的紧前科室“outside”
            if i == 0:
                preceding_rooms[room].append('outside')
                following_rooms['outside'].append(room)  # 将这个科室添加到从 "outside" 进入的列表中

            # 对于序列中的每个科室，除了第一个之外的科室
            # 将前一个科室添加到紧前科室列表，将后一个科室添加到紧后科室列表
            if i > 0:
                preceding_rooms[room].append(sequence[i - 1])
                following_rooms[sequence[i - 1]].append(room)

    # 去除重复的科室
    for room in preceding_rooms:
        preceding_rooms[room] = list(set(preceding_rooms[room]))
    for key in following_rooms:
        following_rooms[key] = list(set(following_rooms[key]))

    return preceding_rooms, following_rooms
# 假设 'patients.yaml' 文件位于脚本的同一目录下
# 你可以像这样调用函数：
# prec_rooms, foll_rooms = build_adjacency_list('patients.yaml所在的路径')

# 读取 patients.yaml 文件并提取患者类型和其就诊序列的函数
def get_patient_consultation_sequences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        patient_data = yaml.safe_load(file)
    # 创建一个字典，其键为患者类型，值为对应的就诊序列
    patient_consultation_sequences = {
        patient['type']: patient['consultation_sequence'] for patient in patient_data
    }
    return patient_consultation_sequences
# 你可以像这样调用上面的函数来获取患者类型和就诊序列：
# patient_consultation_sequences = get_patient_consultation_sequences('patients.yaml所在的路径')

#计算根据输入的流入频率各个科室之间的累计转移人数
def calculate_flow_rates(entry_rates, following_rooms, patient_consultation_sequences):
    """
    根据每种患者类型的进入频率和就诊序列，计算每个科室间的流量。
    :param entry_rates: 每种患者类型的进入频率（人/分钟）。
    :param following_rooms: 各科室的紧后科室列表，包括“outside”。
    :param patient_consultation_sequences: 每种患者类型的就诊序列。
    :return: 含有科室间转移流量的字典。
    """
    # 初始化流量字典，包含所有科室和“outside”
    flow_rates = {room: {} for room in following_rooms}

    # 遍历每种患者类型及其就诊序列
    for patient_type, sequence in patient_consultation_sequences.items():
        # 获取这种患者类型的进入频率
        rate = entry_rates[patient_type]

        # 处理从“outside”进入系统的流量
        first_room = sequence[0]
        flow_rates['outside'].setdefault(first_room, 0)
        flow_rates['outside'][first_room] += rate

        # 遍历序列中的科室，计算每个科室间的流量
        for i, room in enumerate(sequence[:-1]):  # 排除序列中的最后一个科室
            next_room = sequence[i + 1]
            flow_rates[room].setdefault(next_room, 0)
            flow_rates[room][next_room] += rate

    return flow_rates
#标准化转移人数到V
def normalize_flow_rates(flow_rates):
    """
    标准化流量字典中的值，使得每个科室的流量为比例值。
    :param flow_rates: 含有科室间流量信息的字典。
    :return: 标准化后的流量字典。
    """
    normalized_flow_rates = {}

    for room, transfers in flow_rates.items():
        # 计算总流量
        total_flow = sum(transfers.values())
        normalized_transfers = {next_room: flow / total_flow for next_room, flow in transfers.items()}
        normalized_flow_rates[room] = normalized_transfers

    return normalized_flow_rates
# 假设 flow_rates 字典已经被计算出来了
# normalized_v = normalize_flow_rates(flow_rates

# 建立解方程空值表
def setup_flow_matrix(following_rooms):
    """
    根据 following_rooms 字典，建立所有有值的 F_i,j 的初始空矩阵。
    :param following_rooms: 各科室的紧后科室列表，包括“outside”。
    :return: 初始化后的流率矩阵 F。
    """
    F = {}
    for room, next_rooms in following_rooms.items():
        F[room] = {next_room: None for next_room in next_rooms}
    return F
# 式子15 初始化outside对应的流量
def update_outside_flows(F, flow_rates, pi):
    """
    更新所有从 'outside' 开始的流率 F['outside'][i]。
    :param F: 初始化后的流率矩阵 F。
    :param flow_rates: 每个科室的流入频率。
    :param pi: 各科室的堵塞概率。
    """
    for room in flow_rates['outside']:
        # 假设流入频率等于从 'outside' 到科室的流量
        lambda_0_i = flow_rates['outside'][room]
        F['outside'][room] = lambda_0_i * (1 - pi[room])

#解方程的函数（14,16）
def build_balance_equations(preceding_rooms, following_rooms, F, V, pi):
    # 确定需要方程的科室（即有射入也有射出的科室）
    rooms_with_equations = [room for room in following_rooms if room != 'outside' and following_rooms[room]!=[] and preceding_rooms.get(room)]
    #print(rooms_with_equations)

    # 为这些科室创建平衡方程
    num_equations = len(rooms_with_equations)
    num_variables = num_equations  # 每个科室一个变量

    # 初始化系数矩阵和常数向量
    A = np.zeros((num_equations, num_variables))
    b = np.zeros(num_equations)

    # 创建变量索引的映射
    variable_indices = {room: i for i, room in enumerate(rooms_with_equations)}

    # 构建每个科室的平衡方程
    for room in rooms_with_equations:
        room_idx = variable_indices[room]
        # 射出流量，由x[i]决定
        A[room_idx, room_idx] = sum(V[room].values())
        # 射入流量，累加到常数项b中
        for prev_room in preceding_rooms[room]:
            if prev_room != 'outside':
                A[room_idx, variable_indices[prev_room]] = -V[prev_room][room]  # 注意这里是负号
            else:
                # 如果前一个房间是外部，使用已知的流量F[prev_room][room]
                b[room_idx] = F[prev_room][room]

    return A, b, rooms_with_equations
# 使用NumPy的linalg.solve来解方程
def solve_flow_balances(A, b):
    x = np.linalg.solve(A, b)
    return x

#------------------------------循环（c）---------------------------------
# 计算 L(λ, μ, s) 的函数
def compute_L(lemda, mu, s):
    rho = lemda / (s * mu)
    rho_0= lemda / mu
    sum_terms = sum([ (rho_0 ** n) / factorial(n)  for n in range(s)])
    term_1=((s**s)*rho**(s+1))/(factorial(s)*((1-rho)**2))
    term_2=(sum_terms+(rho_0**s/(factorial(s)*(1-rho))))**(-1)
    return term_1+term_2

# 计算 L_i,j(F, μ, s) 的函数
def compute_L_ij(F, mu, s, i, j, rooms):
    # 计算所有射入科室j的流量之和
    lemda_j = sum(F[k].get(j, 0) for k in rooms if k != j)  # 使用 dict.get 避免 KeyError
    if lemda_j == 0:
        return 0  # 避免除以0的情况
    L_ij = F[i].get(j, 0) / lemda_j * compute_L(lemda_j, mu[j], s[j])
    return L_ij
# 根据式子（18），计算更新的s_i值
def update_effective_servers(F, mu, s, rooms):
    # 初始化有效服务器数量的字典
    s_star = {}
    for i in rooms:
        if i != 'outside':  # 跳过外部节点
            # 计算每个科室的 L_i,j(F, μ, s) 总和
            L_i_sum = sum(compute_L_ij(F, mu, s, i, j, rooms) for j in rooms if j != 'outside' and j in F[i])
            # 使用公式(18)更新 s^*_i
            print(s[i] - L_i_sum,i,s[i],L_i_sum)
            s_star[i] = int(max(s[i] - L_i_sum, 0)) # 注意这里使用 max 函数保证 s^*_i 不会小于0
    return s_star
# 根据式子（19），计算更新的μ_i值
def update_effective_service_rate(F, mu, s, s_star, rooms):
    mu_star = {}
    for i in rooms:
        if i != 'outside':  # 跳过外部节点
            # 初始化求和部分
            sum_part = 0
            for j in rooms:
                if j != 'outside' and j in F[i]:
                    # 检查以避免除以零
                    total_F_j = sum(F[k][j] for k in rooms if k != j)
                    if s[i] > 0 and total_F_j > 0 and s[j] > 0 and mu[j] > 0 and F[i][j]>0:
                        L_ij = compute_L_ij(F, mu, s_star, i, j, rooms)
                        sum_part += L_ij / (s[i] / (F[i][j] / total_F_j * s[j] * mu[j]))
                    else:
                        # 可以选择跳过或设定默认值
                        sum_part += 0

            # 使用公式(19)更新 mu_i*
            if s[i] > 0:
                mu_star[i] = ((s_star[i] / s[i]) * (1 / mu[i]) + sum_part) ** (-1)
            else:
                mu_star[i] = mu[i]  # 若s[i]为0，暂且保持原mu值不变，或根据需要进行处理

    return mu_star

#----------------------------循环（d）--------------------------------
# 计算稳态概率 π^C(λ, μ, s) 对于外部输入的拥堵概率--（1）
def pi_C(lam, mu, s):
    sum_terms = sum([(lam / mu)**j / sp_factorial(j) for j in range(s+1)])
    if (lam / mu)**s / sp_factorial(s) / sum_terms >1:
        return 1
    if (lam / mu) ** s / sp_factorial(s) / sum_terms < 0:
        return 0
    return (lam / mu)**s / sp_factorial(s) / sum_terms
# 计算稳态概率 π^UC(λ, μ, s) 对于系统内部的拥堵概率--（20）
def pi_UC(lam, mu, s):
    sum_terms = sum([(lam / mu)**n / sp_factorial(n) for n in range(s)])
    uc_terms_1= (lam /(mu*s))
    uc_terms_2= s**s / sp_factorial(s)
    if ((uc_terms_2 * uc_terms_1**s)/(1-uc_terms_1)) * (sum_terms + ((uc_terms_2 * uc_terms_1**s)/(1-uc_terms_1)))**-1 >1:
        return 1
    if ((uc_terms_2 * uc_terms_1**s)/(1-uc_terms_1)) * (sum_terms + ((uc_terms_2 * uc_terms_1**s)/(1-uc_terms_1)))**-1 <0:
        return 0
    return ((uc_terms_2 * uc_terms_1**s)/(1-uc_terms_1)) * (sum_terms + ((uc_terms_2 * uc_terms_1**s)/(1-uc_terms_1)))**-1
# 根据科室i计算稳态概率 π_i(λ, μ, s)
def pi_i(lam, mu, s, i, F, rooms):
    # 计算所有射入科室j的流量之和
    lemda_i = sum(F[k].get(i, 0) for k in rooms if k != i)  # 使用 dict.get 避免 KeyError
    lemda_i_0 = sum(F[k].get(i, 0) for k in rooms if (k != i and k != 'outside'))  # 使用 dict.get 避免 KeyError
    if lemda_i==0:
        print(i,"这个诊室没有进入")

    return (F['outside'][i] / lemda_i) * pi_C(lam, mu, s) + (lemda_i_0 / lemda_i) * pi_UC(lam, mu, s)


def main():
    # 这个阈值应该是一个较小的数字，用来判断算法是否收敛
    delta = 0.00001

    # 初始化 m 为 0，表示第一次迭代
    m = 0

    # 从rooms.yaml中读取科室列表
    rooms = load_rooms_from_yaml('rooms.yaml')

    rooms = rooms+['outside']

    #print(rooms)

    # 初始化完成的字典
    pi, mu, s = initialize_parameters('rooms.yaml')
    #print(pi)
    #print(mu)
    print(s)

    # 用于构建每一个科室的紧前科室（indegree），紧后科室（outdegree）
    preceding_rooms, following_rooms = build_adjacency_list("patients.yaml")
    preceding_rooms = {k: v for k, v in preceding_rooms.items() if v}
    following_rooms = {k: v for k, v in following_rooms.items() if v}
    #print(following_rooms)
    # 生成网络内部的分配流结构
    patient_types = load_patient_types('patients.yaml')
    entry_rates = get_entry_rates_for_patient_types(patient_types)
    #print(entry_rates)
    patient_consultation_sequences = get_patient_consultation_sequences('patients.yaml')
    flow_rates = calculate_flow_rates(entry_rates, following_rooms, patient_consultation_sequences)
    #print(flow_rates)
    V = normalize_flow_rates(flow_rates) #V 是 v_{i,j}的集合 v_{i,j}表示从i出发 到j的人数比例 比如'ultrasound': {'routine check-up': 0.5, 'fetal heart rate': 0.5} 代表 从 ultrasound 到 fetal heart rate 的人数比例为0.5
    #print(V)



    # 迭代过程开始

    while True:
        m += 1  # 增加迭代次数的索引

            # ... 这里放置解算方程 (14), (15) 并计算流率 Fm 的代码 ...
        #以下是循环解方程的部分
        F = setup_flow_matrix(following_rooms)  # 初始化流量矩阵 F
        update_outside_flows(F, flow_rates, pi)    #更新outside流入的流量
        # 初始化平衡方程
        A, b, rooms_with_equations = build_balance_equations(preceding_rooms, following_rooms, F, V, pi)
        # 解方程
        outflows = solve_flow_balances(A, b)
        # 使用解更新流量矩阵 F
        for i, room in enumerate(rooms_with_equations):
            outflow = outflows[i]  # 该房间的总射出流量
            for next_room in following_rooms[room]:
                if next_room != 'outside':  # 仅更新非 'outside' 房间的流量
                    F[room][next_room] = outflow * V[room][next_room]
        # 补全 F 矩阵中的缺失条目
        for room in rooms + ['outside']:  # 包括 'outside' 以确保从外部的流入也被考虑
            if room not in F:
                F[room] = {}
            for next_room in rooms + ['outside']:  # 确保每个科室都有到每个科室包括自己的条目
                if next_room not in F[room]:
                    F[room][next_room] = 0  # 未指定的流量设置为0
        # 现在 F 是完全初始化的，可以安全地进行后续的计算和更新


        # 对每个科室进行更新
        # ... 使用 (17), (18), (19) 更新有效服务器的数量 sm 和相应的有效服务率 mum ...
        # 更新有效服务器的数量 s_star 和相应的有效服务率 mu_star
        s_star = update_effective_servers(F, mu, s, rooms)
        mu_star = update_effective_service_rate(F, mu, s, s_star, rooms)
        # 更新原始的s和mu
        s.update(s_star)
        mu.update(mu_star)
        #print(s)
        #print(mu)


        previous_pi=pi

        # ... 更新每个科室的堵塞概率 pi ...
        for room in rooms:
            if room !='outside':
                lam = sum(F[k].get(room, 0) for k in rooms if k != room)
                pi[room]= pi_i(lam, mu[room], s[room], room, F, rooms)

        # 计算所有科室的堵塞概率之和的变化是否小于 delta
        pi_change = sum(abs(pi[room_name] - previous_pi[room_name]) for room_name in pi)
        if pi_change < delta and m>2:
            print(pi)
            print(m)
            break  # 如果变化小于阈值，则认为算法已经收敛，结束循环


if __name__ == '__main__':
    main()